"""Router agent: decide what kind of document arrived, and where it goes.

Classification is scored rather than first-match. Clinical documents overlap
heavily - a discharge summary contains lab values, a referral quotes a note -
so the router weighs several signals and reports the margin between the top
two candidates. A narrow margin is itself information: it routes to human
review instead of guessing.
"""
from __future__ import annotations

import re

from ..types import AgentResult, DocumentType, Issue, Severity, WorkflowState
from .base import Agent

# (pattern, weight) per document type. Section headers score higher than
# incidental vocabulary, because a header is structural evidence.
SIGNALS: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.REFERRAL: [
        (r"(?i)\breferr(al|ing)\b", 3.0),
        (r"(?i)\breason\s+for\s+referral\b", 5.0),
        (r"(?i)\brefer(red|ring)\s+(to|physician|provider)\b", 3.0),
        (r"(?i)\bconsult(ation)?\s+request\b", 4.0),
        (r"(?i)\brequesting\s+(provider|physician)\b", 2.5),
    ],
    DocumentType.DISCHARGE_SUMMARY: [
        (r"(?i)\bdischarge\s+summary\b", 6.0),
        (r"(?i)\bdischarge\s+(date|diagnosis|medications|instructions)\b", 4.0),
        (r"(?i)\bhospital\s+course\b", 5.0),
        (r"(?i)\badmi(ssion|tted)\s+(date|on)\b", 2.5),
        (r"(?i)\bfollow[-\s]?up\s+appointment\b", 1.5),
    ],
    DocumentType.LAB_REPORT: [
        (r"(?i)\b(laboratory|lab)\s+(report|results?)\b", 5.0),
        (r"(?i)\breference\s+range\b", 4.5),
        (r"(?i)\b(specimen|collected)\s*(id|date|:)", 3.0),
        (r"(?i)\b(hemoglobin|creatinine|glucose|wbc|platelet|hba1c|tsh)\b", 2.0),
        (r"(?i)\b(mg/dL|mmol/L|g/dL|mEq/L|IU/L)\b", 2.0),
    ],
    DocumentType.PRIOR_AUTHORIZATION: [
        (r"(?i)\bprior\s+auth(oriz|oris)ation\b", 6.0),
        (r"(?i)\b(pre[-\s]?certification|pre[-\s]?approval)\b", 4.0),
        (r"(?i)\b(member|policy)\s+id\b", 2.5),
        (r"(?i)\brequested\s+(service|procedure|medication)\b", 4.0),
        (r"(?i)\b(cpt|hcpcs)\s*(code)?\b", 3.0),
        (r"(?i)\bmedical\s+necessity\b", 3.5),
    ],
    DocumentType.IMAGING_REPORT: [
        (r"(?i)\b(radiology|imaging)\s+report\b", 6.0),
        (r"(?i)\b(impression|findings)\s*:", 3.0),
        (r"(?i)\b(ct|mri|x[-\s]?ray|ultrasound|echocardiogram)\b", 3.0),
        (r"(?i)\bcontrast\b", 1.5),
    ],
    DocumentType.CLINICAL_NOTE: [
        (r"(?i)\b(progress|clinical|soap)\s+note\b", 5.0),
        (r"(?i)\b(subjective|objective|assessment\s+and\s+plan)\s*:", 3.5),
        (r"(?i)\bchief\s+complaint\b", 4.0),
        (r"(?i)\bhistory\s+of\s+present\s+illness\b", 4.0),
    ],
}

_COMPILED = {
    doc_type: [(re.compile(pattern), weight) for pattern, weight in signals]
    for doc_type, signals in SIGNALS.items()
}

#: Downstream agents to invoke per document type. The orchestrator reads this,
#: so adding a document type does not mean editing the pipeline.
ROUTES: dict[DocumentType, list[str]] = {
    DocumentType.REFERRAL: ["retriever", "extractor", "validator", "summarizer"],
    DocumentType.DISCHARGE_SUMMARY: ["retriever", "extractor", "validator", "summarizer"],
    DocumentType.LAB_REPORT: ["extractor", "validator"],
    DocumentType.PRIOR_AUTHORIZATION: ["retriever", "extractor", "validator", "summarizer"],
    DocumentType.IMAGING_REPORT: ["extractor", "validator", "summarizer"],
    DocumentType.CLINICAL_NOTE: ["retriever", "extractor", "validator", "summarizer"],
    DocumentType.UNKNOWN: ["extractor", "validator"],
}


def score_document(text: str) -> dict[DocumentType, float]:
    """Weighted signal score per document type, highest first."""
    scores: dict[DocumentType, float] = {}
    for doc_type, patterns in _COMPILED.items():
        total = 0.0
        for pattern, weight in patterns:
            hits = len(pattern.findall(text))
            if hits:
                # Diminishing returns: the second "reference range" adds less
                # than the first, so a long document cannot win on volume.
                total += weight * (1 + 0.3 * (hits - 1))
        if total:
            scores[doc_type] = round(total, 2)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


class RouterAgent(Agent):
    """Classifies the document and selects the downstream agent path."""

    name = "router"
    critical = True

    #: Below this score the classification is not trustworthy at all.
    min_score = 4.0
    #: Below this ratio between the top two candidates the document is
    #: genuinely ambiguous and a human should look at it.
    min_margin = 1.4

    def run(self, state: WorkflowState) -> AgentResult:
        scores = score_document(state.document.text)
        issues: list[Issue] = []

        if not scores:
            state.document.doc_type = DocumentType.UNKNOWN
            issues.append(
                Issue(
                    code="CF-ROUTE-001",
                    severity=Severity.WARNING,
                    message="no document-type signals found; using the generic path",
                    agent=self.name,
                )
            )
            return AgentResult(
                agent=self.name,
                confidence=0.0,
                issues=issues,
                output={
                    "doc_type": DocumentType.UNKNOWN.value,
                    "route": ROUTES[DocumentType.UNKNOWN],
                    "scores": {},
                },
            )

        ranked = list(scores.items())
        top_type, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score / runner_up if runner_up else float("inf")

        if top_score < self.min_score:
            issues.append(
                Issue(
                    code="CF-ROUTE-002",
                    severity=Severity.WARNING,
                    message=(
                        f"weak classification signal for {top_type.value} "
                        f"(score {top_score:.1f} below {self.min_score})"
                    ),
                    agent=self.name,
                    detail={"scores": {k.value: v for k, v in scores.items()}},
                )
            )
        elif margin < self.min_margin:
            issues.append(
                Issue(
                    code="CF-ROUTE-003",
                    severity=Severity.WARNING,
                    message=(
                        f"ambiguous document type: {top_type.value} vs "
                        f"{ranked[1][0].value} (margin {margin:.2f})"
                    ),
                    agent=self.name,
                    detail={"scores": {k.value: v for k, v in scores.items()}},
                )
            )

        state.document.doc_type = top_type
        # Confidence blends absolute evidence with separation from the runner
        # up: a strong score that barely beats another type is not confident.
        confidence = min(top_score / 12.0, 1.0) * min(margin / 2.0, 1.0)

        return AgentResult(
            agent=self.name,
            confidence=round(confidence, 3),
            issues=issues,
            output={
                "doc_type": top_type.value,
                "route": ROUTES[top_type],
                "scores": {k.value: v for k, v in scores.items()},
                "margin": round(margin, 2) if margin != float("inf") else None,
            },
        )
