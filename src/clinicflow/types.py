"""Core types for the workflow.

Two ideas run through all of these:

**Provenance is not optional.** Every extracted field carries the span of
source text it came from. In a clinical setting, a value a reviewer cannot
trace back to the document is worse than no value — it looks authoritative
while being unverifiable. :class:`Field` therefore cannot be constructed
without a source, and the validator rejects any field whose quoted evidence
does not appear in the document.

**Confidence is carried, not collapsed.** Agents report how sure they are, and
the orchestrator routes low-confidence output to human review rather than
silently accepting it.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    REFERRAL = "referral"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_REPORT = "lab_report"
    PRIOR_AUTHORIZATION = "prior_authorization"
    CLINICAL_NOTE = "clinical_note"
    IMAGING_REPORT = "imaging_report"
    UNKNOWN = "unknown"


class Decision(str, Enum):
    """What the workflow concluded about a document."""

    COMPLETE = "complete"
    #: Processed, but a human must sign off before the output is used.
    NEEDS_REVIEW = "needs_review"
    #: Could not be processed safely; nothing downstream should consume it.
    REJECTED = "rejected"
    FAILED = "failed"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2, "critical": 3}[self.value]

    # Defined explicitly: this enum subclasses str, so without them the
    # comparisons fall back to lexical string ordering.
    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        return self.rank < other.rank if isinstance(other, Severity) else NotImplemented

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        return self.rank <= other.rank if isinstance(other, Severity) else NotImplemented

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        return self.rank > other.rank if isinstance(other, Severity) else NotImplemented

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        return self.rank >= other.rank if isinstance(other, Severity) else NotImplemented


@dataclass(frozen=True)
class Field:
    """One extracted value, with the evidence that produced it."""

    name: str
    value: Any
    confidence: float
    #: Character span in the source document.
    span: tuple[int, int]
    #: The exact source text. Checked against the document by the validator.
    evidence: str
    unit: str | None = None
    extractor: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "confidence": round(self.confidence, 3),
            "span": list(self.span),
            "evidence": self.evidence,
            "extractor": self.extractor,
        }


@dataclass
class Document:
    """A clinical document entering the workflow."""

    text: str
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "upload"
    doc_type: DocumentType = DocumentType.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def excerpt(self, span: tuple[int, int], pad: int = 0) -> str:
        start = max(span[0] - pad, 0)
        end = min(span[1] + pad, len(self.text))
        return self.text[start:end]


@dataclass
class Issue:
    """Something an agent noticed that a human may need to act on."""

    code: str
    severity: Severity
    message: str
    agent: str
    field_name: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    #: Whether this issue means the document cannot be processed at all, as
    #: opposed to being processed and flagged.
    #:
    #: Severity and blocking are deliberately independent. A potassium of 6.4
    #: is CRITICAL and must reach a clinician urgently - but it is a real
    #: finding, correctly extracted, and rejecting the document would throw
    #: away the very result that needs acting on. An injected instruction or a
    #: fabricated citation is also CRITICAL, and there the output cannot be
    #: trusted at all. Only the second kind is blocking.
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "agent": self.agent,
            "field": self.field_name,
            "blocking": self.blocking,
            "detail": self.detail,
        }


@dataclass
class AgentResult:
    """What one agent produced for one document."""

    agent: str
    ok: bool = True
    fields: list[Field] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def max_severity(self) -> Severity:
        return max((i.severity for i in self.issues), default=Severity.INFO)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "ok": self.ok,
            "confidence": round(self.confidence, 3),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "fields": [f.to_dict() for f in self.fields],
            "issues": [i.to_dict() for i in self.issues],
            "output": self.output,
        }


@dataclass
class WorkflowState:
    """Everything known about a document as it moves between agents.

    Passed by reference through the pipeline; each agent reads what it needs
    and appends its own result. Keeping one mutable state object (rather than
    threading tuples) is what makes the audit trail complete: whatever the
    workflow knew at the end is exactly what is recorded.
    """

    document: Document
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    results: list[AgentResult] = field(default_factory=list)
    fields: dict[str, Field] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    decision: Decision = Decision.COMPLETE
    review_reasons: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_result(self, result: AgentResult) -> None:
        self.results.append(result)
        self.issues.extend(result.issues)
        for extracted in result.fields:
            existing = self.fields.get(extracted.name)
            # Higher-confidence extraction wins; ties keep the first, so a
            # deterministic rule beats a later probabilistic guess.
            if existing is None or extracted.confidence > existing.confidence:
                self.fields[extracted.name] = extracted
        self.trace.append(
            {
                "agent": result.agent,
                "ok": result.ok,
                "confidence": round(result.confidence, 3),
                "elapsed_ms": round(result.elapsed_ms, 2),
                "fields": len(result.fields),
                "issues": len(result.issues),
                "error": result.error,
            }
        )

    def require_review(self, reason: str) -> None:
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)
        if self.decision is Decision.COMPLETE:
            self.decision = Decision.NEEDS_REVIEW

    @property
    def max_severity(self) -> Severity:
        return max((i.severity for i in self.issues), default=Severity.INFO)

    @property
    def duration_ms(self) -> float:
        end = self.finished_at or time.time()
        return (end - self.started_at) * 1000

    @property
    def mean_confidence(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.confidence for f in self.fields.values()) / len(self.fields)

    def result_for(self, agent: str) -> AgentResult | None:
        return next((r for r in self.results if r.agent == agent), None)

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "doc_id": self.document.doc_id,
            "doc_type": self.document.doc_type.value,
            "checksum": self.document.checksum,
            "decision": self.decision.value,
            "review_reasons": self.review_reasons,
            "max_severity": self.max_severity.value,
            "mean_confidence": round(self.mean_confidence, 3),
            "duration_ms": round(self.duration_ms, 2),
            "fields": {name: f.to_dict() for name, f in sorted(self.fields.items())},
            "issues": [i.to_dict() for i in self.issues],
            "trace": self.trace,
        }
        if include_text:
            payload["document_text"] = self.document.text
        return payload
