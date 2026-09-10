"""Summariser agent: assemble the human-facing brief.

The template path is the default and produces a summary composed entirely of
values other agents already extracted and validated - so it cannot hallucinate,
because it never generates a claim, it only arranges verified ones.

When a model is configured it may write the narrative instead, but the output
is then screened: PHI leaks and instruction-shaped payloads reject it, and the
template summary is used instead. A summary is the artefact a clinician
actually reads, which makes it the worst place in the pipeline for an
unverified sentence.
"""
from __future__ import annotations

from ..guardrails import check_output, neutralize
from ..llm import ProviderError
from ..types import AgentResult, Issue, Severity, WorkflowState
from .base import Agent

#: Order matters: this is reading order for a clinician, not schema order.
SUMMARY_FIELDS = [
    ("patient_name", "Patient"),
    ("date_of_birth", "DOB"),
    ("mrn", "MRN"),
    ("primary_diagnosis", "Diagnosis"),
    ("chief_complaint", "Chief complaint"),
    ("reason_for_referral", "Reason for referral"),
    ("requested_service", "Requested service"),
    ("admission_date", "Admitted"),
    ("discharge_date", "Discharged"),
    ("ordering_provider", "Provider"),
]

VITALS_FIELDS = [
    ("blood_pressure", "BP"),
    ("heart_rate", "HR"),
    ("temperature", "Temp"),
    ("respiratory_rate", "RR"),
    ("oxygen_saturation", "SpO2"),
]

LAB_FIELDS = [
    ("hemoglobin", "Hgb"),
    ("creatinine", "Cr"),
    ("glucose", "Glucose"),
    ("hba1c", "HbA1c"),
    ("potassium", "K"),
    ("wbc", "WBC"),
    ("platelets", "Plt"),
]


class SummarizerAgent(Agent):
    """Produces the reviewer-facing summary."""

    name = "summarizer"
    critical = False

    def run(self, state: WorkflowState) -> AgentResult:
        template = self._template_summary(state)
        issues: list[Issue] = []
        summary = template
        source = "template"

        if self._model_available():
            narrative, model_issues = self._model_summary(state)
            issues.extend(model_issues)
            if narrative:
                summary = narrative
                source = "model"

        leak_issues = check_output(summary, state.document.text, agent=self.name)
        if leak_issues and source == "model":
            # The model produced something unsafe; fall back rather than ship it.
            issues.extend(leak_issues)
            issues.append(
                Issue(
                    code="CF-SUM-003",
                    severity=Severity.WARNING,
                    message="model summary failed output screening; using the template summary",
                    agent=self.name,
                )
            )
            summary = template
            source = "template"
        else:
            issues.extend(leak_issues)

        return AgentResult(
            agent=self.name,
            issues=issues,
            # A template summary is certain because it only restates verified
            # fields; a model narrative is not.
            confidence=1.0 if source == "template" else 0.7,
            output={
                "summary": summary,
                "source": source,
                "citations": self._citations(state),
            },
        )

    def _template_summary(self, state: WorkflowState) -> str:
        lines: list[str] = [
            f"{state.document.doc_type.value.replace('_', ' ').title()}"
            f" ({state.document.doc_id})",
            "",
        ]

        for name, label in SUMMARY_FIELDS:
            extracted = state.fields.get(name)
            if extracted is not None:
                lines.append(f"{label}: {extracted.value}")

        vitals = [
            f"{label} {state.fields[name].value}"
            f"{' ' + (state.fields[name].unit or '') if state.fields[name].unit else ''}"
            for name, label in VITALS_FIELDS
            if name in state.fields
        ]
        if vitals:
            lines += ["", "Vitals: " + ", ".join(vitals)]

        labs = [
            f"{label} {state.fields[name].value}"
            f"{' ' + (state.fields[name].unit or '') if state.fields[name].unit else ''}"
            for name, label in LAB_FIELDS
            if name in state.fields
        ]
        if labs:
            lines += ["", "Labs: " + ", ".join(labs)]

        medications = state.fields.get("medications")
        if medications is not None and isinstance(medications.value, list):
            lines += ["", "Medications:"]
            for entry in medications.value:
                if not isinstance(entry, dict):
                    continue
                frequency = f" {entry['frequency']}" if entry.get("frequency") else ""
                lines.append(f"  - {entry.get('drug')} {entry.get('dose')}{entry.get('unit')}{frequency}")

        allergies = state.fields.get("allergies")
        if allergies is not None:
            value = allergies.value
            rendered = ", ".join(str(a) for a in value) if isinstance(value, list) and value else "none documented"
            lines += ["", f"Allergies: {rendered}"]

        flagged = [i for i in state.issues if i.severity >= Severity.WARNING]
        if flagged:
            lines += ["", "Flagged for review:"]
            for issue in flagged:
                lines.append(f"  [{issue.severity.value}] {issue.message}")

        citations = self._citations(state)
        if citations:
            lines += ["", "Policy context: " + ", ".join(citations)]

        return "\n".join(lines)

    def _citations(self, state: WorkflowState) -> list[str]:
        retrieval = state.result_for("retriever")
        if retrieval is None:
            return []
        return list(retrieval.output.get("citations", []))

    def _model_summary(self, state: WorkflowState) -> tuple[str | None, list[Issue]]:
        """Ask the model to write the narrative from verified fields only.

        The document text is not in this prompt. The model summarises the
        extracted fields, which have already been grounded and validated, so
        there is nothing new for it to invent and no injected instruction for
        it to read.
        """
        facts = "\n".join(
            f"- {name}: {extracted.value}"
            for name, extracted in sorted(state.fields.items())
        )
        flagged = "\n".join(
            f"- [{i.severity.value}] {i.message}"
            for i in state.issues
            if i.severity >= Severity.WARNING
        )
        prompt = (
            "Write a concise clinical summary (at most 6 sentences) for a "
            "reviewing clinician, using ONLY the verified fields below. Do not "
            "add any clinical detail that is not listed. Do not speculate.\n\n"
            f"Verified fields:\n{neutralize(facts)}\n\n"
            f"Flagged issues:\n{neutralize(flagged) or '- none'}\n"
        )
        system = (
            "You write factual clinical summaries from pre-verified structured "
            "data. You never introduce facts that are not in the input."
        )
        try:
            completion = self.provider.complete(prompt, system=system, max_tokens=400)
        except ProviderError as exc:
            return None, [
                Issue(
                    code="CF-SUM-001",
                    severity=Severity.INFO,
                    message=f"model summariser unavailable, using template: {exc}",
                    agent=self.name,
                )
            ]

        text = completion.text.strip()
        if not text:
            return None, [
                Issue(
                    code="CF-SUM-002",
                    severity=Severity.WARNING,
                    message="model returned an empty summary; using the template",
                    agent=self.name,
                )
            ]
        return text, []
