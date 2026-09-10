"""Validator agent: the control that decides whether output is usable.

Three kinds of check, in order of how badly they fail:

1. **Grounding.** Does every field quote text that actually appears in the
   document? A value that cannot be traced is a fabrication, and it stops the
   run - not a warning, a rejection.
2. **Plausibility.** Is the value physiologically possible, and is it inside
   the normal range? These are different questions with different answers: a
   potassium of 6.8 is abnormal and clinically urgent; a potassium of 68 is
   an extraction error. The first is a finding to surface, the second is a bug.
3. **Completeness and consistency.** Are the fields this document type requires
   present, do the dates order correctly, do the medications conflict?

Reference ranges are adult values and deliberately conservative. They are here
to catch extraction errors and flag outliers for a clinician, not to make
clinical decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from ..guardrails import check_grounding
from ..types import AgentResult, DocumentType, Issue, Severity, WorkflowState
from .base import Agent


@dataclass(frozen=True)
class Range:
    """Normal range plus the wider bound of physiological possibility."""

    low: float
    high: float
    #: Outside this, the number is an extraction error rather than a finding.
    impossible_low: float
    impossible_high: float
    unit: str
    #: Outside this inner band, escalate to CRITICAL rather than WARNING.
    critical_low: float | None = None
    critical_high: float | None = None


REFERENCE_RANGES: dict[str, Range] = {
    "heart_rate": Range(60, 100, 20, 250, "bpm", 40, 150),
    "respiratory_rate": Range(12, 20, 4, 60, "breaths/min", 8, 30),
    "temperature": Range(97.0, 99.5, 90.0, 110.0, "F", 95.0, 104.0),
    "oxygen_saturation": Range(95, 100, 50, 100, "%", 88, None),
    "hemoglobin": Range(12.0, 17.5, 2.0, 25.0, "g/dL", 7.0, 20.0),
    "creatinine": Range(0.6, 1.3, 0.1, 20.0, "mg/dL", None, 4.0),
    "glucose": Range(70, 140, 10, 1200, "mg/dL", 50, 400),
    "hba1c": Range(4.0, 5.7, 2.0, 20.0, "%", None, 10.0),
    "potassium": Range(3.5, 5.1, 1.0, 10.0, "mEq/L", 2.8, 6.0),
    "wbc": Range(4.5, 11.0, 0.1, 200.0, "K/uL", 1.0, 30.0),
    "platelets": Range(150, 400, 1, 2000, "K/uL", 50, 1000),
    "weight_kg": Range(40, 150, 0.5, 400, "kg", None, None),
}

#: Fields each document type must have before its output can be trusted.
REQUIRED_FIELDS: dict[DocumentType, list[str]] = {
    DocumentType.REFERRAL: ["patient_name", "reason_for_referral"],
    DocumentType.DISCHARGE_SUMMARY: ["patient_name", "discharge_date", "primary_diagnosis"],
    DocumentType.LAB_REPORT: ["patient_name"],
    DocumentType.PRIOR_AUTHORIZATION: ["patient_name", "requested_service"],
    DocumentType.IMAGING_REPORT: ["patient_name"],
    DocumentType.CLINICAL_NOTE: ["patient_name"],
    DocumentType.UNKNOWN: [],
}

#: Interacting drug classes. A short, defensible list rather than a pretend
#: clinical database - the point is that the check exists and is auditable.
INTERACTIONS: list[tuple[set[str], str, Severity]] = [
    ({"warfarin", "aspirin"}, "additive bleeding risk", Severity.CRITICAL),
    ({"warfarin", "ibuprofen"}, "additive bleeding risk", Severity.CRITICAL),
    ({"warfarin", "amiodarone"}, "amiodarone potentiates warfarin", Severity.CRITICAL),
    ({"lisinopril", "spironolactone"}, "hyperkalemia risk", Severity.WARNING),
    ({"lisinopril", "losartan"}, "duplicate renin-angiotensin blockade", Severity.WARNING),
    ({"simvastatin", "clarithromycin"}, "increased rhabdomyolysis risk", Severity.CRITICAL),
    ({"metformin", "contrast"}, "hold metformin around contrast administration", Severity.WARNING),
    ({"sildenafil", "nitroglycerin"}, "severe hypotension risk", Severity.CRITICAL),
    ({"tramadol", "sertraline"}, "serotonin syndrome risk", Severity.WARNING),
]

#: Plausible dose ceilings, to catch a decimal point in the wrong place.
MAX_DOSE_MG: dict[str, float] = {
    "warfarin": 20, "lisinopril": 80, "metformin": 2550, "amlodipine": 10,
    "atorvastatin": 80, "simvastatin": 40, "sertraline": 200, "furosemide": 600,
    "prednisone": 100, "levothyroxine": 0.5, "digoxin": 0.5,
}

_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y")


def parse_date(value: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


class ValidatorAgent(Agent):
    """Checks extracted output for grounding, plausibility and completeness."""

    name = "validator"
    critical = True

    def run(self, state: WorkflowState) -> AgentResult:
        issues: list[Issue] = []

        if self.guardrails.check_grounding:
            issues.extend(
                check_grounding(
                    [(f.name, f.evidence) for f in state.fields.values()],
                    state.document.text,
                    agent=self.name,
                )
            )

        issues.extend(self._check_ranges(state))
        issues.extend(self._check_required(state))
        issues.extend(self._check_dates(state))
        issues.extend(self._check_medications(state))
        issues.extend(self._check_confidence(state))
        issues.extend(self._check_blood_pressure(state))

        errors = [i for i in issues if i.severity >= Severity.ERROR]
        confidence = 0.0 if errors else max(0.0, 1.0 - 0.1 * len(issues))

        return AgentResult(
            agent=self.name,
            # The validator succeeded: finding problems is the job. Document
            # validity is reported in `output`, not by failing the agent.
            ok=True,
            issues=issues,
            confidence=round(confidence, 3),
            output={
                "document_valid": not errors,
                "checked_fields": len(state.fields),
                "errors": len(errors),
                "warnings": sum(1 for i in issues if i.severity is Severity.WARNING),
            },
        )

    def _check_ranges(self, state: WorkflowState) -> list[Issue]:
        issues: list[Issue] = []
        for name, reference in REFERENCE_RANGES.items():
            extracted = state.fields.get(name)
            if extracted is None:
                continue
            try:
                value = float(extracted.value)
            except (TypeError, ValueError):
                continue

            if value < reference.impossible_low or value > reference.impossible_high:
                # Not a clinical finding: no patient has a potassium of 68.
                issues.append(
                    Issue(
                        code="CF-VAL-001",
                        severity=Severity.ERROR,
                        message=(
                            f"{name} of {value} {reference.unit} is physiologically "
                            "impossible - likely an extraction error"
                        ),
                        agent=self.name,
                        field_name=name,
                        detail={"value": value, "plausible_range":
                                [reference.impossible_low, reference.impossible_high]},
                    )
                )
                continue

            critical = (
                (reference.critical_low is not None and value < reference.critical_low)
                or (reference.critical_high is not None and value > reference.critical_high)
            )
            if critical:
                issues.append(
                    Issue(
                        code="CF-VAL-002",
                        severity=Severity.CRITICAL,
                        message=(
                            f"{name} of {value} {reference.unit} is critically abnormal "
                            f"(normal {reference.low}-{reference.high})"
                        ),
                        agent=self.name,
                        field_name=name,
                        detail={"value": value, "normal_range": [reference.low, reference.high]},
                    )
                )
            elif value < reference.low or value > reference.high:
                issues.append(
                    Issue(
                        code="CF-VAL-003",
                        severity=Severity.WARNING,
                        message=(
                            f"{name} of {value} {reference.unit} is outside the normal "
                            f"range ({reference.low}-{reference.high})"
                        ),
                        agent=self.name,
                        field_name=name,
                        detail={"value": value, "normal_range": [reference.low, reference.high]},
                    )
                )
        return issues

    def _check_blood_pressure(self, state: WorkflowState) -> list[Issue]:
        extracted = state.fields.get("blood_pressure")
        if extracted is None:
            return []
        match = re.match(r"\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$", str(extracted.value))
        if match is None:
            return [
                Issue(
                    code="CF-VAL-010",
                    severity=Severity.WARNING,
                    message=f"blood pressure {extracted.value!r} is not in systolic/diastolic form",
                    agent=self.name,
                    field_name="blood_pressure",
                )
            ]
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        issues: list[Issue] = []
        if systolic <= diastolic:
            issues.append(
                Issue(
                    code="CF-VAL-011",
                    severity=Severity.ERROR,
                    message=f"systolic ({systolic}) is not above diastolic ({diastolic})",
                    agent=self.name,
                    field_name="blood_pressure",
                )
            )
        elif systolic >= 180 or diastolic >= 120:
            issues.append(
                Issue(
                    code="CF-VAL-012",
                    severity=Severity.CRITICAL,
                    message=f"blood pressure {systolic}/{diastolic} is in the hypertensive crisis range",
                    agent=self.name,
                    field_name="blood_pressure",
                )
            )
        elif systolic >= 140 or diastolic >= 90:
            issues.append(
                Issue(
                    code="CF-VAL-013",
                    severity=Severity.WARNING,
                    message=f"blood pressure {systolic}/{diastolic} is elevated",
                    agent=self.name,
                    field_name="blood_pressure",
                )
            )
        return issues

    def _check_required(self, state: WorkflowState) -> list[Issue]:
        required = REQUIRED_FIELDS.get(state.document.doc_type, [])
        missing = [name for name in required if name not in state.fields]
        if not missing:
            return []
        return [
            Issue(
                code="CF-VAL-020",
                severity=Severity.ERROR,
                message=(
                    f"{state.document.doc_type.value} is missing required field(s): "
                    f"{', '.join(missing)}"
                ),
                agent=self.name,
                detail={"missing": missing},
            )
        ]

    def _check_dates(self, state: WorkflowState) -> list[Issue]:
        issues: list[Issue] = []
        admission = state.fields.get("admission_date")
        discharge = state.fields.get("discharge_date")
        birth = state.fields.get("date_of_birth")

        if admission and discharge:
            start = parse_date(str(admission.value))
            end = parse_date(str(discharge.value))
            if start and end:
                if end < start:
                    issues.append(
                        Issue(
                            code="CF-VAL-030",
                            severity=Severity.ERROR,
                            message=f"discharge date {discharge.value} precedes admission {admission.value}",
                            agent=self.name,
                            field_name="discharge_date",
                        )
                    )
                elif (end - start).days > 365:
                    issues.append(
                        Issue(
                            code="CF-VAL-031",
                            severity=Severity.WARNING,
                            message=f"length of stay of {(end - start).days} days is implausible",
                            agent=self.name,
                            field_name="discharge_date",
                        )
                    )

        for name, extracted in (("admission_date", admission), ("discharge_date", discharge),
                                ("date_of_birth", birth)):
            if extracted is None:
                continue
            parsed = parse_date(str(extracted.value))
            if parsed and parsed > datetime.now():
                issues.append(
                    Issue(
                        code="CF-VAL-032",
                        severity=Severity.ERROR,
                        message=f"{name} {extracted.value} is in the future",
                        agent=self.name,
                        field_name=name,
                    )
                )
        return issues

    def _check_medications(self, state: WorkflowState) -> list[Issue]:
        extracted = state.fields.get("medications")
        if extracted is None or not isinstance(extracted.value, list):
            return []

        issues: list[Issue] = []
        entries = [e for e in extracted.value if isinstance(e, dict)]
        names = {str(e.get("drug", "")).lower() for e in entries}

        for pair, reason, severity in INTERACTIONS:
            if pair <= names:
                issues.append(
                    Issue(
                        code="CF-VAL-040",
                        severity=severity,
                        message=f"interaction: {' + '.join(sorted(pair))} - {reason}",
                        agent=self.name,
                        field_name="medications",
                        detail={"drugs": sorted(pair), "reason": reason},
                    )
                )

        for entry in entries:
            drug = str(entry.get("drug", "")).lower()
            ceiling = MAX_DOSE_MG.get(drug)
            dose = entry.get("dose")
            unit = str(entry.get("unit", "")).lower()
            if ceiling is None or not isinstance(dose, (int, float)):
                continue
            dose_mg = float(dose) / 1000 if unit == "mcg" else float(dose)
            if dose_mg > ceiling:
                issues.append(
                    Issue(
                        code="CF-VAL-041",
                        severity=Severity.CRITICAL,
                        message=(
                            f"{drug} dose of {dose} {unit} exceeds the plausible "
                            f"maximum of {ceiling} mg"
                        ),
                        agent=self.name,
                        field_name="medications",
                        detail={"drug": drug, "dose": dose, "unit": unit, "max_mg": ceiling},
                    )
                )
            if not entry.get("frequency"):
                issues.append(
                    Issue(
                        code="CF-VAL-042",
                        severity=Severity.WARNING,
                        message=f"{drug} has a dose but no frequency",
                        agent=self.name,
                        field_name="medications",
                        detail={"drug": drug},
                    )
                )

        # An allergy and a matching prescription in the same document is the
        # kind of contradiction that should never pass silently.
        allergies = state.fields.get("allergies")
        if allergies is not None and isinstance(allergies.value, list):
            allergy_terms = {str(a).lower() for a in allergies.value}
            for drug in names:
                if any(drug and drug in term or term and term in drug for term in allergy_terms if term):
                    issues.append(
                        Issue(
                            code="CF-VAL-043",
                            severity=Severity.CRITICAL,
                            message=f"{drug} is prescribed but also listed as an allergy",
                            agent=self.name,
                            field_name="medications",
                            detail={"drug": drug},
                        )
                    )
        return issues

    def _check_confidence(self, state: WorkflowState) -> list[Issue]:
        threshold = self.guardrails.min_field_confidence
        low = [f.name for f in state.fields.values() if f.confidence < threshold]
        if not low:
            return []
        return [
            Issue(
                code="CF-VAL-050",
                severity=Severity.WARNING,
                message=f"low-confidence extraction for: {', '.join(sorted(low))}",
                agent=self.name,
                detail={"fields": sorted(low), "threshold": threshold},
            )
        ]
