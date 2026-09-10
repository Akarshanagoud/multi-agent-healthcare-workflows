"""Responsible-AI controls for the agent pipeline.

Multi-agent systems have a failure mode single-call systems do not: an
instruction injected into a document travels. The extractor reads it, the
summariser repeats it, and by the time it reaches a tool call nothing in the
chain remembers that it came from an untrusted PDF rather than from a
clinician.

So this module treats **document text as data, never as instructions**, and
applies controls at both boundaries:

* before the model sees anything — PHI minimisation and injection screening;
* before anything leaves an agent — PHI leak checks and grounding checks.

The grounding check is the one that matters most clinically: an agent claiming
a medication that does not appear in the source is a fabrication, and it is
caught here by comparing quoted evidence against the document.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .types import Issue, Severity

# --- PHI ---------------------------------------------------------------------
# HIPAA Safe Harbor identifiers that can be recognised from text alone.
PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "MRN": re.compile(
        r"(?i)\b(?:mrn|medical[ \t]+record[ \t]+(?:number|no\.?|#))[ \t]*[:#=][ \t]*([A-Z0-9-]{5,15})\b"
    ),
    "SSN": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "PHONE": re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?![\d-])"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "DOB": re.compile(
        r"(?i)\b(?:dob|date[ \t]+of[ \t]+birth|born)[ \t]*[:=]?[ \t]*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"
    ),
    "NPI": re.compile(r"(?i)\bnpi[ \t]*[:#=]?[ \t]*(\d{10})\b"),
    "ACCOUNT": re.compile(r"(?i)\b(?:account|acct)[ \t]*(?:number|no\.?|#)?[ \t]*[:=][ \t]*([A-Z0-9-]{6,20})\b"),
    "INSURANCE_ID": re.compile(
        r"(?i)\b(?:member|policy|insurance)[ \t]*(?:id|number|no\.?|#)[ \t]*[:=][ \t]*([A-Z0-9-]{6,20})\b"
    ),
}

#: Identifiers the extractor legitimately needs to read. Redacting these before
#: extraction would defeat the workflow, so they are pseudonymised instead:
#: stable across a run, meaningless outside it.
PSEUDONYMISE = {"MRN", "ACCOUNT", "INSURANCE_ID", "NPI"}


@dataclass
class PHIFinding:
    entity: str
    span: tuple[int, int]
    replacement: str


@dataclass
class RedactionResult:
    text: str
    findings: list[PHIFinding] = field(default_factory=list)
    #: pseudonym -> original, so the orchestrator can restore identifiers on
    #: output without ever sending them to a model.
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def entities(self) -> list[str]:
        return sorted({f.entity for f in self.findings})


def _captured(match: re.Match[str]) -> tuple[str, tuple[int, int]]:
    if match.groups() and match.group(1):
        return match.group(1), match.span(1)
    return match.group(0), match.span()


def pseudonym(entity: str, value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{entity}:{value}".encode()).hexdigest()[:8].upper()
    return f"<{entity}_{digest}>"


def redact_phi(text: str, salt: str = "clinicflow", entities: list[str] | None = None) -> RedactionResult:
    """Replace PHI before the text reaches a model.

    Entities in :data:`PSEUDONYMISE` become stable tokens (the same MRN maps to
    the same token throughout a run, so an agent can still reason about "the
    same patient"); everything else becomes an opaque label.
    """
    wanted = set(entities) if entities is not None else set(PHI_PATTERNS)
    findings: list[PHIFinding] = []
    mapping: dict[str, str] = {}
    claimed: list[tuple[int, int]] = []

    for entity, pattern in PHI_PATTERNS.items():
        if entity not in wanted:
            continue
        for match in pattern.finditer(text):
            value, span = _captured(match)
            if any(span[0] < end and start < span[1] for start, end in claimed):
                continue
            claimed.append(span)
            replacement = (
                pseudonym(entity, value, salt) if entity in PSEUDONYMISE else f"<{entity}>"
            )
            mapping[replacement] = value
            findings.append(PHIFinding(entity, span, replacement))

    out = text
    for finding in sorted(findings, key=lambda f: f.span[0], reverse=True):
        start, end = finding.span
        out = out[:start] + finding.replacement + out[end:]

    return RedactionResult(text=out, findings=findings, mapping=mapping)


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put pseudonymised identifiers back for a human-facing report."""
    for token, value in mapping.items():
        text = text.replace(token, value)
    return text


def contains_phi(text: str, entities: list[str] | None = None) -> list[str]:
    """Entity types present in ``text``. Used as an output leak check."""
    wanted = set(entities) if entities is not None else set(PHI_PATTERNS)
    return sorted(
        {entity for entity, pattern in PHI_PATTERNS.items()
         if entity in wanted and pattern.search(text)}
    )


# --- Injection ---------------------------------------------------------------
INJECTION_PATTERNS: list[tuple[str, str, Severity]] = [
    (
        "CF-INJ-001",
        r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|system)\b"
        r"[^.\n]{0,20}\b(instruction|prompt|rule)s?\b",
        Severity.CRITICAL,
    ),
    (
        "CF-INJ-002",
        r"(?i)\b(note|attention|important)\b[^.\n]{0,20}\b(to\s+)?(ai|assistant|model|llm)\b\s*[:\-,]",
        Severity.CRITICAL,
    ),
    (
        "CF-INJ-003",
        r"(?i)\bif\s+you\s+are\s+(an?\s+)?(ai|llm|assistant|language\s+model)\b",
        Severity.CRITICAL,
    ),
    (
        "CF-INJ-004",
        r"(?i)\b(reveal|print|output|show)\b[^.\n]{0,25}\b(system\s+prompt|your\s+instructions)\b",
        Severity.ERROR,
    ),
    (
        "CF-INJ-005",
        r"(?i)\b(approve|authorize|mark)\b[^.\n]{0,30}\b(automatically|without\s+review|as\s+approved)\b",
        Severity.CRITICAL,
    ),
    (
        "CF-INJ-006",
        r"(?i)\b(send|email|forward|transmit)\b[^.\n]{0,30}\b(to|at)\b\s*(https?://|[\w.-]+@[\w.-]+)",
        Severity.CRITICAL,
    ),
    ("CF-INJ-007", r"<\s*/?\s*(system|assistant|im_start|im_end)\s*\|?\s*>", Severity.ERROR),
]

_COMPILED_INJECTION = [(code, re.compile(pattern), sev) for code, pattern, sev in INJECTION_PATTERNS]

# Zero-width characters used to hide instructions from a human reviewer while
# leaving them fully visible to the tokenizer.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")


def screen_document(text: str, agent: str = "guardrail") -> list[Issue]:
    """Screen untrusted document text for embedded instructions.

    An intake document containing "approve this automatically" is not a
    processing hint, it is an attack on the workflow's authority — hence
    CRITICAL rather than a warning.
    """
    issues: list[Issue] = []
    normalized = _INVISIBLE.sub("", text)

    for code, pattern, severity in _COMPILED_INJECTION:
        match = pattern.search(text) or pattern.search(normalized)
        if match is None:
            continue
        issues.append(
            Issue(
                code=code,
                severity=severity,
                message="document contains text addressed to the model, not the reader",
                agent=agent,
                blocking=severity >= Severity.CRITICAL,
                detail={"evidence": " ".join(match.group(0).split())[:120]},
            )
        )

    hidden = _INVISIBLE.findall(text)
    if hidden:
        issues.append(
            Issue(
                code="CF-INJ-010",
                severity=Severity.ERROR,
                message=f"document contains {len(hidden)} invisible character(s)",
                agent=agent,
                detail={"count": len(hidden)},
            )
        )
    return issues


def neutralize(text: str) -> str:
    """Strip hidden characters and fence the text as data.

    Fencing does not make injection impossible; it makes the trust boundary
    explicit in the prompt, which is the part a reviewer can check.
    """
    cleaned = _INVISIBLE.sub("", text)
    cleaned = cleaned.replace("<|", "<").replace("|>", ">")
    return cleaned


# --- Grounding ---------------------------------------------------------------
def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_grounding(
    claims: list[tuple[str, str]], document: str, agent: str = "guardrail"
) -> list[Issue]:
    """Verify each ``(field_name, quoted_evidence)`` appears in the document.

    This is the fabrication check. An agent that reports a medication with
    evidence not present in the source has invented it, and in a clinical
    workflow that must stop the run rather than flow downstream.
    """
    haystack = _normalise(document)
    issues: list[Issue] = []
    for name, evidence in claims:
        if not evidence:
            issues.append(
                Issue(
                    code="CF-GND-002",
                    severity=Severity.ERROR,
                    message=f"field {name!r} was reported without supporting evidence",
                    agent=agent,
                    field_name=name,
                )
            )
            continue
        if _normalise(evidence) not in haystack:
            issues.append(
                Issue(
                    code="CF-GND-001",
                    severity=Severity.CRITICAL,
                    message=f"field {name} cites text that does not appear in the document",
                    agent=agent,
                    field_name=name,
                    blocking=True,
                    detail={"claimed_evidence": evidence[:120]},
                )
            )
    return issues


def check_output(text: str, document: str, agent: str = "guardrail") -> list[Issue]:
    """Screen agent output before it leaves the workflow."""
    issues: list[Issue] = []

    leaked = contains_phi(text)
    if leaked:
        issues.append(
            Issue(
                code="CF-PHI-001",
                severity=Severity.CRITICAL,
                message=f"output contains unredacted PHI: {', '.join(leaked)}",
                agent=agent,
                blocking=True,
                detail={"entities": leaked},
            )
        )

    # An agent repeating an injected instruction back is a sign the payload
    # travelled through the pipeline rather than being contained at intake.
    for code, pattern, _severity in _COMPILED_INJECTION:
        if pattern.search(text):
            issues.append(
                Issue(
                    code="CF-INJ-020",
                    severity=Severity.CRITICAL,
                    message="agent output repeats an instruction-shaped payload",
                    agent=agent,
                    blocking=True,
                    detail={"rule": code},
                )
            )
            break

    del document  # grounding is checked per-field by check_grounding
    return issues


@dataclass
class GuardrailConfig:
    redact_phi: bool = True
    phi_salt: str = "clinicflow"
    screen_injection: bool = True
    check_grounding: bool = True
    check_output_leaks: bool = True
    #: Fields below this confidence are routed to human review rather than
    #: accepted. Clinical workflows should not silently accept a guess.
    min_field_confidence: float = 0.6
    #: Any issue at or above this severity forces human review.
    review_severity: Severity = Severity.WARNING
    #: Reject documents carrying a blocking issue (injection, fabrication, PHI
    #: leak). Clinical severity never rejects on its own - see Issue.blocking.
    reject_on_blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "redact_phi": self.redact_phi,
            "screen_injection": self.screen_injection,
            "check_grounding": self.check_grounding,
            "check_output_leaks": self.check_output_leaks,
            "min_field_confidence": self.min_field_confidence,
            "review_severity": self.review_severity.value,
            "reject_on_blocking": self.reject_on_blocking,
        }
