"""Review desk: the reviewer-facing web UI over the workflow.

Mounted by ``clinicflow.server`` at ``/ui``. It runs the same workflow as
``/v1/process`` and adds what a reviewer needs on screen: the text the agents
actually read, where every field came from, a queue ordered by severity, and
exactly what PHI minimisation removed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents.router import ROUTES
from .guardrails import PSEUDONYMISE, redact_phi, screen_document
from .orchestrator import Workflow, summarize_batch
from .types import Decision, Document, DocumentType, WorkflowState

WEB_DIR = Path(__file__).parent / "web"
DEFAULT_SAMPLES = Path(__file__).resolve().parents[2] / "data" / "sample_documents.json"
PIPELINE = ["intake", "router", "retriever", "extractor", "validator", "summarizer"]
FLAGGED = (Decision.REJECTED, Decision.FAILED, Decision.NEEDS_REVIEW)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    doc_id: str | None = Field(None, max_length=80)


class ScreenRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)


def load_samples(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline(state: WorkflowState) -> list[dict[str, Any]]:
    """Every agent in order, with what it did or why it did not run."""
    ran = {entry["agent"]: entry for entry in state.trace}
    try:
        route = ROUTES[DocumentType(state.document.doc_type)] if state.document.doc_type else []
    except ValueError:
        route = []
    steps = []
    for agent in PIPELINE:
        entry = ran.get(agent)
        if entry is not None:
            steps.append({**entry, "ran": True, "skipped_reason": None})
        elif state.decision is Decision.REJECTED:
            steps.append({"agent": agent, "ran": False, "skipped_reason": "document quarantined at intake"})
        elif agent in ("intake", "router") or agent in route:
            steps.append({"agent": agent, "ran": False, "skipped_reason": "pipeline stopped before this agent"})
        else:
            steps.append({"agent": agent, "ran": False, "skipped_reason": "not on the route for this document type"})
    return steps


def review(workflow: Workflow, text: str, doc_id: str | None = None) -> dict[str, Any]:
    document = Document(text=text, source="review-desk", **({"doc_id": doc_id} if doc_id else {}))
    state = workflow.run(document)
    payload = state.to_dict()
    summary = state.result_for("summarizer")
    retrieval = state.result_for("retriever")
    payload.update(
        original_text=text,
        # Spans and grounding refer to this text: PHI-minimised, what the agents read.
        agent_text=state.document.text,
        phi_entities=state.document.metadata.get("phi_entities", []),
        summary=(summary.output.get("summary_for_review") or summary.output.get("summary")) if summary else None,
        summary_source=summary.output.get("source") if summary else None,
        policy_context=retrieval.output.get("passages", []) if retrieval else [],
        pipeline=_pipeline(state),
    )
    return payload


def queue(workflow: Workflow, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise HTTPException(status_code=409, detail="no sample documents are configured")
    states = [workflow.run(Document(text=s["text"], doc_id=s["doc_id"], source="samples")) for s in samples]
    rows = []
    for sample, state in zip(samples, states, strict=True):
        rows.append(
            {
                "doc_id": sample["doc_id"],
                "label": sample.get("label", ""),
                "doc_type": state.document.doc_type,
                "decision": state.decision.value,
                "expected_decision": sample.get("expected_decision"),
                "matches_expected": sample.get("expected_decision") in (None, state.decision.value),
                "max_severity": state.max_severity.value,
                "severity_rank": state.max_severity.rank,
                "confidence": round(state.mean_confidence, 3),
                "fields": len(state.fields),
                "issues": len(state.issues),
                "reasons": state.review_reasons,
                "duration_ms": round(((state.finished_at or state.started_at) - state.started_at) * 1000, 2),
            }
        )
    # Worst first: anything a human must see, by severity then by how unsure the system was.
    rows.sort(key=lambda r: (r["decision"] == Decision.COMPLETE.value, -r["severity_rank"], r["confidence"]))
    return {"summary": summarize_batch(states), "rows": rows}


def screen(text: str) -> dict[str, Any]:
    redaction = redact_phi(text)
    findings = []
    shift = 0
    for finding in sorted(redaction.findings, key=lambda f: f.span[0]):
        start, end = finding.span
        r_start = start + shift
        r_end = r_start + len(finding.replacement)
        shift += len(finding.replacement) - (end - start)
        findings.append(
            {
                "entity": finding.entity,
                "start": start,
                "end": end,
                "replacement": finding.replacement,
                "redacted_start": r_start,
                "redacted_end": r_end,
                "pseudonymised": finding.entity in PSEUDONYMISE,
            }
        )
    issues = screen_document(text)
    return {
        "original_text": text,
        "redacted_text": redaction.text,
        "findings": findings,
        "entities": redaction.entities,
        "injection_issues": [i.to_dict() for i in issues],
        "would_be_rejected": any(i.blocking for i in issues),
    }


def build_router(workflow: Workflow, samples: list[dict[str, Any]]) -> APIRouter:
    router = APIRouter(prefix="/v1/ui", tags=["review desk"])

    @router.get("/samples")
    def get_samples() -> list[dict[str, Any]]:
        return samples

    @router.post("/queue")
    def run_queue() -> dict[str, Any]:
        return queue(workflow, samples)

    @router.post("/review")
    def run_review(body: ReviewRequest) -> dict[str, Any]:
        return review(workflow, body.text, body.doc_id)

    @router.post("/screen")
    def run_screen(body: ScreenRequest) -> dict[str, Any]:
        return screen(body.text)

    @router.get("/kb")
    def kb(q: str = "", top_k: int = 6) -> dict[str, Any]:
        knowledge_base = workflow.agents["retriever"].knowledge_base
        titles: dict[str, int] = {}
        for passage in knowledge_base.passages:
            titles[passage.title] = titles.get(passage.title, 0) + 1
        hits = knowledge_base.search(q, top_k=max(1, min(top_k, 20))) if q.strip() else []
        return {
            "passages": len(knowledge_base),
            "documents": [{"title": t, "sections": n} for t, n in sorted(titles.items())],
            "hits": [h.to_dict() for h in hits],
        }

    @router.get("/audit")
    def audit(limit: int = 200) -> dict[str, Any]:
        records = workflow.audit[-max(1, min(limit, 2000)):]
        return {
            "total": len(workflow.audit),
            "records": [
                {
                    "run_id": r.get("run_id"),
                    "doc_id": r.get("doc_id"),
                    "doc_type": r.get("doc_type"),
                    "decision": r.get("decision"),
                    "max_severity": r.get("max_severity"),
                    "mean_confidence": r.get("mean_confidence"),
                    "duration_ms": r.get("duration_ms"),
                    "checksum": r.get("checksum"),
                    "review_reasons": r.get("review_reasons", []),
                    "fields": sorted((r.get("fields") or {}).keys()),
                    "agents": [t.get("agent") for t in r.get("trace", [])],
                }
                for r in reversed(records)
            ],
        }

    return router


def mount_console(app: FastAPI, workflow: Workflow) -> list[dict[str, Any]]:
    """Attach the review desk API, its static front end and a ``/`` redirect."""
    env = os.getenv("CLINICFLOW_SAMPLES")
    samples = load_samples(Path(env) if env else DEFAULT_SAMPLES)
    app.include_router(build_router(workflow, samples))
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="review-desk")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return samples
