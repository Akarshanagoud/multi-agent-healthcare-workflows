"""FastAPI service.

    pip install "clinicflow[server]"
    uvicorn clinicflow.server:app --port 8090

Environment:
    CLINICFLOW_PROVIDER   null | echo | ollama | openai-compat  (default: null)
    CLINICFLOW_MODEL      model name for the chosen provider
    CLINICFLOW_KB         path to a JSON policy corpus
    CLINICFLOW_AUDIT_LOG  JSONL path for run records
    CLINICFLOW_SAMPLES    sample documents for the review desk queue

The review desk web UI is served from the same app at ``/ui``.

Note that ``/v1/process`` never returns a bare answer. It returns a decision,
the fields with their provenance, the issues, and the agent trace - because a
clinical workflow that returns only its conclusion cannot be reviewed, and a
conclusion nobody can review cannot be used.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "clinicflow.server requires the server extra: pip install 'clinicflow[server]'"
    ) from exc

from . import __version__
from .agents.retriever import KnowledgeBase
from .console import mount_console
from .guardrails import GuardrailConfig, redact_phi, screen_document
from .llm import build_provider
from .orchestrator import Workflow, WorkflowConfig, escalations, summarize_batch
from .types import Document


def build_workflow() -> Workflow:
    provider_kind = os.getenv("CLINICFLOW_PROVIDER", "null")
    provider_kwargs: dict[str, Any] = {}
    if provider_kind in {"ollama", "openai-compat", "vllm"}:
        provider_kwargs["model"] = os.getenv("CLINICFLOW_MODEL", "llama3.2")

    kb_path = os.getenv("CLINICFLOW_KB")
    knowledge_base = KnowledgeBase.from_file(kb_path) if kb_path else None

    config = WorkflowConfig(
        guardrails=GuardrailConfig(),
        audit_log_path=os.getenv("CLINICFLOW_AUDIT_LOG"),
    )
    return Workflow(
        config=config,
        provider=build_provider(provider_kind, **provider_kwargs),
        knowledge_base=knowledge_base,
    )


workflow = build_workflow()

app = FastAPI(
    title="Clinical Document Workflow",
    version=__version__,
    description=(
        "Multi-agent document processing with PHI minimisation, injection "
        "screening, grounded extraction and human-review routing."
    ),
)


class ProcessRequest(BaseModel):
    text: str = Field(..., description="The clinical document text")
    doc_id: str | None = None
    source: str = "api"


class BatchRequest(BaseModel):
    documents: list[ProcessRequest]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "provider": workflow.provider.name,
        "knowledge_base_size": len(workflow.agents["retriever"].knowledge_base),
    }


@app.get("/policy")
def policy() -> dict[str, Any]:
    """The active governance configuration, so callers know the rules applied."""
    return workflow.config.to_dict()


@app.post("/v1/process")
def process(request: ProcessRequest) -> dict[str, Any]:
    document = Document(
        text=request.text, source=request.source,
        **({"doc_id": request.doc_id} if request.doc_id else {}),
    )
    state = workflow.run(document)
    payload = state.to_dict()

    summary = state.result_for("summarizer")
    if summary is not None:
        payload["summary"] = summary.output.get("summary")
        payload["citations"] = summary.output.get("citations", [])

    retrieval = state.result_for("retriever")
    if retrieval is not None:
        payload["policy_context"] = retrieval.output.get("passages", [])

    return payload


@app.post("/v1/batch")
def batch(request: BatchRequest) -> dict[str, Any]:
    states = [
        workflow.run(Document(text=item.text, source=item.source,
                              **({"doc_id": item.doc_id} if item.doc_id else {})))
        for item in request.documents
    ]
    return {
        "summary": summarize_batch(states),
        "review_queue": escalations(states),
        "results": [s.to_dict() for s in states],
    }


@app.post("/v1/screen")
def screen(request: ProcessRequest) -> dict[str, Any]:
    """Show what redaction and injection screening do, without processing.

    Useful as a pre-flight check, and as the endpoint a privacy reviewer asks
    for when they want to see exactly what leaves the boundary.
    """
    redaction = redact_phi(request.text)
    issues = screen_document(request.text)
    return {
        "phi_entities": redaction.entities,
        "redacted_text": redaction.text,
        "injection_issues": [i.to_dict() for i in issues],
        "would_be_rejected": any(i.blocking for i in issues),
    }


@app.get("/v1/kb")
def search_kb(q: str, top_k: int = 4) -> list[dict[str, Any]]:
    knowledge_base = workflow.agents["retriever"].knowledge_base
    if len(knowledge_base) == 0:
        raise HTTPException(status_code=409, detail="no knowledge base configured")
    return [hit.to_dict() for hit in knowledge_base.search(q, top_k=top_k)]


@app.get("/v1/audit")
def audit(limit: int = 50) -> dict[str, Any]:
    records = workflow.audit[-limit:]
    return {"records": len(workflow.audit), "returned": len(records), "audit": records}


# The review desk (``/ui``) runs this workflow, so its queue and audit are the service's own.
samples = mount_console(app, workflow)
