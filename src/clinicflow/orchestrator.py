"""Workflow orchestration.

The orchestrator is where governance actually lives. Individual agents produce
findings; this decides what those findings mean for the document, and it is
built around one rule:

    **Nothing reaches a downstream system without either passing every control
    or being marked for human review.**

There is no path that silently accepts partial output. A document either
completes cleanly, is escalated to a reviewer with the reasons attached, or is
rejected. That is the difference between a demo that chains agents together
and a workflow you could put in front of a clinical governance committee.

The pipeline is:

    intake (PHI + injection screening)
      -> router (classify, choose the path)
      -> [retriever] -> extractor -> validator -> [summarizer]
      -> decision

Agent order is data (``ROUTES``), not control flow, so adding a document type
does not mean editing this file.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agents.base import Agent
from .agents.extractor import ExtractorAgent
from .agents.retriever import KnowledgeBase, RetrieverAgent
from .agents.router import ROUTES, RouterAgent
from .agents.summarizer import SummarizerAgent
from .agents.validator import ValidatorAgent
from .guardrails import GuardrailConfig, redact_phi, restore, screen_document
from .llm import NullProvider, Provider
from .types import (
    AgentResult,
    Decision,
    Document,
    DocumentType,
    Issue,
    WorkflowState,
)


@dataclass
class WorkflowConfig:
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    #: Retry a failed non-critical agent this many times before giving up.
    max_agent_retries: int = 1
    #: Escalate to review when the mean field confidence falls below this.
    min_mean_confidence: float = 0.65
    #: Escalate when the router is this unsure about the document type.
    min_router_confidence: float = 0.35
    #: Stop the run as soon as a critical agent fails.
    fail_fast: bool = True
    audit_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "guardrails": self.guardrails.to_dict(),
            "max_agent_retries": self.max_agent_retries,
            "min_mean_confidence": self.min_mean_confidence,
            "min_router_confidence": self.min_router_confidence,
            "fail_fast": self.fail_fast,
        }


class Workflow:
    """Runs the agent pipeline over a document and decides the outcome."""

    def __init__(
        self,
        config: WorkflowConfig | None = None,
        provider: Provider | None = None,
        knowledge_base: KnowledgeBase | None = None,
        agents: dict[str, Agent] | None = None,
    ) -> None:
        self.config = config or WorkflowConfig()
        self.provider = provider or NullProvider()
        guardrails = self.config.guardrails

        self.agents: dict[str, Agent] = agents or {
            "router": RouterAgent(self.provider, guardrails),
            "retriever": RetrieverAgent(
                knowledge_base=knowledge_base, provider=self.provider, guardrails=guardrails
            ),
            "extractor": ExtractorAgent(self.provider, guardrails),
            "validator": ValidatorAgent(self.provider, guardrails),
            "summarizer": SummarizerAgent(self.provider, guardrails),
        }
        self._hooks: list[Callable[[str, AgentResult, WorkflowState], None]] = []
        self._audit: list[dict[str, Any]] = []

    def add_hook(self, hook: Callable[[str, AgentResult, WorkflowState], None]) -> None:
        """Called after each agent. Used for tracing and metrics."""
        self._hooks.append(hook)

    # -- pipeline -----------------------------------------------------------
    def run(self, document: Document | str) -> WorkflowState:
        if isinstance(document, str):
            document = Document(text=document)

        state = WorkflowState(document=document)
        redaction_mapping: dict[str, str] = {}

        # --- intake --------------------------------------------------------
        if self.config.guardrails.screen_injection:
            intake_issues = screen_document(document.text, agent="intake")
            state.issues.extend(intake_issues)
            state.trace.append(
                {"agent": "intake", "ok": True, "issues": len(intake_issues),
                 "fields": 0, "confidence": 1.0, "elapsed_ms": 0.0, "error": None}
            )
            if any(i.blocking for i in intake_issues):
                # A document carrying instructions for the model is not
                # processed on a best-effort basis. It is quarantined.
                state.require_review("document contains embedded instructions")
                state.decision = Decision.REJECTED
                state.finished_at = time.time()
                self._record(state)
                return state

        if self.config.guardrails.redact_phi:
            redaction = redact_phi(document.text, salt=self.config.guardrails.phi_salt)
            redaction_mapping = redaction.mapping
            # Agents see the minimised text; the original is never mutated, so
            # spans and grounding are checked against what the reviewer sees.
            state.document = Document(
                text=redaction.text,
                doc_id=document.doc_id,
                source=document.source,
                doc_type=document.doc_type,
                metadata={**document.metadata, "phi_entities": redaction.entities},
                received_at=document.received_at,
            )

        # --- routing -------------------------------------------------------
        router_result = self._run_agent("router", state)
        if not router_result.ok:
            state.decision = Decision.FAILED
            state.require_review("router failed")
            state.finished_at = time.time()
            self._record(state)
            return state

        route = router_result.output.get("route") or ROUTES[DocumentType.UNKNOWN]
        if router_result.confidence < self.config.min_router_confidence:
            state.require_review(
                f"low routing confidence ({router_result.confidence:.2f})"
            )

        # --- specialised agents --------------------------------------------
        for agent_name in route:
            if agent_name not in self.agents:
                continue
            result = self._run_agent(agent_name, state)
            # ok is about the agent, not the document: a validator that finds
            # ten problems has done its job perfectly. Only an agent that
            # actually raised stops the pipeline.
            if result.error and self.agents[agent_name].critical and self.config.fail_fast:
                state.require_review(f"{agent_name} failed: {result.error}")
                state.decision = Decision.FAILED
                break

        # --- decision ------------------------------------------------------
        if state.decision not in (Decision.FAILED, Decision.REJECTED):
            self._decide(state)

        if redaction_mapping:
            state.document.metadata["phi_restorable"] = True
            summary = state.result_for("summarizer")
            if summary is not None and "summary" in summary.output:
                # The reviewer needs the real identifiers; the model never saw them.
                summary.output["summary_for_review"] = restore(
                    summary.output["summary"], redaction_mapping
                )

        state.finished_at = time.time()
        self._record(state)
        return state

    def _run_agent(self, name: str, state: WorkflowState) -> AgentResult:
        agent = self.agents[name]
        attempts = 0
        result = agent.execute(state)

        while (
            not result.ok
            and not agent.critical
            and attempts < self.config.max_agent_retries
        ):
            attempts += 1
            result = agent.execute(state)

        if attempts:
            result.output["retries"] = attempts

        state.add_result(result)
        for hook in self._hooks:
            try:
                hook(name, result, state)
            except Exception:  # noqa: BLE001 - observability must not break the run
                continue
        return result

    def _decide(self, state: WorkflowState) -> None:
        """Turn accumulated issues and confidences into an outcome."""
        guardrails = self.config.guardrails

        blocking = [i for i in state.issues if i.blocking]
        if blocking and guardrails.reject_on_blocking:
            # Integrity failures only: injection, fabricated evidence, PHI
            # leaks. A critically abnormal lab value is urgent, not blocking -
            # rejecting it would discard the finding that needs acting on.
            for issue in blocking:
                state.require_review(issue.message)
            state.decision = Decision.REJECTED
            return

        if any(i.severity >= guardrails.review_severity for i in state.issues):
            for issue in state.issues:
                if issue.severity >= guardrails.review_severity:
                    state.require_review(issue.message)

        if state.fields and state.mean_confidence < self.config.min_mean_confidence:
            state.require_review(
                f"mean extraction confidence {state.mean_confidence:.2f} below "
                f"{self.config.min_mean_confidence}"
            )

        if not state.fields:
            state.require_review("no fields were extracted")

    # -- audit --------------------------------------------------------------
    def _record(self, state: WorkflowState) -> None:
        record = state.to_dict()
        record["config"] = self.config.to_dict()
        self._audit.append(record)
        if self.config.audit_log_path:
            import json
            from pathlib import Path

            path = Path(self.config.audit_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self._audit

    def run_batch(self, documents: list[Document | str]) -> list[WorkflowState]:
        return [self.run(document) for document in documents]


def summarize_batch(states: list[WorkflowState]) -> dict[str, Any]:
    """Aggregate outcomes across a batch, for the CLI and the service."""
    if not states:
        return {"documents": 0}

    decisions: dict[str, int] = {}
    issue_codes: dict[str, int] = {}
    for state in states:
        decisions[state.decision.value] = decisions.get(state.decision.value, 0) + 1
        for issue in state.issues:
            issue_codes[issue.code] = issue_codes.get(issue.code, 0) + 1

    reviewed = decisions.get(Decision.NEEDS_REVIEW.value, 0)
    return {
        "documents": len(states),
        "decisions": decisions,
        "straight_through_rate": round(
            decisions.get(Decision.COMPLETE.value, 0) / len(states), 4
        ),
        "review_rate": round(reviewed / len(states), 4),
        "mean_confidence": round(
            sum(s.mean_confidence for s in states) / len(states), 3
        ),
        "mean_duration_ms": round(
            sum(s.duration_ms for s in states) / len(states), 2
        ),
        "top_issues": sorted(issue_codes.items(), key=lambda kv: -kv[1])[:10],
    }


def escalations(states: list[WorkflowState]) -> list[dict[str, Any]]:
    """Documents a human must look at, worst first."""
    flagged = [
        s for s in states
        if s.decision in (Decision.NEEDS_REVIEW, Decision.REJECTED, Decision.FAILED)
    ]
    flagged.sort(key=lambda s: (-s.max_severity.rank, s.mean_confidence))
    return [
        {
            "doc_id": s.document.doc_id,
            "decision": s.decision.value,
            "max_severity": s.max_severity.value,
            "confidence": round(s.mean_confidence, 3),
            "reasons": s.review_reasons,
        }
        for s in flagged
    ]


__all__ = [
    "Issue",
    "Workflow",
    "WorkflowConfig",
    "escalations",
    "summarize_batch",
]
