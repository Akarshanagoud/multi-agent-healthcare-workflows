"""Agent base class.

Every agent runs inside the same envelope: timing, error containment and a
uniform result shape. Containment matters more here than in a single-call
system - one agent raising must degrade the run to "needs review", not take
down the pipeline and lose the partial work a clinician could still use.
"""
from __future__ import annotations

import abc
import time

from ..guardrails import GuardrailConfig
from ..llm import NullProvider, Provider
from ..types import AgentResult, Issue, Severity, WorkflowState


class Agent(abc.ABC):
    """One specialised step in the workflow."""

    name: str = "agent"
    #: Whether a failure of this agent should stop the pipeline. Extraction
    #: failing is fatal; summarisation failing is not.
    critical: bool = False

    def __init__(
        self,
        provider: Provider | None = None,
        guardrails: GuardrailConfig | None = None,
    ) -> None:
        self.provider = provider or NullProvider()
        self.guardrails = guardrails or GuardrailConfig()

    @abc.abstractmethod
    def run(self, state: WorkflowState) -> AgentResult:
        """Do the work. Raise freely; execute() contains it."""

    def execute(self, state: WorkflowState) -> AgentResult:
        """Run with timing and error containment."""
        started = time.perf_counter()
        try:
            result = self.run(state)
        except Exception as exc:  # noqa: BLE001 - contained by design
            result = AgentResult(
                agent=self.name,
                ok=False,
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
                issues=[
                    Issue(
                        code="CF-AGENT-500",
                        severity=Severity.CRITICAL if self.critical else Severity.ERROR,
                        message=f"agent {self.name} failed: {exc}",
                        agent=self.name,
                    )
                ],
            )
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    def _model_available(self) -> bool:
        """Whether a real model is configured, as opposed to the rules-only default."""
        return getattr(self.provider, "available", True) and not isinstance(
            self.provider, NullProvider
        )
