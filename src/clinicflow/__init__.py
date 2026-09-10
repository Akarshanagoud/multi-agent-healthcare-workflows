"""clinicflow - multi-agent clinical document workflows with responsible-AI controls."""
from .agents import (
    Agent,
    ExtractorAgent,
    KnowledgeBase,
    Passage,
    RetrieverAgent,
    RouterAgent,
    SummarizerAgent,
    ValidatorAgent,
)
from .guardrails import GuardrailConfig, contains_phi, redact_phi, screen_document
from .llm import EchoProvider, NullProvider, OllamaProvider, Provider, build_provider
from .orchestrator import Workflow, WorkflowConfig, escalations, summarize_batch
from .types import (
    AgentResult,
    Decision,
    Document,
    DocumentType,
    Field,
    Issue,
    Severity,
    WorkflowState,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "Decision",
    "Document",
    "DocumentType",
    "EchoProvider",
    "ExtractorAgent",
    "Field",
    "GuardrailConfig",
    "Issue",
    "KnowledgeBase",
    "NullProvider",
    "OllamaProvider",
    "Passage",
    "Provider",
    "RetrieverAgent",
    "RouterAgent",
    "Severity",
    "SummarizerAgent",
    "ValidatorAgent",
    "Workflow",
    "WorkflowConfig",
    "WorkflowState",
    "build_provider",
    "contains_phi",
    "escalations",
    "redact_phi",
    "screen_document",
    "summarize_batch",
    "__version__",
]
