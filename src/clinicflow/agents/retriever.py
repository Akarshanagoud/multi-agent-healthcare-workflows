"""Retriever agent: pull the policy and guideline context a decision needs.

Retrieval here is BM25 over a small local knowledge base - no embedding model,
no vector database, no network. For a corpus of policy documents in the
hundreds, lexical retrieval is not a compromise: clinical policy text is
dense with exact terms (drug names, CPT codes, "medical necessity"), and exact
terms are precisely what lexical scoring is good at. An embedding model would
add a dependency, a download and a source of nondeterminism to solve a problem
this corpus does not have.

Every retrieved passage keeps its source id and section, because a downstream
summary that cites "policy" without saying which policy is not auditable.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import AgentResult, Issue, Severity, WorkflowState
from .base import Agent

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

# Words that carry no retrieval signal in clinical policy text.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "was",
    "were", "be", "been", "with", "as", "at", "by", "on", "that", "this", "it",
    "from", "has", "have", "had", "not", "but", "if", "then", "than", "when",
    "which", "who", "will", "would", "should", "may", "can", "must", "shall",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Passage:
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}"


@dataclass
class Retrieved:
    passage: Passage
    score: float
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation": self.passage.citation,
            "title": self.passage.title,
            "section": self.passage.section,
            "score": round(self.score, 3),
            "matched_terms": self.matched_terms[:8],
            "text": self.passage.text,
        }


class KnowledgeBase:
    """BM25 index over policy and guideline passages."""

    #: Standard BM25 parameters. k1 controls term-frequency saturation, b
    #: controls length normalisation.
    K1 = 1.5
    B = 0.75

    def __init__(self, passages: list[Passage] | None = None) -> None:
        self.passages: list[Passage] = []
        self._tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        for passage in passages or []:
            self.add(passage)

    def add(self, passage: Passage) -> None:
        tokens = tokenize(f"{passage.title} {passage.section} {passage.text}")
        self.passages.append(passage)
        self._tokens.append(tokens)
        counts = Counter(tokens)
        self._term_frequencies.append(counts)
        for term in counts:
            self._document_frequency[term] += 1
        lengths = [len(t) for t in self._tokens]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def __len__(self) -> int:
        return len(self.passages)

    def _idf(self, term: str) -> float:
        n = len(self.passages)
        df = self._document_frequency.get(term, 0)
        # BM25 IDF with the +0.5 smoothing that keeps common terms positive.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 4, min_score: float = 0.5) -> list[Retrieved]:
        query_terms = tokenize(query)
        if not query_terms or not self.passages:
            return []

        results: list[Retrieved] = []
        for index, passage in enumerate(self.passages):
            counts = self._term_frequencies[index]
            length = len(self._tokens[index]) or 1
            score = 0.0
            matched: list[str] = []
            for term in set(query_terms):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                matched.append(term)
                numerator = frequency * (self.K1 + 1)
                denominator = frequency + self.K1 * (
                    1 - self.B + self.B * length / (self._average_length or 1)
                )
                score += self._idf(term) * numerator / denominator
            if score >= min_score:
                results.append(
                    Retrieved(
                        passage=passage,
                        score=score,
                        matched_terms=sorted(matched, key=lambda t: -self._idf(t)),
                    )
                )

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    @classmethod
    def from_file(cls, path: str | Path) -> KnowledgeBase:
        """Load a JSON list of passage objects."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [
                Passage(
                    doc_id=item["doc_id"],
                    title=item.get("title", item["doc_id"]),
                    section=item.get("section", "general"),
                    text=item["text"],
                    metadata=item.get("metadata", {}),
                )
                for item in data
            ]
        )

    @classmethod
    def default(cls) -> KnowledgeBase:
        """The bundled synthetic policy corpus."""
        path = Path(__file__).resolve().parents[3] / "data" / "knowledge_base.json"
        if path.exists():
            return cls.from_file(path)
        return cls([])


class RetrieverAgent(Agent):
    """Finds the policy context relevant to this document."""

    name = "retriever"
    critical = False

    def __init__(self, knowledge_base: KnowledgeBase | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base if knowledge_base is not None else KnowledgeBase.default()

    def build_query(self, state: WorkflowState) -> str:
        """Query from the document type plus whatever the extractor found.

        Running after extraction is deliberate: querying with the raw document
        retrieves on incidental vocabulary, while querying with the diagnosis
        and requested service retrieves on what the decision is about.
        """
        parts = [state.document.doc_type.value.replace("_", " ")]
        for name in (
            "primary_diagnosis", "reason_for_referral", "requested_service",
            "chief_complaint", "icd10_code", "cpt_code",
        ):
            extracted = state.fields.get(name)
            if extracted is not None:
                parts.append(str(extracted.value))

        medications = state.fields.get("medications")
        if medications is not None and isinstance(medications.value, list):
            parts.extend(
                str(entry.get("drug", "")) for entry in medications.value[:5]
                if isinstance(entry, dict)
            )

        # Fall back to the document head only if extraction produced nothing.
        if len(parts) == 1:
            parts.append(state.document.text[:400])
        return " ".join(p for p in parts if p)

    def run(self, state: WorkflowState) -> AgentResult:
        if len(self.knowledge_base) == 0:
            return AgentResult(
                agent=self.name,
                confidence=0.0,
                issues=[
                    Issue(
                        code="CF-RET-001",
                        severity=Severity.INFO,
                        message="knowledge base is empty; no policy context retrieved",
                        agent=self.name,
                    )
                ],
                output={"query": "", "passages": []},
            )

        query = self.build_query(state)
        hits = self.knowledge_base.search(query, top_k=4)
        issues: list[Issue] = []

        if not hits:
            issues.append(
                Issue(
                    code="CF-RET-002",
                    severity=Severity.WARNING,
                    message="no policy context matched this document",
                    agent=self.name,
                    detail={"query": query[:200]},
                )
            )

        # Confidence is the top score normalised, damped when the top two hits
        # are close: several equally-relevant policies means the retriever has
        # not actually discriminated.
        confidence = 0.0
        if hits:
            top = hits[0].score
            second = hits[1].score if len(hits) > 1 else 0.0
            separation = min((top - second) / top, 1.0) if top else 0.0
            confidence = min(top / 8.0, 1.0) * (0.6 + 0.4 * separation)

        return AgentResult(
            agent=self.name,
            confidence=round(confidence, 3),
            issues=issues,
            output={
                "query": query[:300],
                "passages": [hit.to_dict() for hit in hits],
                "citations": [hit.passage.citation for hit in hits],
            },
        )
