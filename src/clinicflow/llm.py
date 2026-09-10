"""Model provider adapters.

The workflow is built so that **the rule-based path is the default and the
model is an augmentation**, not the other way round. That is a deliberate
clinical-governance choice: a regex that finds "BP 148/92" is deterministic,
auditable and testable, and no review board has to reason about its failure
modes. The model earns its place on the parts rules genuinely cannot do —
narrative summarisation, ambiguous phrasing, unusual document layouts.

Providers here are all free and self-hostable:

* :class:`NullProvider`   - no model at all; rules only (the default)
* :class:`EchoProvider`   - deterministic stub for tests and CI
* :class:`OllamaProvider` - local models via Ollama (llama3, mistral, phi)
* :class:`OpenAICompatProvider` - any OpenAI-shaped endpoint you run yourself

No provider is required for the workflow to run end to end.
"""
from __future__ import annotations

import abc
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a model call fails. Callers degrade to rules."""


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """A text-completion backend."""

    name: str = "provider"
    available: bool = True

    @abc.abstractmethod
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        ...

    def complete_json(
        self, prompt: str, *, system: str = "", max_tokens: int = 512
    ) -> tuple[Any | None, Completion]:
        """Complete and parse JSON, tolerating the usual wrappers.

        Models wrap JSON in prose and fences constantly. Retrying costs a round
        trip and usually produces the same wrapper, so repair first.
        """
        completion = self.complete(prompt, system=system, max_tokens=max_tokens)
        return extract_json(completion.text), completion


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Pull the first JSON value out of ``text``, repairing common wrappers."""
    candidate = text.strip()
    for attempt in (candidate, _fenced(candidate), _balanced(candidate)):
        if attempt is None:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", attempt)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def _fenced(text: str) -> str | None:
    match = _FENCE.search(text)
    return match.group(1).strip() if match else None


def _balanced(text: str) -> str | None:
    starts = [i for i, c in enumerate(text) if c in "{["]
    if not starts:
        return None
    start = starts[0]
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class NullProvider(Provider):
    """No model. Agents fall back to their rule-based path.

    This is the default so that the workflow, its tests and its CI never
    depend on a model being present.
    """

    name = "null"
    available = False

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        raise ProviderError("no model provider configured")


class EchoProvider(Provider):
    """Deterministic stub used by tests.

    Responses are keyed on a hash of the prompt, so a test that asserts on
    model-assisted behaviour is reproducible. Canned replies can be registered
    by substring for specific scenarios.
    """

    name = "echo"

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        self.calls.append(prompt)
        for needle, response in self.responses.items():
            if needle.lower() in prompt.lower():
                return Completion(text=response, model="echo")
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return Completion(
            text=self.default.replace("{digest}", digest),
            model="echo",
            prompt_tokens=max(1, len(prompt) // 4),
        )


class OllamaProvider(Provider):
    """Local models through Ollama. Free, offline, no account.

        ollama pull llama3.2
        ollama serve
    """

    name = "ollama"

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = _post(f"{self.base_url}/api/generate", payload, self.timeout)
        return Completion(
            text=data.get("response", ""),
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            raw=data,
        )

    def health(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:  # noqa: BLE001 - availability probe
            return False


class OpenAICompatProvider(Provider):
    """Any OpenAI-shaped endpoint: vLLM, llama.cpp, LocalAI, text-generation-inference."""

    name = "openai-compat"

    def __init__(
        self, model: str = "default", base_url: str = "http://localhost:8000",
        api_key: str | None = None, timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            self.timeout,
            self._headers,
        )
        choices = data.get("choices") or [{}]
        usage = data.get("usage") or {}
        return Completion(
            text=(choices[0].get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )


def _post(url: str, payload: dict[str, Any], timeout: float,
          headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"{url} returned {exc.code}: {exc.read()[:200]!r}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(f"{url} unreachable: {exc}") from exc


def build_provider(kind: str = "null", **kwargs: Any) -> Provider:
    providers = {
        "null": NullProvider,
        "none": NullProvider,
        "echo": EchoProvider,
        "ollama": OllamaProvider,
        "openai-compat": OpenAICompatProvider,
        "vllm": OpenAICompatProvider,
    }
    if kind not in providers:
        raise ValueError(f"unknown provider {kind!r}; choose from {sorted(providers)}")
    return providers[kind](**kwargs)
