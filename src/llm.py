"""Shared Groq chat-completion inference for baseline and RAG systems.

This module intentionally contains no prompting, retrieval, or evaluation logic.
Callers provide chat messages and receive a structured result, including sanitized
error information when configuration or API calls fail.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


GROQ_API_KEY_ENV = "GROQ_API_KEY"


@dataclass(frozen=True)
class GenerationSettings:
    """Generation settings shared by every experimental system."""

    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 1024
    seed: int = 42
    stream: bool = False
    n: int = 1

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.n != 1:
            raise ValueError("Groq chat completions currently require n=1")
        if self.stream:
            raise ValueError("stream must be False so usage is returned consistently")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable settings for experiment metadata."""

        return asdict(self)


DEFAULT_GENERATION_SETTINGS = GenerationSettings()


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by Groq, when present in the response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class LLMError:
    """Safe, structured information about a failed inference request."""

    category: str
    error_type: str
    message: str
    status_code: int | None = None
    request_id: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResult:
    """Result of one chat-completion attempt."""

    text: str | None
    model: str
    usage: TokenUsage
    latency_seconds: float
    response_id: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without credentials."""

        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_seconds": self.latency_seconds,
            "response_id": self.response_id,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "error": self.error.to_dict() if self.error else None,
        }


class _ConfigurationFailure(Exception):
    """Internal marker for local configuration failures."""


class _ResponseFailure(Exception):
    """Internal marker for malformed or incomplete API responses."""


class GroqLLM:
    """Reusable Groq inference client with shared, explicit settings.

    ``client`` is injectable so tests can supply a mock without importing the
    Groq SDK or making network calls. When it is omitted, the client is created
    lazily from ``GROQ_API_KEY`` on the first call to :meth:`generate`.
    """

    def __init__(
        self,
        settings: GenerationSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or DEFAULT_GENERATION_SETTINGS
        self._client = client

    def generate(self, messages: Sequence[Mapping[str, str]]) -> LLMResult:
        """Generate one response from caller-provided chat messages.

        API, network, authentication, configuration, and response-shape failures
        are returned in ``LLMResult.error`` rather than raised. Messages are
        validated before sending, and no credential is included in the result.
        """

        started = time.perf_counter()

        try:
            normalized_messages = _normalize_messages(messages)
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.settings.model,
                messages=normalized_messages,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                max_completion_tokens=self.settings.max_completion_tokens,
                seed=self.settings.seed,
                stream=self.settings.stream,
                n=self.settings.n,
            )

            choices = getattr(response, "choices", None)
            if not choices:
                raise _ResponseFailure("Groq returned no completion choices")

            choice = choices[0]
            message = getattr(choice, "message", None)
            text = getattr(message, "content", None)
            if text is None:
                raise _ResponseFailure("Groq returned a choice without message content")

            return LLMResult(
                text=text,
                model=getattr(response, "model", None) or self.settings.model,
                usage=_extract_usage(response),
                latency_seconds=time.perf_counter() - started,
                response_id=getattr(response, "id", None),
                system_fingerprint=getattr(response, "system_fingerprint", None),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except Exception as exc:  # Convert expected SDK/runtime failures to data.
            return LLMResult(
                text=None,
                model=self.settings.model,
                usage=TokenUsage(),
                latency_seconds=time.perf_counter() - started,
                error=_structured_error(exc),
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.getenv(GROQ_API_KEY_ENV)
        if not api_key:
            raise _ConfigurationFailure(
                f"Required environment variable {GROQ_API_KEY_ENV} is not set"
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise _ConfigurationFailure(
                "The 'groq' package is not installed; install project requirements"
            ) from exc

        self._client = Groq(api_key=api_key)
        return self._client


def _normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must contain at least one chat message")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is invalid")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        normalized.append({"role": role, "content": content})

    return normalized


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _structured_error(exc: Exception) -> LLMError:
    status_code = _status_code(exc)
    request_id = _request_id(exc)
    category = _error_category(exc, status_code)
    return LLMError(
        category=category,
        error_type=type(exc).__name__,
        message=_sanitize_error_message(str(exc)),
        status_code=status_code,
        request_id=request_id,
        retryable=_is_retryable(exc, status_code),
    )


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def _request_id(exc: Exception) -> str | None:
    request_id = getattr(exc, "request_id", None)
    if request_id:
        return str(request_id)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        value = headers.get("x-request-id") or headers.get("X-Request-Id")
        return str(value) if value else None
    return None


def _error_category(exc: Exception, status_code: int | None) -> str:
    if isinstance(exc, _ConfigurationFailure):
        return "configuration"
    if isinstance(exc, ValueError):
        return "validation"
    if isinstance(exc, _ResponseFailure):
        return "response"
    if status_code in {401, 403}:
        return "authentication"
    if status_code == 429:
        return "rate_limit"
    if status_code == 408 or (status_code is not None and status_code >= 500):
        return "transient_api"

    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connection" in name:
        return "network"
    return "api"


def _is_retryable(exc: Exception, status_code: int | None) -> bool:
    if status_code in {408, 409, 429}:
        return True
    if status_code is not None and status_code >= 500:
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


_SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)(api[_-]?key\s*[:=]\s*)[\"']?[^,\s\"']+[\"']?"
    ),
)


def _sanitize_error_message(message: str) -> str:
    """Redact credential-shaped text without retaining raw API responses."""

    safe = message or "Inference request failed"
    api_key = os.getenv(GROQ_API_KEY_ENV)
    if api_key:
        safe = safe.replace(api_key, "[REDACTED]")

    safe = _SECRET_PATTERNS[0].sub("gsk_[REDACTED]", safe)
    safe = _SECRET_PATTERNS[1].sub(r"\1[REDACTED]", safe)
    safe = _SECRET_PATTERNS[2].sub(r"\1[REDACTED]", safe)

    # Exception strings can include large response bodies; retain only a concise
    # diagnostic message and never serialize raw response objects or headers.
    return safe[:1000]


__all__ = [
    "DEFAULT_GENERATION_SETTINGS",
    "GROQ_API_KEY_ENV",
    "GenerationSettings",
    "GroqLLM",
    "LLMError",
    "LLMResult",
    "TokenUsage",
]
