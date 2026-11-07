"""Make one live Groq request through the project's shared LLM module."""

from __future__ import annotations

import sys

from src.llm import GenerationSettings, GroqLLM


QUESTION = "What should a vehicle owner do after learning about a safety recall?"


def main() -> int:
    settings = GenerationSettings()
    llm = GroqLLM(settings=settings)
    result = llm.generate([{"role": "user", "content": QUESTION}])

    if not result.ok:
        error = result.error
        assert error is not None
        print("Groq inference failed.", file=sys.stderr)
        print(f"Category: {error.category}", file=sys.stderr)
        print(f"Type: {error.error_type}", file=sys.stderr)
        print(f"Message: {error.message}", file=sys.stderr)
        print(f"Model: {result.model}", file=sys.stderr)
        print(f"Latency: {result.latency_seconds:.3f} seconds", file=sys.stderr)
        print(f"Finish reason: {result.finish_reason}", file=sys.stderr)
        print("Token usage:", file=sys.stderr)
        print(f"  Prompt: {result.usage.prompt_tokens}", file=sys.stderr)
        print(f"  Completion: {result.usage.completion_tokens}", file=sys.stderr)
        print(f"  Reasoning: {result.usage.reasoning_tokens}", file=sys.stderr)
        print(f"  Total: {result.usage.total_tokens}", file=sys.stderr)
        if error.status_code is not None:
            print(f"HTTP status: {error.status_code}", file=sys.stderr)
        if error.request_id is not None:
            print(f"Request ID: {error.request_id}", file=sys.stderr)
        print(f"Retryable: {error.retryable}", file=sys.stderr)
        return 1

    print(f"Question: {QUESTION}")
    print(f"Model: {result.model}")
    print(f"Latency: {result.latency_seconds:.3f} seconds")
    print(f"Finish reason: {result.finish_reason}")
    print("Token usage:")
    print(f"  Prompt: {result.usage.prompt_tokens}")
    print(f"  Completion: {result.usage.completion_tokens}")
    print(f"  Reasoning: {result.usage.reasoning_tokens}")
    print(f"  Total: {result.usage.total_tokens}")
    print("Response:")
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
