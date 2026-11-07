"""Command-line runner for the LLM-only evaluation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm import GenerationSettings, GroqLLM


BASELINE_PROMPT_TEMPLATE = """Answer this automotive safety recall question without external documents or retrieved context.
Use only your existing knowledge. If you do not know, say "I don't know" rather than inventing information.

Question: {question}"""


def load_questions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load questions from a JSON list or an object containing ``questions``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        dataset_metadata: dict[str, Any] = {}
        raw_questions = payload
    elif isinstance(payload, Mapping):
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            raise ValueError("question JSON object must contain a 'questions' list")
        dataset_metadata = {
            key: value for key, value in payload.items() if key != "questions"
        }
    else:
        raise ValueError("question JSON must be a list or an object with 'questions'")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_questions, start=1):
        default_id = f"q_{index:04d}"
        if isinstance(item, str):
            question_id = default_id
            question = item
            metadata: dict[str, Any] = {}
        elif isinstance(item, Mapping):
            question_id = str(item.get("id", default_id))
            question = item.get("question")
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"id", "question"}
            }
        else:
            raise ValueError(f"question {index} must be a string or JSON object")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {index} must contain non-empty text")
        if not question_id.strip():
            raise ValueError(f"question {index} must have a non-empty id")
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id: {question_id}")

        seen_ids.add(question_id)
        questions.append(
            {
                "id": question_id,
                "question": question.strip(),
                "metadata": metadata,
            }
        )

    return dataset_metadata, questions


def run_baseline(
    questions_path: Path,
    output_path: Path,
    llm: GroqLLM | None = None,
) -> dict[str, Any]:
    """Run every question without retrieval and write a JSON results file."""

    dataset_metadata, questions = load_questions(questions_path)
    inference = llm or GroqLLM(settings=GenerationSettings())
    started_at = _utc_now()
    results: list[dict[str, Any]] = []

    for item in questions:
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=item["question"])
        response = inference.generate([{"role": "user", "content": prompt}])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "metadata": item["metadata"],
                "status": "success" if response.ok else "failure",
                "answer": response.text,
                "model": response.model,
                "latency_seconds": response.latency_seconds,
                "usage": response.usage.to_dict(),
                "finish_reason": response.finish_reason,
                "response_id": response.response_id,
                "system_fingerprint": response.system_fingerprint,
                "error": response.error.to_dict() if response.error else None,
            }
        )

    success_count = sum(result["status"] == "success" for result in results)
    output = {
        "run": {
            "system": "llm_only_baseline",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "questions_file": str(questions_path),
            "question_count": len(questions),
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "generation_settings": inference.settings.to_dict(),
            "prompt_template": BASELINE_PROMPT_TEMPLATE,
            "dataset_metadata": dataset_metadata,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="JSON file containing evaluation questions",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON file to receive baseline results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_baseline(args.questions, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Baseline setup failed: {exc}", file=sys.stderr)
        return 2

    run = output["run"]
    print(f"Saved {run['question_count']} results to {args.output}")
    print(f"Successes: {run['success_count']}; failures: {run['failure_count']}")
    return 0 if run["failure_count"] == 0 else 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
