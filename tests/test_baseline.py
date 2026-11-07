"""Offline tests for the LLM-only baseline runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.baseline import load_questions, run_baseline
from src.llm import GroqLLM


class FakeRateLimitError(Exception):
    status_code = 429
    request_id = "req_mock_rate_limit"


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.call_count += 1
        if self.call_count == 2:
            raise FakeRateLimitError("Bearer gsk_mock_secret was rate limited")

        return SimpleNamespace(
            id="resp_mock_success",
            model=kwargs["model"],
            system_fingerprint="fp_mock",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Mock recall answer."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=12,
                total_tokens=32,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
            ),
        )


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_directory = Path(__file__).parent / ".baseline_test_tmp"
        self.test_directory.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for path in self.test_directory.iterdir():
            path.unlink()
        self.test_directory.rmdir()

    def test_runner_records_success_and_structured_failure(self) -> None:
        questions_path = self.test_directory / "questions.json"
        output_path = self.test_directory / "results.json"
        questions_path.write_text(
            json.dumps(
                {
                    "dataset_name": "mock_evaluation",
                    "questions": [
                        {
                            "id": "q1",
                            "question": "What should an owner do after a recall?",
                            "category": "lookup",
                        },
                        {
                            "id": "q2",
                            "question": "Was this fictional vehicle recalled?",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        output = run_baseline(
            questions_path,
            output_path,
            llm=GroqLLM(client=fake_client),
        )

        self.assertTrue(output_path.is_file())
        self.assertEqual(output["run"]["success_count"], 1)
        self.assertEqual(output["run"]["failure_count"], 1)
        self.assertEqual(
            output["run"]["generation_settings"]["reasoning_effort"], "low"
        )
        self.assertEqual(
            output["run"]["generation_settings"]["max_completion_tokens"], 4096
        )

        success, failure = output["results"]
        self.assertEqual(success["status"], "success")
        self.assertEqual(success["answer"], "Mock recall answer.")
        self.assertEqual(success["usage"]["reasoning_tokens"], 5)
        self.assertEqual(success["metadata"]["category"], "lookup")

        self.assertEqual(failure["status"], "failure")
        self.assertEqual(failure["error"]["category"], "rate_limit")
        self.assertTrue(failure["error"]["retryable"])
        self.assertNotIn("gsk_mock_secret", failure["error"]["message"])

        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[0]["reasoning_effort"], "low")
        self.assertEqual(completions.calls[0]["max_completion_tokens"], 4096)

        saved_output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_output, output)

    def test_loader_accepts_a_top_level_question_list(self) -> None:
        questions_path = self.test_directory / "questions.json"
        questions_path.write_text(
            json.dumps(
                ["Question one", {"id": "custom", "question": "Question two"}]
            ),
            encoding="utf-8",
        )

        metadata, questions = load_questions(questions_path)

        self.assertEqual(metadata, {})
        self.assertEqual(questions[0]["id"], "q_0001")
        self.assertEqual(questions[1]["id"], "custom")


if __name__ == "__main__":
    unittest.main()
