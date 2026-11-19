"""Offline tests for chunking strategies and chunking-evaluation metrics"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.chunking import (
    STRATEGIES,
    Chunk,
    build_records,
    compress_vehicles,
    fixed_split,
    make_chunks,
    recursive_split,
)
from src.evaluate_chunking import cited_campaigns, retrieval_metrics
from src.rag import COLUMNS, select_context


def _row(campno, make, model, year, **fields):
    row = {c: "" for c in COLUMNS}
    row.update(CAMPNO=campno, MAKETXT=make, MODELTXT=model, YEARTXT=year,
               MFGNAME="Honda (American Honda Motor Co.)", RCDATE="20210325", **fields)
    return row


FUEL = dict(
    COMPNAME="FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP",
    DESC_DEFECT="Honda is recalling certain 2018-2019 CR-V vehicles.  The low-pressure fuel pump "
                "inside the fuel tank may fail. " * 3,
    CONEQUENCE_DEFECT="If the fuel pump fails, the engine can stall while driving, increasing the risk of a crash.",
    CORRECTIVE_ACTION="Dealers will replace the fuel pump assembly, free of charge. Owner letters were "
                      "mailed May 18, 2021. Note: This recall is an expansion of recall 20V-314.",
)
FRAME = dict(
    COMPNAME="STRUCTURE:FRAME AND MEMBERS",
    DESC_DEFECT="The rear subframe bolts may loosen, allowing the rear subframe to separate.",
    CONEQUENCE_DEFECT="A separated subframe increases the risk of a crash.",
    CORRECTIVE_ACTION="Dealers will inspect the rear subframe bolts and retorque or replace them.",
)


def fake_embed(texts):
    """Deterministic bag-of-words embedding so semantic chunking runs offline"""
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vectors[i, hash(word) % 64] += 1
    vectors += 1e-6
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [_row("21V215000", "HONDA", "CR-V", y, **FUEL) for y in ("2018", "2019")]
        rows += [_row("21V215000", "ACURA", "RDX", "2019", **FUEL)]
        rows += [_row("19V865000", "HONDA", "CR-V", y, **FRAME) for y in ("2019", "2020")]
        cls.records = build_records(pd.DataFrame(rows))

    def test_records_one_per_campaign_with_original_doc(self):
        self.assertEqual([r["campaign"] for r in self.records], ["19V865000", "21V215000"])
        doc = self.records[1]["doc"]
        self.assertTrue(doc.startswith("Recall campaign 21V215000.\n"))
        self.assertIn("Affected vehicles: ACURA RDX 2019; HONDA CR-V 2018; HONDA CR-V 2019.", doc)
        self.assertTrue(doc.endswith("\nReported to NHTSA: 2021-03-25."))

    def test_compress_vehicles_collapses_year_ranges(self):
        vehicles = [("HONDA", "CR-V", y) for y in ("2017", "2018", "2019", "2021")]
        vehicles += [("ACURA", "RDX", "2019"), ("ACME", "TIRE", "9999")]
        self.assertEqual(compress_vehicles(vehicles),
                         "ACME TIRE unknown year; ACURA RDX 2019; HONDA CR-V 2017-2019, 2021")

    def test_splitters_respect_size_and_keep_all_words(self):
        text = " ".join(f"word{i}." for i in range(400))
        for chunks in (fixed_split(text, 120, 30), recursive_split(text, 120, 30)):
            self.assertTrue(all(len(c) <= 120 for c in chunks))
            words = set(" ".join(chunks).split())
            self.assertEqual(words, set(text.split()))

    def test_every_strategy_produces_valid_chunks(self):
        campaigns = {r["campaign"] for r in self.records}
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                chunks = make_chunks(name, self.records, embed_fn=fake_embed)
                self.assertTrue(chunks)
                self.assertEqual(len({c.id for c in chunks}), len(chunks), "chunk ids must be unique")
                self.assertTrue(all(c.text.strip() for c in chunks))
                covered = {cid for c in chunks for cid in c.campaigns}
                self.assertEqual(covered, campaigns)

    def test_small_chunk_strategies_split_and_ctx_variants_add_headers(self):
        plain = make_chunks("recursive", self.records, size=200)
        ctx = make_chunks("recursive_ctx", self.records, size=200)
        self.assertGreater(len(plain), len(self.records))
        self.assertEqual(len(plain), len(ctx))
        self.assertTrue(all(c.text.startswith("Recall ") for c in ctx))
        self.assertTrue(any("21V215000" not in c.text for c in plain if "21V215000" in c.campaigns))

    def test_adaptive_keeps_short_campaigns_whole(self):
        chunks = make_chunks("adaptive", self.records)
        self.assertEqual([c.id for c in chunks], ["19V865000", "21V215000"])
        self.assertIn("HONDA CR-V 2018-2019", chunks[1].text)

    def test_vehicle_summaries_list_every_recall_for_the_vehicle(self):
        chunks = make_chunks("vehicle", self.records)
        summary = next(c for c in chunks if c.id == "vehicle:2019 HONDA CR-V#0")
        self.assertEqual(set(summary.campaigns), {"19V865000", "21V215000"})
        self.assertIn("2 recall campaign(s)", summary.text)
        self.assertNotIn("is recalling", summary.text)


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        ranked = [("A",), ("X", "B"), ("Y",), ("C",)]
        m = retrieval_metrics(ranked, ["A", "B"], ["B", "C"])
        self.assertEqual(m["hit@1"], 0.0)
        self.assertEqual(m["recall@3"], 0.5)
        self.assertEqual(m["recall@5"], 1.0)
        self.assertEqual(m["mrr"], 0.5)
        self.assertEqual(m["ctx_recall"], 0.5)

    def test_cited_campaigns_normalizes_formats(self):
        text = "See 21V-215 and 23V858000; also 20V 314."
        self.assertEqual(cited_campaigns(text), ["20V314000", "21V215000", "23V858000"])

    def test_load_questions_csv_and_json_agree(self):
        import json
        import tempfile
        from pathlib import Path
        from src.evaluate_chunking import load_questions

        csv_text = ("id,task_type,question,gold_answer,source_records\n"
                    'q1,lookup,"Q one?","A",19V865000;20V314000\n'
                    "\n"  # main's questions.csv has a blank line
                    'q2,unanswerable,"Q two?","Not in corpus",\n'
                    'q3,aggregate,"Q three?","19",\n')
        payload = {"questions": [
            {"id": "q1", "task_type": "lookup", "question": "Q one?", "gold_answer": "A",
             "source_records": ["19V865000", "20V314000"]},
            {"id": "q2", "task_type": "unanswerable", "question": "Q two?",
             "gold_answer": "Not in corpus", "source_records": []},
            {"id": "q3", "task_type": "aggregate", "question": "Q three?", "gold_answer": "19",
             "source_records": []},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = Path(tmp, "q.csv"), Path(tmp, "q.json")
            csv_path.write_text(csv_text)
            json_path.write_text(json.dumps(payload))
            from_csv, from_json = load_questions(csv_path), load_questions(json_path)
        self.assertEqual(from_csv, from_json)
        self.assertEqual([q["gold_campaigns"] for q in from_csv],
                         [["19V865000", "20V314000"], [], []])

    def test_token_throttle_waits_when_window_is_full(self):
        from unittest import mock
        import src.rag as rag
        clock = [1000.0]
        with mock.patch.object(rag.time, "time", side_effect=lambda: clock[0]), \
             mock.patch.object(rag.time, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)):
            throttle = rag.TokenThrottle(tokens_per_minute=5000)
            throttle.wait(3000)
            throttle.record(3000)
            self.assertEqual(clock[0], 1000.0)  # fits, no wait
            throttle.wait(3000)                 # 6000 > 5000: wait for the window to clear
            self.assertGreaterEqual(clock[0], 1060.0)

    def test_ollama_llm_maps_response_and_errors(self):
        from types import SimpleNamespace
        from unittest import mock
        import requests
        from src.llm import OllamaLLM

        payload = {"model": "gpt-oss:20b", "message": {"content": "Recall A.", "thinking": "..."},
                   "prompt_eval_count": 1600, "eval_count": 150, "done_reason": "stop"}
        ok = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
        llm = OllamaLLM()
        with mock.patch.object(requests, "post", return_value=ok) as post:
            result = llm.generate([{"role": "user", "content": "q"}])
        sent = post.call_args.kwargs["json"]
        self.assertEqual((sent["model"], sent["think"], sent["options"]["temperature"]),
                         ("gpt-oss:20b", "low", 0.0))
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "Recall A.")
        self.assertEqual(result.usage.total_tokens, 1750)

        with mock.patch.object(requests, "post", side_effect=requests.ConnectionError("refused")):
            result = llm.generate([{"role": "user", "content": "q"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.error.category, "network")

    def test_local_llm_is_not_paced(self):
        from unittest import mock
        import src.rag as rag
        from src.llm import LLMResult, OllamaLLM, TokenUsage

        llm = OllamaLLM()
        done = LLMResult("ok", "gpt-oss:20b", TokenUsage(total_tokens=9000), 0.0)
        with mock.patch.object(llm, "generate", return_value=done), \
             mock.patch.object(rag.TokenThrottle, "wait") as wait:
            for _ in range(3):
                rag.ask_llm("x" * 20000, llm=llm)
        wait.assert_not_called()

    def test_select_context_respects_budget(self):
        results = [(Chunk(str(i), f"{i:03d}" + "x" * 397, ("A",)), 1.0) for i in range(10)]
        self.assertEqual(len(select_context(results, max_chars=1000)), 2)
        self.assertEqual(len(select_context(results, max_chars=10)), 1)
        duplicates = [(Chunk(str(i), "same text", ("A",)), 1.0) for i in range(3)]
        self.assertEqual(len(select_context(duplicates)), 1)


class GenerationFailureTests(unittest.TestCase):
    """API failures must be missing data, never scored as wrong answers"""

    def setUp(self):
        from types import SimpleNamespace
        import src.evaluate_chunking as ev
        from src.rag import Index

        ev.JUDGE_STATE.update(consecutive_failures=0, disabled_reason=None)
        self.addCleanup(ev.JUDGE_STATE.update, consecutive_failures=0, disabled_reason=None)

        chunks = [Chunk("A", "Recall A text", ("A",)), Chunk("B", "Recall B text", ("B",))]
        self.index = Index("fake", chunks, np.eye(2, dtype=np.float32))
        self.cache = SimpleNamespace(embed=lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))
        self.questions = [{"id": f"q{i}", "task_type": "lookup", "question": "?",
                           "gold_answer": "A", "gold_campaigns": ["A"]} for i in range(4)]

    def _result(self, ok, text="Recall A. [A]"):
        from src.llm import LLMError, LLMResult, TokenUsage
        error = None if ok else LLMError("rate_limit", "RateLimitError", "429 too many", retryable=True)
        return LLMResult(text if ok else None, "m", TokenUsage(prompt_tokens=5), 0.0, error=error)

    def _run(self, outcomes):
        from unittest import mock
        import src.evaluate_chunking as ev
        calls = iter(outcomes)
        with mock.patch.object(ev, "build_index", return_value=self.index), \
             mock.patch.object(ev, "ask_llm", side_effect=lambda prompt, **kw: next(calls)):
            return ev.evaluate_strategy("fake", [], self.questions, self.cache, 6000, generate=True)

    def test_single_failure_is_excluded_not_scored_zero(self):
        judge_ok = self._result(True, '{"score": 1, "reason": "ok"}')
        outcomes = [self._result(False)] + [self._result(True), judge_ok] * 3
        _, rows = self._run(outcomes)
        self.assertIsNone(rows[0].get("correct"))
        self.assertEqual(rows[0]["error"], "429 too many")
        self.assertEqual([r["correct"] for r in rows[1:]], [1.0, 1.0, 1.0])

    def test_grading_rules_match_team_grader(self):
        from unittest import mock
        import src.evaluate_chunking as ev
        unanswerable = {"task_type": "unanswerable", "question": "?", "gold_answer": "Not in corpus"}
        answerable = {"task_type": "lookup", "question": "?", "gold_answer": "A"}
        with mock.patch.object(ev, "ask_llm") as judge_call:
            self.assertEqual(ev.grade(unanswerable, "I don't know.")[0], 1.0)
            self.assertEqual(ev.grade(unanswerable, "It is recall 19V865000.")[0], 0.0)
            self.assertEqual(ev.grade(answerable, "The context does not say; I don't know.")[0], 0.0)
            judge_call.assert_not_called()
            judge_call.return_value = self._result(True, 'Sure: {"score": 0.5, "reason": "partly"}')
            self.assertEqual(ev.grade(answerable, "Recall A."), (0.5, "partly", False))

    def test_judge_outage_keeps_generating_and_regrade_fills_scores(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock
        import src.evaluate_chunking as ev

        answer = self._result(True)
        judge_down = self._result(False)
        outcomes = [answer, judge_down] * 3 + [answer]  # 3 judge failures, then judge is skipped
        _, rows = self._run(outcomes)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["answer"] for r in rows))
        self.assertTrue(all(r["correct"] is None for r in rows))
        self.assertEqual(rows[3]["judge_reason"], ev.UNGRADED_REASON)
        self.assertEqual(ev.count_ungraded(rows), 4)

        # Later, with the quota back, --regrade scores only the missing rows
        ev.JUDGE_STATE.update(consecutive_failures=0, disabled_reason=None)
        rows[0]["correct"] = 1.0
        run = {"strategies": [{"strategy": "fake", "n_chunks": 2, "chunk_chars_mean": 13.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "results.json")
            path.write_text(json.dumps({"run": run, "summary": [], "results": rows}))
            judge_ok = self._result(True, '{"score": 0.5, "reason": "partly"}')
            with mock.patch.object(ev, "ask_llm", return_value=judge_ok) as judge_call:
                remaining = ev.regrade(path)
            saved = json.loads(path.read_text())
        self.assertEqual(remaining, 0)
        self.assertEqual(judge_call.call_count, 3)
        self.assertEqual([r["correct"] for r in saved["results"]], [1.0, 0.5, 0.5, 0.5])
        self.assertEqual(saved["run"]["ungraded"], 0)

    def test_consecutive_failures_abort_with_partial_rows(self):
        import src.evaluate_chunking as ev
        with self.assertRaises(ev.GenerationAborted) as ctx:
            self._run([self._result(False)] * 4)
        self.assertEqual(len(ctx.exception.rows), 2)
        self.assertEqual(ctx.exception.info["strategy"], "fake")


if __name__ == "__main__":
    unittest.main()
