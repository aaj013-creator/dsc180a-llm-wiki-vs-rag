"""Offline tests for BM25 recall retrieval."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.rag import build_bm25_index, retrieve_bm25, tokenize_bm25


def synthetic_docs():
    """Return documents whose order intentionally differs from campaign order."""
    return [
        {
            "id": "19V865000",
            "text": (
                "Recall campaign 19V865000. Rear subframe bolts may loosen. "
                "Dealers will inspect and retorque the bolts."
            ),
        },
        {
            "id": "21V215000",
            "text": (
                "Recall campaign 21V215000. A fuel pump may fail and stall "
                "the engine."
            ),
        },
        {
            "id": "23V158000",
            "text": (
                "Recall campaign 23V158000. Front seat belt buckles may not "
                "latch."
            ),
        },
        {
            "id": "19V569000",
            "text": (
                "Recall campaign 19V569000. A fuel tank weld may separate and "
                "leak."
            ),
        },
    ]


class BM25RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = synthetic_docs()
        self.texts, self.index = build_bm25_index(self.docs)

    def test_tokenizer_normalizes_campaign_id_variants(self):
        tokens = tokenize_bm25("21V215000 21V-215 21v215 Fuel-Pump")
        self.assertEqual(
            tokens,
            ["21v215000", "21v215000", "21v215000", "fuel", "pump"],
        )

    def test_full_hyphenated_and_abbreviated_ids_rank_same_document_first(self):
        for query in ("21V215000", "21V-215", "21v215"):
            with self.subTest(query=query):
                ranked = retrieve_bm25(query, self.index, k=5)
                top_index, top_score = ranked[0]
                self.assertEqual(self.docs[top_index]["id"], "21V215000")
                self.assertGreater(top_score, 0.0)

    def test_ranked_indices_keep_ids_and_documents_aligned(self):
        ranked = retrieve_bm25("rear subframe retorque", self.index, k=2)
        top_index, _ = ranked[0]

        self.assertEqual(self.docs[top_index]["id"], "19V865000")
        self.assertIn("19V865000", self.texts[top_index])
        self.assertTrue(self.texts[top_index].startswith("search_document: "))

    def test_no_match_ranking_is_deterministic_source_order(self):
        first = retrieve_bm25("quasar nebula", self.index, k=3)
        second = retrieve_bm25("quasar nebula", self.index, k=3)

        self.assertEqual(first, second)
        self.assertEqual([index for index, _ in first], [0, 1, 2])
        self.assertEqual([score for _, score in first], [0.0, 0.0, 0.0])

    def test_fewer_documents_than_k_returns_each_document_once(self):
        ranked = retrieve_bm25("fuel", self.index, k=10)

        self.assertEqual(len(ranked), len(self.docs))
        self.assertEqual(len({index for index, _ in ranked}), len(self.docs))

    def test_bm25_path_does_not_use_ollama_or_dense_cache(self):
        with (
            patch("src.rag.embed", side_effect=AssertionError("Ollama was called")),
            patch("src.rag.np.load", side_effect=AssertionError("cache was loaded")),
        ):
            texts, index = build_bm25_index(self.docs)
            ranked = retrieve_bm25("seat belt buckle", index, k=5)

        self.assertEqual(len(texts), len(self.docs))
        self.assertEqual(self.docs[ranked[0][0]]["id"], "23V158000")


if __name__ == "__main__":
    unittest.main()
