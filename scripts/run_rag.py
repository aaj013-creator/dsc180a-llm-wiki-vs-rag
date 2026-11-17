"""Run the RAG system on every question and save results in the baseline's format.

Usage:
    python -m scripts.run_rag --retriever dense --questions data/evaluation/questions.json
    python -m scripts.run_rag --retriever bm25 --questions data/evaluation/questions.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from groq import Groq

from src.rag import (
    BM25_B,
    BM25_EPSILON,
    BM25_K1,
    BM25_TOKENIZER,
    CHAT_MODEL,
    DOCUMENT_PREFIX,
    EMBEDDINGS_PATH,
    EMBED_MODEL,
    OLLAMA,
    build_bm25_index,
    build_docs,
    build_index,
    embed,
    load_data,
    retrieve_bm25,
)

TOP_K = 5

# Same generation settings as the baseline, so the only difference is retrieval
GEN_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "seed": 42,
}

# Same prompt as ask_llm_rag in src/rag.py
PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}"""


def retrieve_dense_with_ids(query, texts, vectors, campaign_ids, k=TOP_K):
    """Run the existing dense retrieval and include aligned IDs and scores."""
    q = embed([query])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    return (
        [texts[i] for i in top],
        [campaign_ids[i] for i in top],
        [float(scores[i]) for i in top],
    )


def retrieve_bm25_with_ids(query, texts, index, campaign_ids, k=TOP_K):
    """Run BM25 retrieval and include aligned document text, IDs, and scores."""
    ranked = retrieve_bm25(query, index, k=k)
    return (
        [texts[i] for i, _ in ranked],
        [campaign_ids[i] for i, _ in ranked],
        [score for _, score in ranked],
    )


def call_groq(client, prompt, retries=3):
    """Call Groq, retrying a few times if we hit a rate limit or network error."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **GEN_SETTINGS,
            )
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  Groq error ({exc}); retrying in 30 seconds")
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25"),
        default="dense",
        help="retrieval method (default: dense)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "result path; defaults to results/rag_<retriever>_results.json so "
            "dense and BM25 runs remain separate"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing result file",
    )
    args = parser.parse_args()

    output_path = args.output or Path(f"results/rag_{args.retriever}_results.json")
    if output_path.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {output_path}; choose another path or pass --overwrite"
        )

    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = payload["questions"]

    print("Loading data and building documents...")
    df = load_data()
    docs = build_docs(df)
    campaign_ids = [doc["id"] for doc in docs]
    assert len(campaign_ids) == len(docs), "document and campaign ID counts differ"

    if args.retriever == "dense":
        print(f"Embedding {len(docs)} documents (cached after the first run)...")
        texts, retrieval_index = build_index(docs)
        retrieval_config = {
            "method": "dense",
            "top_k": TOP_K,
            "document_prefix": DOCUMENT_PREFIX,
            "embedding_model": EMBED_MODEL,
            "embedding_endpoint": OLLAMA,
            "similarity": "cosine",
            "embeddings_cache": str(EMBEDDINGS_PATH),
            "query_prefix": None,
        }
    else:
        print(f"Building BM25 index for {len(docs)} documents...")
        # This branch intentionally never calls build_index/embed and therefore
        # never contacts Ollama or loads embeddings.npy.
        texts, retrieval_index = build_bm25_index(docs)
        retrieval_config = {
            "method": "bm25",
            "implementation": "rank_bm25.BM25Okapi",
            "top_k": TOP_K,
            "document_prefix": DOCUMENT_PREFIX,
            "k1": BM25_K1,
            "b": BM25_B,
            "epsilon": BM25_EPSILON,
            "tokenizer": BM25_TOKENIZER,
            "tie_breaker": "ascending_document_index",
            "no_match_behavior": (
                "return first min(top_k, corpus_size) documents in source order "
                "with score 0.0"
            ),
        }

    client = Groq()
    results = []
    for item in questions:
        print(f"{item['id']}: {item['question'][:70]}")
        if args.retriever == "dense":
            context_texts, retrieved_ids, retrieval_scores = retrieve_dense_with_ids(
                item["question"], texts, retrieval_index, campaign_ids
            )
        else:
            context_texts, retrieved_ids, retrieval_scores = retrieve_bm25_with_ids(
                item["question"], texts, retrieval_index, campaign_ids
            )
        prompt = PROMPT_TEMPLATE.format(
            context="\n\n".join(context_texts), question=item["question"])

        result = {
            "id": item["id"],
            "question": item["question"],
            "metadata": {k: v for k, v in item.items() if k not in ("id", "question")},
            "retrieved_ids": retrieved_ids,
            "retrieval_method": args.retriever,
            "retrieval_scores": retrieval_scores,
            "ranked_retrieval": [
                {
                    "rank": rank,
                    "campaign_id": campaign_id,
                    "score": score,
                }
                for rank, (campaign_id, score) in enumerate(
                    zip(retrieved_ids, retrieval_scores), start=1
                )
            ],
        }
        try:
            start = time.time()
            response = call_groq(client, prompt)
            result.update({
                "status": "success",
                "answer": response.choices[0].message.content,
                "model": response.model,
                "latency_seconds": round(time.time() - start, 3),
                "usage": response.usage.model_dump() if response.usage else None,
                "error": None,
            })
        except Exception as exc:
            result.update({"status": "error", "answer": None, "usage": None,
                           "error": str(exc)})
        results.append(result)
        time.sleep(2)  # stay under Groq's free-tier rate limits

    output = {
        "run": {
            "system": "rag",
            "model": CHAT_MODEL,
            "top_k": TOP_K,
            "retrieval": retrieval_config,
            "generation_settings": GEN_SETTINGS,
            "prompt_template": PROMPT_TEMPLATE,
            "questions_file": str(args.questions),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
