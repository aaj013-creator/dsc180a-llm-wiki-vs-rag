"""Run the RAG system on every question and save results in the baseline's format.

Usage:
    python -m scripts.run_rag --questions data/evaluation/questions.json --output results/rag_results.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from groq import Groq

from src.rag import CHAT_MODEL, build_docs, build_index, embed, load_data

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


def retrieve_with_ids(query, texts, vectors, campaign_ids, k=TOP_K):
    """Same cosine-similarity retrieval as src/rag.py, but also returns campaign IDs."""
    q = embed([query])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    return [texts[i] for i in top], [campaign_ids[i] for i in top]


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.questions.read_text())
    questions = payload["questions"]

    print("Loading data and building documents...")
    df = load_data()
    docs = build_docs(df)
    # build_docs groups by CAMPNO, so documents are in the same order as these IDs
    campaign_ids = [campno for campno, _ in df.groupby("CAMPNO")]
    assert len(campaign_ids) == len(docs), "document and campaign ID counts differ"

    print(f"Embedding {len(docs)} documents (cached after the first run)...")
    texts, vectors = build_index(docs)

    client = Groq()
    results = []
    for item in questions:
        print(f"{item['id']}: {item['question'][:70]}")
        context_texts, retrieved_ids = retrieve_with_ids(
            item["question"], texts, vectors, campaign_ids)
        prompt = PROMPT_TEMPLATE.format(
            context="\n\n".join(context_texts), question=item["question"])

        result = {
            "id": item["id"],
            "question": item["question"],
            "metadata": {k: v for k, v in item.items() if k not in ("id", "question")},
            "retrieved_ids": retrieved_ids,
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
            "generation_settings": GEN_SETTINGS,
            "prompt_template": PROMPT_TEMPLATE,
            "questions_file": str(args.questions),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()