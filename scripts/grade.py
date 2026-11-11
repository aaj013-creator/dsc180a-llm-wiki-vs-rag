"""Grade a results file against the gold answers stored in its metadata.

Usage:
    python scripts/grade.py results/baseline_results.json
"""

import argparse
import json
import os
import re
from pathlib import Path

from groq import Groq

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

# Matches campaign IDs like 19V569000, 20V-314, or 21V215
CAMPAIGN_RE = re.compile(r"\b\d{2}V-?\d{3}(?:\d{3})?\b", re.IGNORECASE)

ABSTAIN_RE = re.compile(
    r"(don['’]t know|do not know|not sure|cannot (?:find|determine|answer)"
    r"|no information|not (?:in|found in) the (?:context|corpus|records))",
    re.IGNORECASE,
)

JUDGE_PROMPT = """You are grading an answer to a question about vehicle safety recalls.

Question: {question}
Reference answer: {gold}
Model answer: {answer}

Score the model answer against the reference answer:
- 1 if it contains the key facts of the reference answer and nothing that contradicts it
- 0.5 if it is partly correct or is missing important facts
- 0 if it is wrong, contradicts the reference, or does not answer the question

Respond with only JSON: {{"score": <0, 0.5, or 1>, "reason": "<one sentence>"}}"""


def normalize_id(campaign_id):
    """Turn 20V-314 or 20V314 into 20V314000 so IDs can be compared."""
    cid = campaign_id.replace("-", "").upper()
    return cid + "000" if len(cid) == 6 else cid


def cited_ids(result):
    """Campaign IDs the system used: retrieved IDs if present, else IDs in the answer."""
    if result.get("retrieved_ids"):
        return {normalize_id(x) for x in result["retrieved_ids"]}
    return {normalize_id(x) for x in CAMPAIGN_RE.findall(result.get("answer") or "")}


def judge(client, model, question, gold, answer):
    """Ask the LLM judge to score one answer. Returns (score, reason)."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, gold=gold, answer=answer)}],
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return float(data["score"]), data.get("reason", "")
    except (AttributeError, ValueError, KeyError):
        return None, f"Could not parse judge output: {text[:200]}"


def grade_one(result, client, judge_model):
    meta = result["metadata"]
    task = meta["task_type"]
    answer = (result.get("answer") or "").strip()
    abstained = (not answer) or bool(ABSTAIN_RE.search(answer))

    if result.get("status") != "success":
        score, reason = None, f"Run failed: {result.get('error')}"
    elif task == "unanswerable":
        score = 1.0 if abstained else 0.0
        reason = "Correctly abstained" if abstained else "Answered an unanswerable question"
    elif abstained:
        score, reason = 0.0, "Abstained on an answerable question"
    else:
        score, reason = judge(client, judge_model, result["question"],
                              meta["gold_answer"], answer)

    gold_ids = {normalize_id(x) for x in meta.get("source_records", [])}
    used_ids = cited_ids(result)
    hits = gold_ids & used_ids

    return {
        "id": result["id"],
        "task_type": task,
        "correctness": score,
        "judge_reason": reason,
        "abstained": abstained,
        "gold_ids": sorted(gold_ids),
        "cited_ids": sorted(used_ids),
        "citation_recall": len(hits) / len(gold_ids) if gold_ids else None,
        "citation_precision": len(hits) / len(used_ids) if used_ids else None,
        "total_tokens": (result.get("usage") or {}).get("total_tokens"),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    return "  n/a" if value is None else f"{value:5.2f}"


def summarize(graded):
    print(f"\n{'task_type':<14}{'n':>3}  {'correct':>7}  {'abstain':>7}  {'cite_rec':>8}  {'tokens':>7}")
    groups = sorted({g["task_type"] for g in graded}) + ["ALL"]
    for task in groups:
        rows = graded if task == "ALL" else [g for g in graded if g["task_type"] == task]
        tokens = mean([g["total_tokens"] for g in rows])
        print(f"{task:<14}{len(rows):>3}  "
              f"{fmt(mean([g['correctness'] for g in rows])):>7}  "
              f"{fmt(mean([float(g['abstained']) for g in rows])):>7}  "
              f"{fmt(mean([g['citation_recall'] for g in rows])):>8}  "
              f"{'n/a' if tokens is None else round(tokens):>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Results JSON file to grade")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())["results"]
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    graded = []
    for result in results:
        g = grade_one(result, client, args.judge_model)
        graded.append(g)
        print(f"{g['id']}: correctness={g['correctness']}  abstained={g['abstained']}  "
              f"cited={g['cited_ids']}  ({g['judge_reason']})")

    out_path = args.results.with_name(args.results.stem + "_graded.json")
    out_path.write_text(json.dumps({"source": str(args.results),
                                    "judge_model": args.judge_model,
                                    "graded": graded}, indent=2) + "\n")
    summarize(graded)
    print(f"\nSaved grades to {out_path}")


if __name__ == "__main__":
    main()