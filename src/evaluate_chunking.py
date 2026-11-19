"""Compare chunking strategies on the evaluation question set

    python run.py eval-chunking                                 # retrieval metrics only
    python run.py eval-chunking --generate --generator ollama   # + answers graded by the judge
    python run.py eval-chunking --regrade results/<file>.json   # grade answers the judge skipped

Retrieval is scored against each question's gold campaigns. Answers are graded
with the same rules and judge (gpt-oss-120b on Groq) as scripts/grade.py.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.chunking import STRATEGIES, build_records
from src.llm import GenerationSettings, GroqLLM, OllamaLLM
from src.rag import (
    DEFAULT_CONTEXT_CHARS,
    RAG_PROMPT_TEMPLATE,
    ROOT,
    EmbeddingCache,
    ask_llm,
    build_index,
    get_llm,
    load_data,
    select_context,
)

QUESTIONS_PATH = ROOT / "data" / "processed" / "questions.csv"
RESULTS_DIR = ROOT / "results"
KS = (1, 3, 5, 10)

# Judge model, prompt, and abstention pattern match scripts/grade.py on main
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT_TEMPLATE = """You are grading an answer to a question about vehicle safety recalls.

Question: {question}
Reference answer: {gold}
Model answer: {answer}

Score the model answer against the reference answer:
- 1 if it contains the key facts of the reference answer and nothing that contradicts it
- 0.5 if it is partly correct or is missing important facts
- 0 if it is wrong, contradicts the reference, or does not answer the question

Respond with only JSON: {{"score": <0, 0.5, or 1>, "reason": "<one sentence>"}}"""

ABSTAIN_PATTERN = re.compile(
    r"(don['’]t know|do not know|not sure|cannot (?:find|determine|answer)"
    r"|no information|not (?:in|found in) the (?:context|corpus|records))",
    re.IGNORECASE,
)
CAMPAIGN_PATTERN = re.compile(r"\b(\d{2})\s?([VETC])\s?-?\s?(\d{3})(?:000)?\b", re.I)

_JUDGE_LLM = None


def get_judge_llm():
    global _JUDGE_LLM
    if _JUDGE_LLM is None:
        _JUDGE_LLM = GroqLLM(settings=GenerationSettings(model=JUDGE_MODEL, reasoning_effort="medium"))
    return _JUDGE_LLM


def load_questions(path=QUESTIONS_PATH):
    """Load questions from the CSV (data/processed) or JSON (data/evaluation) format"""
    path = Path(path)
    if path.suffix == ".json":
        items = json.loads(path.read_text())["questions"]
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        items = [
            {**row, "source_records": row["source_records"].split(";")}
            for row in df.to_dict(orient="records")
        ]
    return [
        {
            "id": item["id"],
            "task_type": item["task_type"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "gold_campaigns": [c.strip() for c in item["source_records"]
                               if c.strip() and c.strip() != "---"],
        }
        for item in items
    ]


def retrieval_metrics(ranked_campaigns, context_campaigns, gold):
    """hit@k, recall@k, mrr, and ctx_recall for one question with gold campaigns

    ranked_campaigns: the campaigns in each retrieved chunk, in rank order
    context_campaigns: campaigns in the context actually sent to the LLM.
    ctx_recall is the fairest retrieval number, since every strategy gets the
    same character budget.
    """
    gold = set(gold)
    out = {}
    for k in KS:
        found = set().union(*ranked_campaigns[:k]) & gold
        out[f"hit@{k}"] = float(bool(found))
        out[f"recall@{k}"] = len(found) / len(gold)
    first = next((i for i, cs in enumerate(ranked_campaigns, 1) if gold & set(cs)), None)
    out["mrr"] = 1 / first if first else 0.0
    out["ctx_recall"] = len(set(context_campaigns) & gold) / len(gold)
    return out


def cited_campaigns(text):
    """Normalize campaign numbers in an answer: '21V-215' and '21V215000' -> '21V215000'"""
    return sorted({f"{a}{b.upper()}{c}000" for a, b, c in CAMPAIGN_PATTERN.findall(text or "")})


MAX_CONSECUTIVE_FAILURES = 3
UNGRADED_REASON = "not graded: judge unavailable (run --regrade later)"
JUDGE_STATE = {"consecutive_failures": 0, "disabled_reason": None}


def grade(question, answer):
    """Score one answer with the same rules as scripts/grade.py; returns (score, reason, abstained)"""
    abstained = not answer.strip() or bool(ABSTAIN_PATTERN.search(answer))
    if question["task_type"] == "unanswerable":
        return (1.0, "Correctly abstained", True) if abstained else \
               (0.0, "Answered an unanswerable question", False)
    if abstained:
        return 0.0, "Abstained on an answerable question", True

    if JUDGE_STATE["disabled_reason"]:
        return None, UNGRADED_REASON, False
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question["question"], gold=question["gold_answer"], answer=answer)
    result = ask_llm(prompt, llm=get_judge_llm())
    if not result.ok:
        JUDGE_STATE["consecutive_failures"] += 1
        if JUDGE_STATE["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            # Stop calling the judge but keep generating answers; grade them
            # later with --regrade once the Groq quota frees up
            JUDGE_STATE["disabled_reason"] = result.error.message
            print(f"  Judge unavailable, continuing without grading: {result.error.message[:200]}",
                  flush=True)
        return None, f"judge error: {result.error.message}", False
    JUDGE_STATE["consecutive_failures"] = 0
    match = re.search(r"\{.*\}", result.text, re.S)
    try:
        data = json.loads(match.group(0))
        return float(data["score"]), data.get("reason", ""), False
    except (AttributeError, ValueError, KeyError, TypeError):
        return None, f"Could not parse judge output: {result.text[:200]}", False


def evaluate_strategy(strategy, records, questions, cache, context_chars, generate,
                      generator_llm=None):
    t0 = time.time()
    index = build_index(strategy, records, cache)
    build_seconds = time.time() - t0

    rows, failures = [], []
    for q in questions:
        results = index.search(q["question"], cache, top_n=100)
        context = select_context(results, context_chars)
        context_campaigns = sorted({c for chunk in context for c in chunk.campaigns})
        row = {
            "strategy": strategy,
            "id": q["id"],
            "task_type": q["task_type"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "gold_campaigns": q["gold_campaigns"],
            "top_chunks": [
                {"id": c.id, "campaigns": list(c.campaigns[:20]), "score": round(s, 4)}
                for c, s in results[:10]
            ],
            "context_chunk_ids": [c.id for c in context],
            "context_campaigns": context_campaigns[:50],
            "ctx_chars": sum(len(c.text) for c in context),
        }
        if q["gold_campaigns"]:
            row.update(retrieval_metrics([c.campaigns for c, _ in results],
                                         context_campaigns, q["gold_campaigns"]))

        if generate:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context="\n\n".join(c.text for c in context), question=q["question"])
            result = ask_llm(prompt, llm=generator_llm)
            row["prompt_tok"] = result.usage.prompt_tokens
            if not result.ok:
                # An API failure is missing data, not a wrong answer: leave the
                # scores empty so they drop out of the averages
                row.update({"answer": None, "error": result.error.message})
                failures.append(result.error.message)
                if len(failures) >= MAX_CONSECUTIVE_FAILURES:
                    raise GenerationAborted(failures[-1], info_for(index, build_seconds), rows)
            else:
                failures.clear()
                answer = result.text
                score, reason, abstained = grade(q, answer)
                cited = cited_campaigns(answer)
                row.update({
                    "answer": answer,
                    "error": None,
                    "correct": score,
                    "judge_reason": reason,
                    "abstain": float(abstained),
                    "cited": cited,
                })
                if q["gold_campaigns"]:
                    row["cite_recall"] = len(set(cited) & set(q["gold_campaigns"])) / len(q["gold_campaigns"])
        rows.append(row)

    return info_for(index, build_seconds), rows


def info_for(index, build_seconds):
    lengths = np.array([len(c.text) for c in index.chunks])
    return {
        "strategy": index.strategy,
        "n_chunks": len(index.chunks),
        "chunk_chars_mean": float(lengths.mean()),
        "chunk_chars_p50": float(np.median(lengths)),
        "chunk_chars_max": int(lengths.max()),
        "build_seconds": round(build_seconds, 1),
    }


class GenerationAborted(Exception):
    """The answer generator kept failing (e.g. a Groq rate limit, or Ollama not running)"""

    def __init__(self, message, info, rows):
        super().__init__(message)
        self.info, self.rows = info, rows

# correct: judge score (1, 0.5, 0); abstain: share of "I don't know" answers;
# cite_recall: share of gold campaigns cited in the answer
METRIC_COLUMNS = (
    [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]
    + ["mrr", "ctx_recall", "ctx_chars", "correct", "abstain", "cite_recall", "prompt_tok"]
)


def summarize(rows, infos):
    df = pd.DataFrame(rows)
    metrics = [m for m in METRIC_COLUMNS if m in df.columns]
    overall = df.groupby("strategy", sort=False)[metrics].mean().assign(task_type="all")
    by_task = df.groupby(["strategy", "task_type"], sort=False)[metrics].mean().reset_index("task_type")
    summary = pd.concat([overall, by_task]).reset_index()
    info = pd.DataFrame(infos)[["strategy", "n_chunks", "chunk_chars_mean"]]
    return summary.merge(info, on="strategy")


def print_summary(summary, generate):
    show = ["strategy", "n_chunks", "chunk_chars_mean", "hit@5", "recall@5", "recall@10",
            "mrr", "ctx_recall", "ctx_chars"]
    if generate:
        show += ["correct", "abstain", "cite_recall", "prompt_tok"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for task in ["all"] + [t for t in summary.task_type.unique() if t != "all"]:
        part = summary[summary.task_type == task][[c for c in show if c in summary.columns]]
        part = part.dropna(axis=1, how="all")
        if part.empty:
            continue
        print(f"\n== {task} ==")
        print(part.to_string(index=False, float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.0f}"))


def write_results(output, run, infos, rows):
    """Write the JSON results and CSV summary; returns the summary DataFrame"""
    summary = summarize(rows, infos)
    run = {**run, "strategies": infos, "ungraded": count_ungraded(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "run": run,
        "summary": summary.to_dict(orient="records"),
        "results": rows,
    }, indent=2, default=str) + "\n")
    summary.to_csv(output.with_suffix(".csv"), index=False)
    return summary


def count_ungraded(rows):
    """Answers that exist but have no score yet (the judge failed or was unavailable)"""
    return sum(1 for r in rows if r.get("answer") and r.get("correct") is None)


def regrade(path):
    """Grade answers in an existing results file that are missing a score"""
    data = json.loads(path.read_text())
    rows = data["results"]
    todo = [r for r in rows if r.get("answer") and r.get("correct") is None]
    print(f"{len(todo)} ungraded answers in {path}", flush=True)
    for i, row in enumerate(todo, 1):
        score, reason, abstained = grade(row, row["answer"])
        row.update({"correct": score, "judge_reason": reason, "abstain": float(abstained)})
        if JUDGE_STATE["disabled_reason"]:
            break
        if i % 25 == 0:
            print(f"  graded {i}/{len(todo)}", flush=True)

    summary = write_results(path, data["run"], data["run"]["strategies"], rows)
    print_summary(summary, generate=True)
    remaining = count_ungraded(rows)
    print(f"\nUpdated {path}. {remaining} answers still ungraded"
          + (" (run --regrade again later)." if remaining else "."))
    return remaining


def default_output_path(generate, n_questions, strategies, context_chars, results_dir=RESULTS_DIR):
    """Name results by run type, e.g. chunking_eval_answers_39q.json

    Mode is "answers" (with --generate) or "retrieval". A strategy subset and a
    non-default context budget are added to the name. An existing file is never
    overwritten: a _2, _3, ... suffix is added instead.
    """
    parts = ["chunking_eval", "answers" if generate else "retrieval", f"{n_questions}q"]
    if list(strategies) != list(STRATEGIES):
        parts.append("-".join(strategies))
    if context_chars != DEFAULT_CONTEXT_CHARS:
        parts.append(f"ctx{context_chars}")
    stem = "_".join(parts)
    path = results_dir / f"{stem}.json"
    n = 2
    while path.exists() or path.with_suffix(".csv").exists():
        path = results_dir / f"{stem}_{n}.json"
        n += 1
    return path


def make_generator(backend, ollama_model):
    """None means the default Groq client from src/rag.py"""
    if backend == "ollama":
        return OllamaLLM(settings=GenerationSettings(model=ollama_model))
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare chunking strategies for RAG")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument("--generate", action="store_true",
                        help="also generate answers and grade them with the LLM judge (Groq)")
    parser.add_argument("--generator", choices=["groq", "ollama"], default="groq",
                        help="where answers are generated: Groq API (default) or local Ollama")
    parser.add_argument("--ollama-model", default="gpt-oss:20b",
                        help="Ollama model used with --generator ollama")
    parser.add_argument("--regrade", type=Path, metavar="RESULTS_JSON",
                        help="only grade the ungraded answers in an existing results file")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (default: results/chunking_eval_<retrieval|answers>_<N>q.json)")
    args = parser.parse_args(argv)

    if args.regrade:
        raise SystemExit(1 if regrade(args.regrade) else 0)

    questions = load_questions(args.questions)
    print(f"Loading data and building records...", flush=True)
    records = build_records(load_data())
    cache = EmbeddingCache()
    print(f"{len(records)} campaigns, {len(questions)} questions, {len(cache.vectors)} cached embeddings")

    generator_llm = make_generator(args.generator, args.ollama_model)
    generator_settings = (generator_llm or get_llm()).settings.to_dict()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or default_output_path(
        args.generate, len(questions), args.strategies, args.context_chars)
    run = {
        "started_at_utc": stamp,
        "questions_file": str(args.questions),
        "context_chars": args.context_chars,
        "generate": args.generate,
        "generator": {"backend": args.generator, "settings": generator_settings} if args.generate else None,
        "judge_model": JUDGE_MODEL if args.generate else None,
        "complete": False,
        "aborted_reason": None,
    }

    infos, rows = [], []
    for strategy in args.strategies:
        print(f"\n--- {strategy} ---", flush=True)
        try:
            info, strategy_rows = evaluate_strategy(
                strategy, records, questions, cache, args.context_chars, args.generate,
                generator_llm=generator_llm)
        except GenerationAborted as exc:
            run["aborted_reason"] = f"{args.generator} kept failing during '{strategy}': {exc}"
            infos.append(exc.info)
            rows.extend(exc.rows)
            break
        infos.append(info)
        rows.extend(strategy_rows)
        quick = pd.DataFrame(strategy_rows)
        quick = {m: quick[m].mean() for m in ("recall@5", "mrr", "ctx_recall", "correct") if m in quick}
        errors = sum(1 for r in strategy_rows if r.get("error"))
        ungraded = count_ungraded(strategy_rows)
        print(f"  {info['n_chunks']} chunks, mean {info['chunk_chars_mean']:.0f} chars | "
              + ", ".join(f"{m}={v:.3f}" for m, v in quick.items())
              + (f" | {errors} generation errors (excluded)" if errors else "")
              + (f" | {ungraded} ungraded" if ungraded else ""), flush=True)
        write_results(output, run, infos, rows)  # keep finished strategies if interrupted

    run["complete"] = run["aborted_reason"] is None
    summary = write_results(output, run, infos, rows)
    print_summary(summary, args.generate)
    print(f"\nSaved {output} and {output.with_suffix('.csv').name}")

    ungraded = count_ungraded(rows)
    if ungraded:
        print(f"\n{ungraded} answers are ungraded because the judge was unavailable"
              f" ({JUDGE_STATE['disabled_reason'] or 'judge errors'}).\n"
              f"Grade them later with: python run.py eval-chunking --regrade {output}")
    if run["aborted_reason"]:
        print(f"\nStopped early. {run['aborted_reason']}\n"
              f"Re-run the remaining strategies later with --strategies.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
