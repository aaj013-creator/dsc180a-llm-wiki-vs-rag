"""RAG implementation with NHTSA post-2010 vehicle recall data

Requirements:
- Groq: put `GROQ_API_KEY=...` in the `.env` file
- Ollama: run `ollama serve` and `ollama pull nomic-embed-text`

Usage:
    python -m src.rag                          # default chunking (campaign)
    python -m src.rag --chunking vehicle       # any strategy in src/chunking.py

To compare chunking strategies on the question set, see src/evaluate_chunking.py.
"""

# Imports
import argparse
import csv
import hashlib
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

if __package__ in (None, ""):  # allow `python src/rag.py` as well as `python -m src.rag`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import STRATEGIES, build_records, make_chunks
from src.llm import GenerationSettings, GroqLLM

# Load GROQ_API_KEY from the .env file
load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "FLAT_RCL_POST_2010.txt"
# Content-addressed embedding cache shared by every chunking strategy.
# Delete the directory to force re-embedding.
EMBED_CACHE_DIR = ROOT / "data" / "processed" / "embeddings"

# Ollama endpoint
OLLAMA = "http://localhost:11434"

# Models
EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text expects task prefixes on both sides
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Characters of retrieved context sent to the LLM. A fixed budget (rather than a
# fixed k) keeps the comparison fair when strategies produce chunks of very
# different sizes. 6000 chars is about 5 campaign documents.
DEFAULT_CONTEXT_CHARS = 6000

COLUMNS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS",
    "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE",
]

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."
Cite the recall campaign numbers you used.

Context:
{context}

Question: {question}"""


# Embedding
def embed(texts, retries=3):
    """Embed a list of texts with Ollama, returns L2-normalized float32 vectors"""
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA}/api/embed",
                json={"model": EMBED_MODEL, "input": texts},
                timeout=600,
            )
            r.raise_for_status()
            v = np.asarray(r.json()["embeddings"], dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        except (requests.RequestException, KeyError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


class EmbeddingCache:
    """Content-addressed embedding cache: sha1(model + text) -> vector

    Shared across strategies, so text that appears in several strategies (or
    queries asked twice) is embedded once. New vectors are flushed to small part
    files as they are computed, so an interrupted run keeps its progress.
    """

    def __init__(self, directory=EMBED_CACHE_DIR, flush_every=4096):
        self.dir = Path(directory)
        self.flush_every = flush_every
        self.vectors = {}
        self._pending = {}
        if self.dir.is_dir():
            for part in sorted(self.dir.glob("part-*.npz")):
                with np.load(part) as z:
                    self.vectors.update(zip(z["keys"].tolist(), z["vectors"]))

    @staticmethod
    def key(text):
        return hashlib.sha1(f"{EMBED_MODEL}\x00{text}".encode()).hexdigest()

    def embed(self, texts, batch_size=64, label=None):
        """Return vectors for `texts`, embedding only those not cached yet"""
        keys = [self.key(t) for t in texts]
        missing = {}
        for k, t in zip(keys, texts):
            if k not in self.vectors and k not in missing:
                missing[k] = t

        if missing:
            todo = list(missing.items())
            started = time.time()
            for i in range(0, len(todo), batch_size):
                batch = todo[i:i + batch_size]
                for (k, _), v in zip(batch, embed([t for _, t in batch])):
                    self.vectors[k] = v
                    self._pending[k] = v
                if len(self._pending) >= self.flush_every:
                    self.flush()
                done = min(i + batch_size, len(todo))
                if label and (done == len(todo) or (i // batch_size) % 50 == 0):
                    rate = done / max(time.time() - started, 1e-6)
                    print(f"  [{label}] embedded {done}/{len(todo)} new texts ({rate:.0f}/s)",
                          flush=True)
            self.flush()

        return np.vstack([self.vectors[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)

    def flush(self):
        if not self._pending:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"part-{time.time_ns()}.npz"
        np.savez(path, keys=np.array(list(self._pending)),
                 vectors=np.vstack(list(self._pending.values())))
        self._pending = {}


# LLM
# Groq's free tier allows 8000 tokens per minute per model; stay a little under
TOKENS_PER_MINUTE = 7000

_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = GroqLLM(settings=GenerationSettings())
    return _LLM


class TokenThrottle:
    """Sliding one-minute token window, so calls wait instead of hitting 429s"""

    def __init__(self, tokens_per_minute=TOKENS_PER_MINUTE):
        self.limit = tokens_per_minute
        self.log = deque()  # (timestamp, tokens)

    def wait(self, expected_tokens):
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] >= 60:
                self.log.popleft()
            used = sum(tokens for _, tokens in self.log)
            if not self.log or used + expected_tokens <= self.limit:
                return
            time.sleep(self.log[0][0] + 60 - now + 0.5)

    def record(self, tokens):
        self.log.append((time.time(), tokens))


_THROTTLES = defaultdict(TokenThrottle)


def ask_llm(prompt, llm=None, retries=6):
    """Send a prompt to a chat model, pacing and retrying API rate limits; returns an LLMResult

    `llm` is any object with `generate(messages)` and `settings` (GroqLLM by
    default, or OllamaLLM for local runs, which is not paced).
    """
    llm = llm or get_llm()
    paced = getattr(llm, "rate_limited", True)
    throttle = _THROTTLES[llm.settings.model]
    expected = len(prompt) // 4 + 1000  # prompt plus a typical completion
    for attempt in range(retries):
        if paced:
            throttle.wait(expected)
        result = llm.generate([{"role": "user", "content": prompt}])
        if paced:
            throttle.record(result.usage.total_tokens or expected)
        if result.ok or not result.error.retryable or attempt == retries - 1:
            return result
        if "per day" in result.error.message:  # daily quota: retrying soon won't help
            return result
        time.sleep(min(60, 4 * 2 ** attempt))


# Data loading
def load_data():
    """Load the raw recall data into a DataFrame"""
    return pd.read_csv(
        DATA_PATH,
        sep="\t",  # Data is stored as .txt but is separated by tabs
        header=None,
        names=COLUMNS,
        dtype=str,
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )


# Indexing
@dataclass
class Index:
    strategy: str
    chunks: list
    vectors: np.ndarray  # (n_chunks, dim), L2-normalized

    def search(self, query, cache, top_n=50):
        """Return [(chunk, score)] for the top_n chunks by cosine similarity"""
        q = cache.embed([QUERY_PREFIX + query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:top_n]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_index(strategy, records, cache, **params):
    """Chunk the records with `strategy` and embed every chunk"""
    def embed_docs(texts):  # used by semantic chunking for sentence embeddings
        return cache.embed([DOC_PREFIX + t for t in texts], label=f"{strategy} sentences")

    chunks = make_chunks(strategy, records, embed_fn=embed_docs, **params)
    vectors = cache.embed([DOC_PREFIX + c.text for c in chunks], label=strategy)
    return Index(strategy, chunks, vectors)


# Retrieval & generation
def select_context(results, max_chars=DEFAULT_CONTEXT_CHARS):
    """Take ranked chunks in order until the character budget is spent (at least one)"""
    selected, used, seen = [], 0, set()
    for chunk, _ in results:
        if chunk.text in seen:
            continue
        if selected and used + len(chunk.text) > max_chars:
            break
        selected.append(chunk)
        seen.add(chunk.text)
        used += len(chunk.text) + 2
    return selected


def ask_llm_rag(question, index, cache, max_chars=DEFAULT_CONTEXT_CHARS):
    """Answer a question using retrieved chunks as context; returns (LLMResult, chunks)"""
    chunks = select_context(index.search(question, cache), max_chars)
    context = "\n\n".join(c.text for c in chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    return ask_llm(prompt), chunks


# Main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Chat with the recall data using RAG")
    parser.add_argument("--chunking", default="campaign", choices=sorted(STRATEGIES))
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    args = parser.parse_args(argv)

    records = build_records(load_data())
    cache = EmbeddingCache()
    index = build_index(args.chunking, records, cache)
    print(f"Indexed {len(index.chunks)} chunks with '{args.chunking}' chunking")

    def answer(question):
        result, _ = ask_llm_rag(question, index, cache, args.context_chars)
        print(result.text if result.ok else f"Error: {result.error.message}")

    # Example question
    question = "What is the recall for the 2015 Ford F-150?"
    print(f"Question: {question}")
    answer(question)

    # Chat loop (q to quit)
    while True:
        question = input("\nAsk a question (q to quit): ")

        if question.lower() == "q":
            break

        answer(question)


if __name__ == "__main__":
    main()
