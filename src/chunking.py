"""Chunking strategies for the NHTSA recall corpus

Every strategy chunks the same per-campaign records, so only the chunking
differs. Strategies are listed in `STRATEGIES` at the bottom; `*_ctx` variants
prepend a metadata header to each chunk. Sizes are in characters.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Signature of an embedding function: list of texts -> (n, d) L2-normalized array
EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text

    `campaigns` lists the campaign numbers whose content is in the chunk, which
    is how retrieval is scored against the gold `source_records`.
    """

    id: str
    text: str
    campaigns: tuple[str, ...]


# Records
def build_records(df: pd.DataFrame) -> list[dict]:
    """Collapse the flat file (one row per vehicle per campaign) into one record per campaign"""
    records = []
    for campno, g in df.groupby("CAMPNO", sort=True):
        first = g.iloc[0]
        vehicle_strings = sorted(set(g["MAKETXT"] + " " + g["MODELTXT"] + " " + g["YEARTXT"]))
        records.append({
            "campaign": campno,
            "manufacturer": _clean(first["MFGNAME"]),
            "component": _clean(first["COMPNAME"]),
            "report_date": _format_date(first["RCDATE"]),
            "vehicles": sorted(set(zip(g["MAKETXT"], g["MODELTXT"], g["YEARTXT"]))),
            "defect": _clean(first["DESC_DEFECT"]),
            "consequence": _clean(first["CONEQUENCE_DEFECT"]),
            "remedy": _clean(first["CORRECTIVE_ACTION"]),
            # The original campaign document (as in main's src/rag.py) plus the
            # report date, which several evaluation questions ask about. Every
            # strategy carries the same content, so only the chunking differs.
            # The date goes last so fixed/recursive/semantic only change their
            # final chunk relative to the original document.
            "doc": (
                f"Recall campaign {first['CAMPNO']}.\n"
                f"Manufacturer: {first['MFGNAME']}.\n"
                f"Component: {first['COMPNAME']}.\n"
                f"Affected vehicles: {'; '.join(vehicle_strings)}.\n"
                f"Defect: {first['DESC_DEFECT']}\n"
                f"Consequence: {first['CONEQUENCE_DEFECT']}\n"
                f"Remedy: {first['CORRECTIVE_ACTION']}\n"
                f"Reported to NHTSA: {_format_date(first['RCDATE'])}."
            ),
        })
    return records


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _format_date(yyyymmdd: str) -> str:
    s = (yyyymmdd or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def compress_vehicles(vehicles) -> str:
    """("HONDA", "CR-V", "2017") ... ("HONDA", "CR-V", "2020") -> "HONDA CR-V 2017-2020" """
    years_by_model = defaultdict(set)
    for make, model, year in vehicles:
        years_by_model[(make.strip(), model.strip())].add(year.strip())
    return "; ".join(
        f"{make} {model} {_year_ranges(years)}".strip()
        for (make, model), years in sorted(years_by_model.items())
    )


def _year_ranges(years) -> str:
    nums = sorted(int(y) for y in years if y.isdigit() and y != "9999")
    other = sorted(y for y in years if not y.isdigit() or y == "9999")
    parts, start = [], None
    for i, y in enumerate(nums):
        if start is None:
            start = y
        if i == len(nums) - 1 or nums[i + 1] != y + 1:
            parts.append(str(start) if start == y else f"{start}-{y}")
            start = None
    parts += ["unknown year" if y in ("9999", "") else y for y in other]
    return ", ".join(parts)


def short_header(r: dict) -> str:
    return f"Recall {r['campaign']} | {r['manufacturer']} | {r['component']} | reported {r['report_date']}"


def context_header(r: dict, max_vehicle_chars: int = 400) -> str:
    """One-line context prepended to chunks so each is self-describing

    This is a deterministic, metadata-based version of "contextual retrieval":
    a chunk like "Dealers will replace the fuel pump" otherwise loses which
    recall and which vehicles it belongs to.
    """
    vehicles = compress_vehicles(r["vehicles"])
    if len(vehicles) > max_vehicle_chars:
        vehicles = vehicles[:max_vehicle_chars].rsplit("; ", 1)[0] + "; and others"
    return f"{short_header(r)} | Vehicles: {vehicles}"


def body_text(r: dict) -> str:
    return f"Defect: {r['defect']}\nConsequence: {r['consequence']}\nRemedy: {r['remedy']}"


# Splitters
def fixed_split(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size windows that step back `overlap` chars and avoid cutting words"""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return [c for c in chunks if c]


SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


def recursive_split(text: str, size: int = 500, overlap: int = 50,
                    separators: tuple[str, ...] = SEPARATORS) -> list[str]:
    """Split on the coarsest separator present, merging pieces up to `size`

    Pieces that are still too long are split again with the next separator,
    falling back to `fixed_split` when no separator is left (the same idea as
    LangChain's RecursiveCharacterTextSplitter, without the dependency).
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    seps = [s for s in separators if s in text]
    if not seps:
        return fixed_split(text, size, overlap)
    sep, rest = seps[0], tuple(separators[separators.index(seps[0]) + 1:])

    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

    chunks, current = [], ""
    for piece in pieces:
        if len(piece) > size:
            if current.strip():
                chunks.append(current.strip())
            chunks.extend(recursive_split(piece, size, overlap, rest))
            current = ""
            continue
        if len(current) + len(piece) > size and current.strip():
            chunks.append(current.strip())
            current = _tail(current, overlap)
            if len(current) + len(piece) > size:
                current = ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting at a word boundary"""
    if n <= 0 or len(text) <= n:
        return "" if n <= 0 else text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


# Strategies
def campaign_chunks(records: list[dict], **_) -> list[Chunk]:
    return [Chunk(r["campaign"], r["doc"], (r["campaign"],)) for r in records]


def fixed_chunks(records: list[dict], size: int = 500, overlap: int = 100, **_) -> list[Chunk]:
    return [
        Chunk(f"{r['campaign']}#{i}", text, (r["campaign"],))
        for r in records
        for i, text in enumerate(fixed_split(r["doc"], size, overlap))
    ]


def recursive_chunks(records: list[dict], size: int = 500, overlap: int = 50,
                     header: bool = False, **_) -> list[Chunk]:
    chunks = []
    for r in records:
        prefix = context_header(r) + "\n" if header else ""
        for i, text in enumerate(recursive_split(r["doc"], size, overlap)):
            chunks.append(Chunk(f"{r['campaign']}#{i}", prefix + text, (r["campaign"],)))
    return chunks


def semantic_chunks(records: list[dict], embed_fn: EmbedFn | None = None, header: bool = False,
                    percentile: float = 90, buffer: int = 1, min_chars: int = 150,
                    max_chars: int = 1000, **_) -> list[Chunk]:
    """Semantic chunking (Kamradt-style)

    1. Split each campaign doc into sentences (overlong ones are split further)
    2. Embed each sentence together with `buffer` neighbours on each side
    3. Start a new chunk where the cosine distance between neighbouring sentences
       exceeds the corpus-wide `percentile` of all such distances, provided the
       current chunk has at least `min_chars`; also cap chunks at `max_chars`

    The threshold is corpus-wide because a per-document percentile would force a
    split in every short campaign.
    """
    if embed_fn is None:
        raise ValueError("semantic chunking needs an embed_fn")

    units = [_semantic_units(r["doc"], max_chars) for r in records]
    windows = [
        " ".join(sents[max(0, i - buffer): i + buffer + 1])
        for sents in units
        for i in range(len(sents))
    ]
    vectors = embed_fn(windows)

    distances, pos = [], 0
    for sents in units:
        v = vectors[pos: pos + len(sents)]
        pos += len(sents)
        distances.append(1 - np.sum(v[:-1] * v[1:], axis=1))
    all_d = np.concatenate([d for d in distances if len(d)])
    threshold = float(np.percentile(all_d, percentile)) if len(all_d) else 1.0

    chunks = []
    for r, sents, d in zip(records, units, distances):
        groups, current = [], []
        for i, sent in enumerate(sents):
            size = sum(len(s) + 1 for s in current)
            if current and (
                (d[i - 1] > threshold and size >= min_chars) or size + len(sent) > max_chars
            ):
                groups.append(current)
                current = []
            current.append(sent)
        if current:
            groups.append(current)

        prefix = context_header(r) + "\n" if header else ""
        for i, g in enumerate(groups):
            chunks.append(Chunk(f"{r['campaign']}#{i}", prefix + " ".join(g), (r["campaign"],)))
    return chunks


def _semantic_units(doc: str, max_chars: int) -> list[str]:
    units = []
    for line in doc.split("\n"):
        for sent in split_sentences(line):
            units.extend(recursive_split(sent, max_chars, 0) if len(sent) > max_chars else [sent])
    return units or [doc]


def field_chunks(records: list[dict], max_chars: int = 1500, **_) -> list[Chunk]:
    """One chunk per logical field, each made self-describing with a header"""
    chunks = []
    for r in records:
        c = r["campaign"]
        overview = f"Affected vehicles: {compress_vehicles(r['vehicles'])}"
        for i, part in enumerate(recursive_split(overview, max_chars, 0)):
            chunks.append(Chunk(f"{c}#vehicles{i}", f"{short_header(r)}\n{part}", (c,)))
        header = context_header(r)
        for field in ("defect", "consequence", "remedy"):
            if r[field]:
                text = f"{header}\n{field.capitalize()}: {r[field]}"
                chunks.append(Chunk(f"{c}#{field}", text, (c,)))
    return chunks


def adaptive_chunks(records: list[dict], max_chars: int = 1500, **_) -> list[Chunk]:
    """Chunk size adapts to each campaign

    Most campaigns fit in `max_chars` and stay whole, so the defect and remedy
    stay together. Long ones (usually huge vehicle lists, which the embedding
    model would otherwise truncate) are split along their structure: vehicle
    overview chunks plus header-prefixed body chunks.
    """
    chunks = []
    for r in records:
        c = r["campaign"]
        vehicles = compress_vehicles(r["vehicles"])
        whole = f"{short_header(r)}\nAffected vehicles: {vehicles}\n{body_text(r)}"
        if len(whole) <= max_chars:
            chunks.append(Chunk(c, whole, (c,)))
            continue

        for i, part in enumerate(recursive_split(f"Affected vehicles: {vehicles}", max_chars, 0)):
            chunks.append(Chunk(f"{c}#vehicles{i}", f"{short_header(r)}\n{part}", (c,)))
        header = context_header(r)
        for i, part in enumerate(recursive_split(body_text(r), max_chars - len(header) - 1, 100)):
            chunks.append(Chunk(f"{c}#body{i}", f"{header}\n{part}", (c,)))
    return chunks


def vehicle_chunks(records: list[dict], max_chars: int = 1500, **_) -> list[Chunk]:
    """`adaptive` chunks plus one summary per make/model/year

    Summary chunks answer "which recalls affect vehicle X" in a single hit,
    which campaign-level chunks struggle with because each campaign is
    retrieved independently. The full campaign chunks are kept for details.
    """
    chunks = adaptive_chunks(records, max_chars=max_chars)

    by_vehicle = defaultdict(list)
    for r in records:
        for v in r["vehicles"]:
            by_vehicle[v].append(r)

    for (make, model, year), rs in sorted(by_vehicle.items()):
        rs = sorted(rs, key=lambda r: (r["report_date"], r["campaign"]))
        name = f"{year} {make} {model}".strip()
        title = f"Recall summary for the {name}: {len(rs)} recall campaign(s)."
        lines = [
            (r["campaign"], f"- {r['campaign']} (reported {r['report_date']}), "
                            f"{r['component']}: {_defect_summary(r['defect'])}")
            for r in rs
        ]
        groups, current, size = [], [], len(title)
        for item in lines:
            if current and size + len(item[1]) + 1 > max_chars:
                groups.append(current)
                current, size = [], len(title)
            current.append(item)
            size += len(item[1]) + 1
        groups.append(current)

        for i, g in enumerate(groups):
            text = title + "\n" + "\n".join(line for _, line in g)
            chunks.append(Chunk(f"vehicle:{name}#{i}", text, tuple(c for c, _ in g)))
    return chunks


def _defect_summary(defect: str, max_chars: int = 200) -> str:
    """Drop the boilerplate "X is recalling certain ... vehicles." opener"""
    sents = split_sentences(defect)
    if len(sents) > 1 and "recalling" in sents[0]:
        sents = sents[1:]
    s = sents[0] if sents else defect
    return s if len(s) <= max_chars else s[:max_chars].rsplit(" ", 1)[0] + "..."


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "campaign": campaign_chunks,
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "recursive_ctx": lambda records, **kw: recursive_chunks(records, header=True, **kw),
    "semantic": semantic_chunks,
    "semantic_ctx": lambda records, **kw: semantic_chunks(records, header=True, **kw),
    "field": field_chunks,
    "adaptive": adaptive_chunks,
    "vehicle": vehicle_chunks,
}


def make_chunks(strategy: str, records: list[dict], embed_fn: EmbedFn | None = None,
                **params) -> list[Chunk]:
    """Build the chunks for a named strategy"""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](records, embed_fn=embed_fn, **params)
