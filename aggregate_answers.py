"""Calculate reference answers directly from the NHTSA recall data with pandas.

Run from the repository root:
    python aggregate_answers.py
    python aggregate_answers.py --component-year 2024 --output results/aggregates_2024.json

The default run answers q25 and q30-q33, plus a component-ranking question.
All queries count distinct campaigns and default to vehicle recalls (type V).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COLUMNS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS",
    "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE",
]


def load_recalls(path: str | Path) -> pd.DataFrame:
    """Load the headerless, 29-column TSV and add a validated report_date.

    Preserve identifiers and model years as strings. Trim whitespace and treat
    empty fields as missing. Retain all rows: repeated campaigns can describe
    different vehicles or components and must not be deduplicated globally.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype="string",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[""],
    )
    if len(df.columns) != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {len(df.columns)}")
    df.columns = COLUMNS

    for column in COLUMNS:
        df[column] = df[column].str.strip().replace("", pd.NA)

    required = ["CAMPNO", "COMPNAME", "RCLTYPECD", "RCDATE"]
    missing = df[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing required values: {missing[missing.gt(0)].to_dict()}")

    dates = pd.to_datetime(df["RCDATE"], format="%Y%m%d", errors="coerce")
    invalid = ~df["RCDATE"].str.fullmatch(r"\d{8}") | dates.isna()
    if invalid.any():
        examples = df.loc[invalid, "RCDATE"].head(3).tolist()
        raise ValueError(f"Invalid RCDATE in {int(invalid.sum())} rows: {examples}")
    df["report_date"] = dates

    dates_per_campaign = df.groupby("CAMPNO")["report_date"].nunique()
    conflicts = dates_per_campaign[dates_per_campaign.gt(1)]
    if not conflicts.empty:
        raise ValueError(f"Conflicting report dates for campaigns: {conflicts.index[:3].tolist()}")
    return df


def count_campaigns(
    df: pd.DataFrame,
    *,
    make: str | None = None,
    model: str | None = None,
    model_year: int | None = None,
    report_year: int | None = None,
    recall_type: str | None = "V",
) -> dict:
    """Count distinct CAMPNO values after applying all filters to the same rows.

    Make/model/type matching is exact, ignoring case and surrounding whitespace.
    model_year filters YEARTXT; report_year filters the parsed RCDATE. Passing
    recall_type=None includes every recall type. Return the contributing IDs
    alongside the count so the answer can be checked against the source data.
    """
    selected = df
    filters = {
        "make": make,
        "model": model,
        "model_year": model_year,
        "report_year": report_year,
        "recall_type": recall_type,
    }
    for column, value in (
        ("MAKETXT", make), ("MODELTXT", model), ("RCLTYPECD", recall_type)
    ):
        if value is not None:
            if not value.strip():
                raise ValueError(f"{column} filter must not be empty")
            selected = selected.loc[
                selected[column].str.casefold().eq(value.strip().casefold())
            ]
    if model_year is not None:
        selected = selected.loc[selected["YEARTXT"].eq(str(model_year))]
    if report_year is not None:
        selected = selected.loc[selected["report_date"].dt.year.eq(report_year)]

    campaign_ids = sorted(selected["CAMPNO"].unique().tolist())
    return {
        "campaign_count": len(campaign_ids),
        "source_records": campaign_ids,
        "filters": filters,
    }


def top_components(
    df: pd.DataFrame, year: int, *, recall_type: str | None = "V"
) -> list[dict]:
    """Return every component tied for the most campaigns reported in year.

    Use the full COMPNAME label, not the broader category before a colon.
    Count each campaign once per component. A campaign with several components
    contributes to each one, so component counts are not additive. Return an
    empty list when no rows match. Each winner includes its supporting IDs.
    """
    selected = df.loc[df["report_date"].dt.year.eq(year)]
    if recall_type is not None:
        if not recall_type.strip():
            raise ValueError("recall_type filter must not be empty")
        selected = selected.loc[
            selected["RCLTYPECD"].str.casefold().eq(recall_type.strip().casefold())
        ]
    if selected.empty:
        return []

    # Deduplicate within each component, preserving the evidence for its count.
    sources = selected.groupby("COMPNAME")["CAMPNO"].agg(
        lambda ids: sorted(set(ids))
    )
    counts = sources.map(len)
    return [
        {
            "component": component,
            "campaign_count": int(counts[component]),
            "source_records": sources[component],
        }
        for component in counts[counts.eq(counts.max())].index
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the fixed aggregate questions and save answers plus provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/FLAT_RCL_POST_2010.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/aggregate_answers.json")
    parser.add_argument(
        "--component-year", type=int, default=2023,
        help="report year for the additional component question (default: 2023)",
    )
    args = parser.parse_args(argv)
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must be different from the input data file")

    try:
        df = load_recalls(args.data)
        vehicle_results = {
            model: count_campaigns(df, make="HONDA", model=model, model_year=2019)
            for model in ("CR-V", "CIVIC", "ACCORD")
        }
        answers = [
            {
                "id": question_id,
                "question": f"How many recall campaigns cover the 2019 Honda {model}?",
                "answer": f"{vehicle_results[model]['campaign_count']} campaigns.",
                **vehicle_results[model],
            }
            for question_id, model in (("q25", "CR-V"), ("q30", "CIVIC"), ("q31", "ACCORD"))
        ]

        honda_2023 = count_campaigns(df, make="HONDA", report_year=2023)
        answers.append({
            "id": "q32",
            "question": "How many recall campaigns covering Honda-brand vehicles were reported to NHTSA in 2023?",
            "answer": f"{honda_2023['campaign_count']} campaigns.",
            **honda_2023,
        })

        counts = {model: result["campaign_count"] for model, result in vehicle_results.items()}
        largest_count = max(counts.values())
        winners = sorted(model for model, count in counts.items() if count == largest_count)
        answers.append({
            "id": "q33",
            "question": "Among the 2019 Honda Civic, 2019 Honda Accord, and 2019 Honda CR-V, which has the most recall campaigns?",
            "answer": f"{', '.join(winners)}: {largest_count} campaigns; counts by model: {counts}.",
            "winners": winners,
            "counts_by_model": counts,
            "source_records": sorted({
                campaign_id for result in vehicle_results.values()
                for campaign_id in result["source_records"]
            }),
            "vehicle_results": vehicle_results,
        })

        components = top_components(df, args.component_year)
        answers.append({
            "id": f"top_component_{args.component_year}",
            "question": f"Which full component label had the most vehicle recall campaigns reported in {args.component_year}?",
            "answer": "; ".join(
                f"{item['component']}: {item['campaign_count']} campaigns"
                for item in components
            ) or "No matching campaigns.",
            "winners": components,
            "source_records": sorted({
                campaign_id for item in components for campaign_id in item["source_records"]
            }),
            "filters": {"report_year": args.component_year, "recall_type": "V"},
        })

        digest = hashlib.sha256()
        with args.data.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        output = {
            "run": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_file": str(args.data.resolve()),
                "data_sha256": digest.hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pandas_version": pd.__version__,
                "loaded_rows": len(df),
                "loaded_campaigns": int(df["CAMPNO"].nunique()),
                "count_unit": "distinct CAMPNO",
                "recall_type": "V",
                "component_definition": "full COMPNAME label after trimming whitespace",
            },
            "answers": answers,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Aggregate calculation failed: {exc}\n")

    for result in answers:
        print(f"{result['id']}: {result['answer']}")
    print(f"Saved answers and campaign IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
