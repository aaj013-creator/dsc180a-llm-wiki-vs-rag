"""Convert data/processed/questions.csv into the JSON format used by src/baseline.py."""

import csv
import json
from pathlib import Path

SRC = Path("data/processed/questions.csv")
DST = Path("data/evaluation/questions.json")


def main():
    questions = []
    with SRC.open(newline="") as f:
        for row in csv.DictReader(f):
            sources = [s.strip() for s in row["source_records"].split(";") if s.strip()]
            questions.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "task_type": row["task_type"],
                    "gold_answer": row["gold_answer"],
                    "source_records": sources,
                }
            )

    payload = {
        "dataset_name": "nhtsa_recall_eval",
        "version": "0.1",
        "questions": questions,
    }

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(questions)} questions to {DST}")


if __name__ == "__main__":
    main()