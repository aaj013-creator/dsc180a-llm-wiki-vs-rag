"""Print model answers next to gold answers for a results file."""

import json
import sys

path = sys.argv[1]
results = json.load(open(path))["results"]

for r in results:
    meta = r["metadata"]
    answer = (r.get("answer") or "").strip()
    print("=" * 70)
    print(f"{r['id']}  [{meta['task_type']}]  status: {r['status']}")
    print("Q:     ", r["question"])
    print("GOLD:  ", meta["gold_answer"])
    print("MODEL: ", answer[:400] + ("..." if len(answer) > 400 else ""))