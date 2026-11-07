"""Print recall records in a readable format for writing evaluation questions."""

import json
import sys

path = sys.argv[1]

with open(path) as f:
    data = json.load(f)

for record in data.get("results", []):
    print("=" * 70)
    print("Campaign: ", record.get("NHTSACampaignNumber"))
    print("Vehicle:  ", record.get("ModelYear"), record.get("Make"), record.get("Model"))
    print("Date:     ", record.get("ReportReceivedDate"))
    print("Component:", record.get("Component"))
    print("Summary:  ", record.get("Summary"))
    print("Remedy:   ", record.get("Remedy"))