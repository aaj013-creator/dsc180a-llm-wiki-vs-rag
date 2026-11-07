"""Download Honda recalls and complaints from the public NHTSA API."""

import json
import os
import time

import requests

MAKE = "honda"
MODELS = ["civic", "accord", "cr-v"]
YEARS = range(2018, 2023)  # 2018 through 2022

RAW_DIR = os.path.join("data", "raw")

ENDPOINTS = {
    "recalls": "https://api.nhtsa.gov/recalls/recallsByVehicle",
    "complaints": "https://api.nhtsa.gov/complaints/complaintsByVehicle",
}


def fetch(source, model, year):
    """Call the NHTSA API for one source, model, and year."""
    params = {"make": MAKE, "model": model, "modelYear": year}
    response = requests.get(ENDPOINTS[source], params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for source in ENDPOINTS:
        for model in MODELS:
            for year in YEARS:
                filename = f"{source}_{model}_{year}.json"
                path = os.path.join(RAW_DIR, filename)

                try:
                    data = fetch(source, model, year)
                except requests.RequestException as error:
                    print(f"FAILED {filename}: {error}")
                    continue

                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

                count = len(data.get("results", []))
                print(f"Saved {filename} ({count} records)")

                time.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    main()