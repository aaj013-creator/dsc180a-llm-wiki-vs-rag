"""Offline checks for the aggregate answer generator's counting rules."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from aggregate_answers import COLUMNS, count_campaigns, load_recalls, main, top_components


class AggregateAnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "recalls.txt"
        self.rows = [
            self.record("23V001000", "CIVIC", "2019", "AIR BAGS", "20230101"),
            self.record("23V001000", "CIVIC", "2019", "AIR BAGS", "20230101"),
            self.record("23V001000", "CIVIC", "2020", "BRAKES", "20230101"),
            self.record("23V002000", "CR-V", "2019", "AIR BAGS", "20230201"),
            self.record("23V003000", "ACCORD", "2019", "BRAKES", "20230301"),
            self.record("22V004000", "CR-V", "2023", "BRAKES", "20221201"),
            self.record("23E005000", "EQUIPMENT", "9999", "AIR BAGS", "20230401", "E"),
        ]
        self.write_rows()

    @staticmethod
    def record(campaign, model, model_year, component, date, recall_type="V"):
        return {
            "CAMPNO": campaign, "MAKETXT": " Honda ", "MODELTXT": model,
            "YEARTXT": model_year, "COMPNAME": component, "RCDATE": date,
            "RCLTYPECD": recall_type, "RCL_CMPT_ID": "0000123", "MFGCAMPNO": "N/A",
        }

    def write_rows(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for record in self.rows:
                writer.writerow([record.get(column, "") for column in COLUMNS])

    def test_loader_preserves_identifiers_literals_and_vehicle_rows(self):
        df = load_recalls(self.path)
        self.assertEqual(len(df), 7)
        self.assertEqual(df.iloc[0]["RCL_CMPT_ID"], "0000123")
        self.assertEqual(df.iloc[0]["MFGCAMPNO"], "N/A")
        self.assertEqual(df.iloc[0]["MAKETXT"], "Honda")
        self.assertEqual(df.iloc[0]["report_date"], pd.Timestamp("2023-01-01"))

    def test_campaign_counts_ignore_repeated_rows_and_keep_vehicle_membership(self):
        df = load_recalls(self.path)
        result = count_campaigns(df, make=" honda ", model="civic", model_year=2019)
        self.assertEqual(result["campaign_count"], 1)
        self.assertEqual(result["source_records"], ["23V001000"])
        self.assertEqual(count_campaigns(df, model="CIVIC", model_year=2020)["campaign_count"], 1)
        self.assertEqual(count_campaigns(pd.concat([df, df]))["campaign_count"], 4)

    def test_report_year_and_model_year_are_separate_filters(self):
        df = load_recalls(self.path)
        self.assertEqual(count_campaigns(df, report_year=2023)["campaign_count"], 3)
        self.assertEqual(count_campaigns(df, model_year=2023)["source_records"], ["22V004000"])
        self.assertEqual(count_campaigns(df, model_year=2023, report_year=2023)["campaign_count"], 0)

    def test_component_ranking_preserves_ties_and_all_components(self):
        df = load_recalls(self.path)
        before = df.copy(deep=True)
        expected = [
            {"component": "AIR BAGS", "campaign_count": 2, "source_records": ["23V001000", "23V002000"]},
            {"component": "BRAKES", "campaign_count": 2, "source_records": ["23V001000", "23V003000"]},
        ]
        self.assertEqual(top_components(df, 2023), expected)
        self.assertEqual(top_components(pd.concat([df, df]), 2023), expected)
        count_campaigns(df, make="HONDA")
        pd.testing.assert_frame_equal(df, before)

    def test_equipment_is_only_included_when_requested(self):
        df = load_recalls(self.path)
        self.assertEqual(count_campaigns(df, report_year=2023, recall_type=None)["campaign_count"], 4)
        winner = top_components(df, 2023, recall_type=None)
        self.assertEqual(len(winner), 1)
        self.assertEqual(winner[0]["component"], "AIR BAGS")
        self.assertEqual(winner[0]["campaign_count"], 3)

    def test_empty_matches_have_zero_count_and_no_winners(self):
        df = load_recalls(self.path)
        result = count_campaigns(df, model="NONEXISTENT")
        self.assertEqual(result["campaign_count"], 0)
        self.assertEqual(result["source_records"], [])
        self.assertEqual(top_components(df, 2030), [])

    def test_invalid_or_ambiguous_source_data_is_rejected(self):
        for field, value, message in (
            ("CAMPNO", "", "Missing required values"),
            ("COMPNAME", "   ", "Missing required values"),
            ("RCDATE", "20230230", "Invalid RCDATE"),
            ("RCDATE", "2023011", "Invalid RCDATE"),
            ("RCDATE", "20240101", "Conflicting report dates"),
        ):
            with self.subTest(field=field, value=value):
                old = self.rows[0][field]
                self.rows[0][field] = value
                self.write_rows()
                with self.assertRaisesRegex(ValueError, message):
                    load_recalls(self.path)
                self.rows[0][field] = old

    def test_wrong_column_count_is_rejected(self):
        self.path.write_text("\t".join(["value"] * 28) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected 29 columns, found 28"):
            load_recalls(self.path)

    def test_runner_writes_all_six_answers_and_dataset_fingerprint(self):
        output = Path(self.temp.name) / "results" / "answers.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data", str(self.path), "--output", str(output)]), 0)
        payload = json.loads(output.read_text(encoding="utf-8"))
        answers = {item["id"]: item for item in payload["answers"]}
        self.assertEqual(set(answers), {"q25", "q30", "q31", "q32", "q33", "top_component_2023"})
        self.assertEqual(answers["q32"]["campaign_count"], 3)
        self.assertEqual(answers["q33"]["winners"], ["ACCORD", "CIVIC", "CR-V"])
        self.assertEqual(answers["q33"]["source_records"], ["23V001000", "23V002000", "23V003000"])
        self.assertEqual(len(answers["top_component_2023"]["winners"]), 2)
        self.assertEqual(payload["run"]["data_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
