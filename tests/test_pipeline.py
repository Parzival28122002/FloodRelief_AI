import unittest
import math

import joblib
import numpy as np
import pandas as pd

from utils.model_loader import ROOT, notebook_evaluation
from utils.prediction import (
    build_flood_input, estimate_relief_resources, predict_cluster, predict_flood, run_pipeline,
)
from utils.rainfall import historical_scenarios, historical_catalogue


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flood = joblib.load(ROOT / "models/floodrelief_model1_complete.pkl")
        cls.cluster = joblib.load(ROOT / "models/floodrelief_model2_complete.pkl")
        cls.districts = pd.read_csv(ROOT / "data/district_clusters.csv")
        cls.district = cls.districts.loc[cls.districts.District == "Anantapur"].iloc[0]
        cls.rainfall = pd.read_csv(ROOT / "data/rainfall_subdivision_monthly_1901_2015.csv")

    def rainfall_value(self, year, month):
        source = self.rainfall
        rows = source[(source.YEAR == year) & source.SUBDIVISION.isin(["COASTAL ANDHRA PRADESH", "RAYALSEEMA"])]
        return float(rows[month].mean())

    def test_recorded_notebook_example_and_real_inference(self):
        # Notebook cells 46/51 record AP Jan 2006 at score 0.0785.
        current = self.rainfall_value(2006, "JAN")
        lag1 = self.rainfall_value(2005, "DEC")
        lag2 = self.rainfall_value(2005, "NOV")
        frame = build_flood_input(self.flood, "Andhra Pradesh", 1, current, lag1, lag2)
        self.assertEqual(list(frame.columns), self.flood["features"])
        result = predict_flood(self.flood, frame)
        self.assertAlmostEqual(result["score"], 0.0785, delta=0.00005)
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["threshold"], 0.45)
        self.assertEqual(frame.Is_Monsoon.dtype.kind, "i")

    def test_flood_and_no_flood_pipeline_paths(self):
        # Both scenarios use actual rainfall records, not synthetic observations.
        for month, current, lag1, lag2, expected in [
            (1, self.rainfall_value(2006, "JAN"), self.rainfall_value(2005, "DEC"), self.rainfall_value(2005, "NOV"), 0),
            (8, self.rainfall_value(2006, "AUG"), self.rainfall_value(2006, "JUL"), self.rainfall_value(2006, "JUN"), 1),
        ]:
            with self.subTest(month=month):
                result = run_pipeline(self.flood, self.cluster, self.district, "Andhra Pradesh", month, current, lag1, lag2)
                self.assertEqual(result["flood"]["label"], expected)
                self.assertEqual(result["relief"]["People_Planned_For"], int(self.district.Population_Exposure))
                self.assertEqual(result["cluster"]["cluster"], int(self.district.Cluster))
                self.assertGreater(result["relief"]["Water_Litres"], 0)

    def test_all_saved_cluster_assignments_and_missing_values(self):
        features = self.cluster["features"]
        X = self.cluster["scaler"].transform(self.cluster["imputer"].transform(self.districts[features]))
        labels = self.cluster["model"].predict(X)
        np.testing.assert_array_equal(labels, self.districts.Cluster.to_numpy())
        missing = self.districts[self.districts[features].isna().any(axis=1)]
        self.assertGreater(len(missing), 0)
        for _, row in missing.iterrows():
            self.assertEqual(predict_cluster(self.cluster, row)["cluster"], int(row.Cluster))

    def test_all_saved_relief_rows(self):
        source = pd.read_csv(ROOT / "data/district_relief_requirements.csv")
        for _, row in source.iterrows():
            result = estimate_relief_resources(row.Population_Exposure, int(row.Relief_Days))
            for key, value in result.items():
                if key in source.columns:
                    self.assertEqual(value, row[key], (row.District, key))
            self.assertLessEqual(result["Drinking_Water_Litres"], result["Water_Litres"])

    def test_coverage_rounding_and_duration(self):
        result = estimate_relief_resources(self.district.Population_Exposure, 10, 0.25)
        people = math.ceil(float(self.district.Population_Exposure) * 0.25)
        self.assertEqual(result["People_Planned_For"], people)
        self.assertEqual(result["Water_Litres"], people * 150)
        self.assertEqual(result["Total_Food_Kcal"], people * 21000)
        self.assertEqual(result["Emergency_Medical_Kits"], math.ceil(people / 100))

    def test_invalid_inputs_are_rejected(self):
        for kwargs in [{"exposed_population": -1}, {"exposed_population": np.nan},
                       {"exposed_population": np.inf}, {"exposed_population": 1, "relief_days": 0},
                       {"exposed_population": 1, "relief_days": 1.5},
                       {"exposed_population": 1, "assistance_coverage": 1.1},
                       {"exposed_population": 1, "people_per_medical_kit": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_relief_resources(**kwargs)
        for state, month, rain in [("Unsupported", 1, 0), ("Andhra Pradesh", 13, 0),
                                   ("Andhra Pradesh", 1, np.nan), ("Andhra Pradesh", 1, -1)]:
            with self.assertRaises(ValueError):
                build_flood_input(self.flood, state, month, rain, 0, 0)
        with self.assertRaises(ValueError):
            run_pipeline(self.flood, self.cluster, self.district, "Bihar", 1, 0, 0, 0)

    def test_recorded_evaluation_provenance(self):
        result = notebook_evaluation()
        self.assertEqual(result["threshold"], self.flood["threshold"])
        self.assertEqual(result["matrix"], [[1613, 302], [137, 477]])
        self.assertAlmostEqual(result["metrics"]["Accuracy"], (1613 + 477) / 2529)
        self.assertIn("2529", result["default_report"])

    def test_historical_rainfall_calendar_lags_and_actual_values(self):
        scenarios = historical_scenarios(self.rainfall, "Andhra Pradesh")
        january = scenarios.loc[(scenarios.YEAR == 2006) & (scenarios.Month == 1)].iloc[0]
        self.assertAlmostEqual(january.Rainfall, self.rainfall_value(2006, "JAN"))
        self.assertAlmostEqual(january.Rainfall_Lag1, self.rainfall_value(2005, "DEC"))
        self.assertAlmostEqual(january.Rainfall_Lag2, self.rainfall_value(2005, "NOV"))
        frame = build_flood_input(self.flood, "Andhra Pradesh", 1, january.Rainfall, january.Rainfall_Lag1, january.Rainfall_Lag2)
        self.assertAlmostEqual(predict_flood(self.flood, frame)["score"], 0.0785, delta=0.00005)
        # Removing source measurements must remove affected triplets, not fill them.
        source = self.rainfall.copy()
        source.loc[(source.YEAR == 2006) & source.SUBDIVISION.isin(["COASTAL ANDHRA PRADESH", "RAYALSEEMA"]), "FEB"] = np.nan
        incomplete = historical_scenarios(source, "Andhra Pradesh")
        self.assertFalse(((incomplete.YEAR == 2006) & incomplete.Month.isin([2, 3, 4])).any())
        # A missing year must not borrow lags from two years earlier.
        without_year = historical_scenarios(self.rainfall.loc[self.rainfall.YEAR != 2005], "Andhra Pradesh")
        self.assertFalse(((without_year.YEAR == 2006) & without_year.Month.isin([1, 2])).any())

    def test_historical_state_formation_and_missing_source(self):
        scenarios = historical_scenarios(self.rainfall, "Telangana")
        self.assertGreater(len(scenarios), 0)
        self.assertGreaterEqual(scenarios.Date.min(), pd.Timestamp("2014-09-01"))
        self.assertTrue(historical_scenarios(self.rainfall.iloc[:0], "Andhra Pradesh").empty)
        with self.assertRaises(ValueError):
            historical_scenarios(self.rainfall, "Unsupported")

    def test_classifier_without_probability_support(self):
        class LabelOnlyClassifier:
            classes_ = [0, 1]

            def predict(self, frame):
                return [0]

        result = predict_flood({"model": LabelOnlyClassifier()}, pd.DataFrame())
        self.assertEqual(result, {"label": 0, "score": None, "threshold": None})

    def test_historical_batch_matches_single_state_predictions(self):
        from utils.prediction import predict_historical_states

        supported = tuple(self.flood["normal_rainfall_reference"].State.unique())
        catalogue, _ = historical_catalogue(self.rainfall, supported)
        for year, month in [(2006, 1), (2015, 8)]:
            results, skipped = predict_historical_states(self.flood, catalogue, year, month)
            self.assertEqual(len(results) + len(skipped), len(supported))
            self.assertEqual(len(results), results.State.nunique())
            for _, row in results.iterrows():
                inputs = build_flood_input(self.flood, row.State, month, row.Rainfall, row.Rainfall_Lag1, row.Rainfall_Lag2)
                actual = predict_flood(self.flood, inputs)
                self.assertEqual(row.Flood, actual["label"])
                self.assertAlmostEqual(row["Model score (uncalibrated)"], actual["score"])
            self.assertIn("Jammu and Kashmir", results.State.tolist())


if __name__ == "__main__":
    unittest.main()
