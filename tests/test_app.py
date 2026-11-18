import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.model_loader import ROOT


class ApplicationTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    def goto(self, app, page):
        app.button(key=f"nav_{page}").click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def predict_custom(self, app, state="Andhra Pradesh"):
        self.goto(app, "Flood Prediction")
        app.selectbox(key="_rainfall_mode").set_value("Custom Rainfall").run()
        app.selectbox(key="_custom_state").set_value(state).run()
        app.button(key="run_custom").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)

    def test_overview_and_all_pages(self):
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(list(metrics), ["Accuracy", "Precision", "Recall", "F1-score", "F2-score"])
        self.assertEqual(metrics["F2-score"], "73.72%")
        self.assertEqual(len(app.sidebar.radio), 0)
        self.assertFalse(any("confusion" in str(m.value).lower() for m in app.markdown))
        for page in ["Flood Prediction", "District Analysis", "Relief Planning", "Overview"]:
            self.goto(app, page)

    def test_custom_selection_relief_and_navigation_persistence(self):
        app = self.app()
        self.predict_custom(app)
        self.assertEqual(app.session_state["selected_state"], "Andhra Pradesh")
        self.assertEqual(app.session_state["prediction_results"]["results"].Prediction.iloc[0], "No Flood")
        self.assertFalse(any(s.label == "Historical year" for s in app.selectbox))
        app.number_input(key="_custom_rainfall").set_value(0.1).run()
        app.button(key="to_districts").click().run()
        self.assertEqual(app.selectbox(key="_analysis_state").value, "Andhra Pradesh")
        frame = pd.read_csv(ROOT / "data/district_clusters.csv")
        districts = frame.loc[frame.State == "Andhra Pradesh"]
        table = app.dataframe[0].value
        self.assertEqual(table.District.tolist(), districts.District.tolist())
        self.assertEqual(table["Historical cluster label"].tolist(), districts.Vulnerability_Group.tolist())
        app.selectbox(key="_district_name").set_value("Anantapur").run()
        app.button(key="to_relief").click().run()
        self.assertEqual(len(app.exception), 0)
        values = {m.label: m.value for m in app.metric}
        exposure = int(districts.loc[districts.District == "Anantapur", "Population_Exposure"].iloc[0])
        self.assertEqual(values["Planning population (people)"], f"{exposure:,}")
        self.assertEqual(values["💧 Drinking Water · litres"], f"{exposure * 7 * 3:,}")
        self.assertEqual(values["🚰 Basic Water · litres"], f"{exposure * 7 * 15:,}")
        self.assertEqual(values["🍽️ Food Energy · kcal"], f"{exposure * 7 * 2100:,}")
        self.assertEqual(values["🥣 Meal Equivalents · meals"], f"{exposure * 7 * 3:,}")
        self.assertEqual(values["🧰 Emergency Medical Kits · kits"], f"{(exposure + 99) // 100:,}")
        app.number_input(key="_assistance_coverage").set_value(50.0).run()
        app.number_input(key="_relief_days").set_value(10).run()
        chosen_row = app.session_state["selected_district_row"]
        self.goto(app, "Overview")
        self.goto(app, "Relief Planning")
        self.assertEqual(app.number_input(key="_relief_days").value, 10)
        self.assertEqual(app.number_input(key="_assistance_coverage").value, 50.0)
        self.assertEqual(app.session_state["selected_district_row"], chosen_row)
        self.goto(app, "Flood Prediction")
        self.assertEqual(app.selectbox(key="_custom_state").value, "Andhra Pradesh")
        self.assertEqual(app.number_input(key="_custom_rainfall").value, 0.1)
        self.assertIsNotNone(app.session_state["prediction_results"])

    def test_historical_all_states_and_carryover(self):
        app = self.app()
        self.goto(app, "Flood Prediction")
        app.selectbox(key="_historical_year").set_value(2006).run()
        app.selectbox(key="_historical_month").set_value(8).run()
        app.button(key="run_historical").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        results = app.session_state["prediction_results"]["results"]
        self.assertGreater(len(results), 15)
        self.assertGreater(int(results.Flood.sum()), 0)
        self.assertEqual(results.Flood.iloc[0], 1)
        app.selectbox(key="_result_state").set_value("Bihar").run()
        app.button(key="to_districts").click().run()
        self.assertEqual(app.selectbox(key="_analysis_state").value, "Bihar")
        self.goto(app, "Flood Prediction")
        self.assertEqual(app.selectbox(key="_historical_year").value, 2006)
        self.assertEqual(app.selectbox(key="_historical_month").value, 8)
        self.assertEqual(app.selectbox(key="_result_state").value, "Bihar")
        self.assertEqual(len(app.session_state["prediction_results"]["results"]), len(results))

    def test_no_district_match_and_state_change_resets_selection(self):
        app = self.app()
        self.predict_custom(app, "Bihar")
        self.goto(app, "District Analysis")
        bihar_row = app.session_state["selected_district_row"]
        app.selectbox(key="_analysis_state").set_value("Maharashtra").run()
        self.assertNotEqual(app.session_state["selected_district_row"], bihar_row)
        app.selectbox(key="_district_name").set_value("Aurangabad").run()
        self.assertEqual(len(app.selectbox(key="_district_record").options), 2)
        indices = pd.read_csv(ROOT / "data/district_clusters.csv").query("State == 'Maharashtra' and District == 'Aurangabad'").index.tolist()
        app.selectbox(key="_district_record").set_value(indices[1]).run()
        self.goto(app, "Relief Planning")
        self.assertEqual(app.session_state["selected_district_row"], indices[1])
        self.predict_custom(app, "Jammu and Kashmir")
        self.assertNotIn("selected_district_row", app.session_state)
        self.goto(app, "District Analysis")
        self.assertTrue(any("No matching district records" in item.value for item in app.info))
        self.assertNotIn("selected_district_row", app.session_state)
        self.goto(app, "Relief Planning")
        self.assertEqual(len(app.metric), 0)

    def test_filters_do_not_restrict_district_selection(self):
        app = self.app()
        self.predict_custom(app)
        self.goto(app, "District Analysis")
        options = app.selectbox(key="_district_name").options
        app.selectbox(key="_cluster_filter_Andhra Pradesh").set_value("High Flood Impact").run()
        table = app.dataframe[0].value
        self.assertTrue((table["Historical cluster label"] == "High Flood Impact").all())
        self.assertEqual(app.selectbox(key="_district_name").options, options)
        app.text_input(key="search_Andhra Pradesh").set_value("not a district").run()
        self.assertEqual(len(app.dataframe[0].value), 0)
        self.assertEqual(app.selectbox(key="_district_name").options, options)

    def test_missing_assets_do_not_disable_unrelated_sections(self):
        with patch("utils.dashboard.load_model", side_effect=FileNotFoundError("Test: model missing")):
            app = self.app()
            self.assertEqual(len(app.exception), 0)
            self.assertGreater(len(app.warning), 0)
            self.goto(app, "District Analysis")
            self.assertGreater(len(app.metric), 0)
        with patch("utils.dashboard.load_districts", side_effect=FileNotFoundError("Test: districts missing")):
            app = self.app()
            self.predict_custom(app)
            self.goto(app, "District Analysis")
            self.assertGreater(len(app.warning), 0)
        with patch("utils.dashboard.load_historical_catalogue", side_effect=FileNotFoundError("Test: rainfall missing")):
            app = self.app()
            self.goto(app, "Flood Prediction")
            self.assertGreater(len(app.info), 0)
            self.predict_custom(app)


if __name__ == "__main__":
    unittest.main()
