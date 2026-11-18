"""Validated inference and transparent, deterministic relief planning."""

import math
import numpy as np
import pandas as pd

FLOOD_FEATURES = [
    "State", "Rainfall", "Rainfall_Lag1", "Rainfall_Lag2", "Rainfall_3Month",
    "Rainfall_Anomaly", "Month_Sin", "Month_Cos", "Is_Monsoon",
]
CLUSTER_FEATURES = ["Corrected_Percent_Flooded_Area", "DFSI", "Mean_Flood_Duration"]


def finite_number(value, name, minimum=0):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return number


def build_flood_input(package, state, month, rainfall, lag1, lag2):
    if list(package["features"]) != FLOOD_FEATURES:
        raise ValueError("The flood model feature contract has changed; review its training notebook.")
    month = finite_number(month, "Month", 1)
    if month != int(month) or month > 12:
        raise ValueError("Month must be an integer from 1 to 12.")
    reference = package["normal_rainfall_reference"]
    normal = reference.loc[(reference["State"] == state) & (reference["Month"] == month), "Normal_Rainfall"]
    if len(normal) != 1:
        raise ValueError("No unique historical rainfall reference exists for this state and month.")
    normal = finite_number(normal.iloc[0], "Historical rainfall reference")
    rainfall = finite_number(rainfall, "Current rainfall")
    lag1 = finite_number(lag1, "Previous month's rainfall")
    lag2 = finite_number(lag2, "Rainfall two months earlier")
    frame = pd.DataFrame([{
        "State": str(state), "Rainfall": rainfall, "Rainfall_Lag1": lag1,
        "Rainfall_Lag2": lag2, "Rainfall_3Month": rainfall + lag1 + lag2,
        "Rainfall_Anomaly": rainfall - normal,
        "Month_Sin": float(np.sin(2 * np.pi * month / 12)),
        "Month_Cos": float(np.cos(2 * np.pi * month / 12)),
        "Is_Monsoon": int(month in (6, 7, 8, 9)),
    }], columns=package["features"])
    trained = list(package["model"].feature_names_in_)
    if trained != list(frame.columns):
        raise ValueError("Input column order does not match the fitted flood pipeline.")
    return frame


def predict_flood(package, frame):
    classes = list(package["model"].classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Expected flood labels 0 (no recorded flood) and 1 (recorded flood).")
    if not callable(getattr(package["model"], "predict_proba", None)):
        label = package["model"].predict(frame)[0]
        if label not in (0, 1):
            raise ValueError("The classifier returned an invalid flood label.")
        return {"label": int(label), "score": None, "threshold": None}
    score = float(package["model"].predict_proba(frame)[0, classes.index(1)])
    threshold = finite_number(package["threshold"], "Flood threshold")
    if threshold > 1 or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The model returned an invalid risk score or threshold.")
    return {"label": int(score >= threshold), "score": score, "threshold": threshold}


def predict_cluster(package, district):
    if list(package["features"]) != CLUSTER_FEATURES:
        raise ValueError("The clustering feature contract has changed.")
    values = {}
    for name in package["features"]:
        value = district[name]
        values[name] = np.nan if pd.isna(value) else finite_number(value, name)
    if not pd.isna(values["Corrected_Percent_Flooded_Area"]) and values["Corrected_Percent_Flooded_Area"] > 100:
        raise ValueError("Corrected flooded area must be between 0 and 100 percent.")
    frame = pd.DataFrame([values], columns=package["features"])
    transformed = package["scaler"].transform(package["imputer"].transform(frame))
    cluster = int(package["model"].predict(transformed)[0])
    if cluster not in package["cluster_names"]:
        raise ValueError("Unknown cluster returned by the saved model.")
    return {"cluster": cluster, "group": package["cluster_names"][cluster],
            "imputed": [key for key, value in values.items() if pd.isna(value)]}


def predict_historical_states(package, catalogue, year, month):
    """Run the unchanged classifier on each valid state-month observation."""
    rows, skipped = [], []
    supported = sorted(set(package["normal_rainfall_reference"]["State"]))
    for state in supported:
        observation = catalogue.loc[(catalogue.State == state) & (catalogue.YEAR == year) & (catalogue.Month == month)]
        if len(observation) != 1:
            skipped.append({"State": state, "Reason": "No unique complete rainfall triplet for the selected year/month"})
            continue
        row = observation.iloc[0]
        frame = build_flood_input(package, state, month, row.Rainfall, row.Rainfall_Lag1, row.Rainfall_Lag2)
        prediction = predict_flood(package, frame)
        result = {"State": state, "Flood": prediction["label"],
                  "Prediction": "Flood" if prediction["label"] else "No Flood",
                  "Rainfall": float(row.Rainfall), "Rainfall_Lag1": float(row.Rainfall_Lag1),
                  "Rainfall_Lag2": float(row.Rainfall_Lag2)}
        if prediction["score"] is not None:
            result["Model score (uncalibrated)"] = prediction["score"]
        rows.append(result)
    return pd.DataFrame(rows), skipped


def estimate_relief_resources(exposed_population, relief_days=7, assistance_coverage=1.0,
                             water_per_person=15, kcal_per_person=2100,
                             meals_per_person=3, people_per_medical_kit=100,
                             drinking_per_person=3):
    population = finite_number(exposed_population, "Planning population")
    days = finite_number(relief_days, "Relief days", 1)
    if days != int(days):
        raise ValueError("Relief days must be a whole number.")
    coverage = finite_number(assistance_coverage, "Assistance coverage")
    if coverage > 1:
        raise ValueError("Assistance coverage must be between 0 and 1.")
    water = finite_number(water_per_person, "Water allowance", 0.01)
    drinking = finite_number(drinking_per_person, "Drinking water allocation", 0.01)
    if drinking > water:
        raise ValueError("Drinking water allocation cannot exceed total basic water.")
    kcal = finite_number(kcal_per_person, "Food energy allowance", 1)
    meals = finite_number(meals_per_person, "Meal equivalents", 0.01)
    kit_capacity = finite_number(people_per_medical_kit, "People per medical kit", 1)
    people = math.ceil(population * coverage)
    return {"People_Planned_For": people, "Relief_Days": int(days),
            "Water_Litres": people * water * int(days),
            "Drinking_Water_Litres": people * drinking * int(days),
            "Total_Food_Kcal": people * kcal * int(days),
            "Total_Meals": people * meals * int(days),
            "Emergency_Medical_Kits": math.ceil(people / kit_capacity)}


def run_pipeline(flood_package, cluster_package, district, state, month, rainfall,
                 lag1, lag2, relief_days=7, assistance_coverage=1.0):
    """Combine inference with an independent district exposure planning scenario.

    There is no population regressor in this repository. Exposure is a historical
    planning proxy and is never presented as a model-predicted affected count.
    """
    if str(district["State"]) != state:
        raise ValueError("District state must match the flood model state.")
    population = finite_number(district["Population"], "District population", 1)
    exposure = finite_number(district["Population_Exposure"], "District population exposure")
    if exposure > population:
        raise ValueError("District population exposure exceeds district population.")
    frame = build_flood_input(flood_package, state, month, rainfall, lag1, lag2)
    flood = predict_flood(flood_package, frame)
    cluster = predict_cluster(cluster_package, district) if cluster_package else None
    # Relief planning remains available for either classifier label.
    relief = estimate_relief_resources(exposure, relief_days, assistance_coverage)
    return {"flood": flood, "cluster": cluster, "relief": relief,
            "historical_exposure": exposure, "input": frame}
