"""Read the project's trusted artifacts without changing or retraining them."""

from pathlib import Path
import json
import re
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SKLEARN_VERSION = "1.6.1"


@st.cache_resource
def load_model(filename):
    path = ROOT / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model is missing: models/{filename}")
    if sklearn.__version__ != SKLEARN_VERSION:
        raise RuntimeError(
            f"These models require scikit-learn {SKLEARN_VERSION}; found "
            f"{sklearn.__version__}. Install requirements.txt using the same Python "
            "that runs Streamlit. No internal compatibility patches are used."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            package = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load models/{filename}: {exc}. Use Python 3.11–3.13 "
            "and install requirements.txt in a fresh virtual environment."
        ) from exc
    required = {"model", "features"}
    if "model1" in filename:
        required |= {"threshold", "normal_rainfall_reference"}
    else:
        required |= {"imputer", "scaler", "cluster_names"}
    if not isinstance(package, dict) or not required.issubset(package):
        raise ValueError(f"Unexpected model package in {filename}; expected {sorted(required)}.")
    return package


@st.cache_data
def load_csv(filename):
    path = ROOT / "data" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is unavailable: data/{filename}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read data/{filename}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"data/{filename} contains no rows.")
    return frame


def load_districts():
    frame = load_csv("district_clusters.csv")
    required = {
        "District", "State", "Population", "Population_Exposure", "Cluster",
        "Vulnerability_Group", "Corrected_Percent_Flooded_Area", "DFSI",
        "Mean_Flood_Duration",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"District dataset is missing columns: {', '.join(sorted(missing))}")
    if frame[["District", "State"]].isna().any().any():
        raise ValueError("District dataset contains missing district or state names.")
    return frame


@st.cache_data
def load_historical_catalogue(states):
    from utils.rainfall import historical_catalogue

    return historical_catalogue(load_csv("rainfall_subdivision_monthly_1901_2015.csv"), states)


@st.cache_data
def notebook_evaluation():
    """Extract recorded outputs, never execute notebook cells or estimate scores."""
    path = ROOT / "notebooks" / "FloodRelief_AI.ipynb"
    if not path.is_file():
        return None
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        result = None
        default_report = None
        roc_auc = None
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell.get("source", []))
            output = "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))
            if 'print("Tuned Logistic Regression Results")' in source:
                metrics = {}
                for name in ["Accuracy", "Precision", "Recall", "F1", "F2"]:
                    match = re.search(rf"^{name}:\s*([0-9.eE+-]+)", output, re.MULTILINE)
                    if match:
                        metrics[name] = float(match.group(1))
                matrix = re.search(r"\[\[\s*(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\]\]", output)
                threshold = re.search(r"selected_threshold_lr\s*=\s*([0-9.]+)", source)
                if len(metrics) == 5 and matrix and threshold:
                    values = [int(x) for x in matrix.groups()]
                    result = {"metrics": metrics, "matrix": [values[:2], values[2:]],
                              "threshold": float(threshold.group(1)), "cell": index}
            if "classification_report(y_test, y_pred_lr)" in source:
                if "Classification Report:" in output:
                    default_report = output.split("Classification Report:", 1)[1].strip()
                match = re.search(r"^ROC-AUC:\s*([0-9.eE+-]+)", output, re.MULTILINE)
                if match:
                    roc_auc = float(match.group(1))
        if result:
            result["source"] = "notebooks/FloodRelief_AI.ipynb"
            result["default_report"] = default_report
            result["roc_auc"] = roc_auc
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def dataset_inventory():
    rows = []
    for path in sorted((ROOT / "data").glob("*.csv")):
        try:
            frame = load_csv(path.name)
            rows.append({"Dataset": path.name, "Rows": len(frame),
                         "Columns": len(frame.columns),
                         "Missing cells": int(frame.isna().sum().sum())})
        except (OSError, ValueError) as exc:
            rows.append({"Dataset": path.name, "Status": str(exc)})
    return pd.DataFrame(rows)
