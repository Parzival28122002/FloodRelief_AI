# FloodRelief AI

FloodRelief AI is a Streamlit application for exploring rainfall-based flood scenarios in India and estimating relief supplies for a selected district. It was developed as an M.Tech AI/ML semester project.

## Objective

Flood preparedness involves decisions at different scales. A rainfall scenario can help assess whether a flood is likely to be recorded in a state during a given month. Historical district records provide local context, and a planning scenario turns population exposure into supply quantities. This project connects those three steps so users can see the inputs and assumptions behind each result.

**Workflow:** Rainfall scenario → state-month flood classification → district impact profile → relief supply estimates

## Approach

### 1. Classify flood occurrence from rainfall

The [flood prediction notebook](notebooks/FloodRelief_AI.ipynb) joins monthly rainfall with the flood inventory at state-month resolution. A month receives a positive label when the inventory records at least one flood event for that state. Unmatched months receive a zero label, which makes the training data sensitive to gaps in reporting.

A class-weighted **logistic regression** pipeline uses the state, current monthly rainfall, rainfall from the previous two calendar months, the three-month total, and the anomaly from the saved 1970–2000 state-month rainfall normal. It also uses cyclical month features and a June–September monsoon indicator. The pipeline standardizes rainfall features and one-hot encodes the state. A model score of **0.45 or higher** produces a Flood label. Scores are uncalibrated.

The app can run historical scenarios for supported states with complete rainfall inputs from **1970–2015**, or use three monthly rainfall values supplied manually for a single state. Historical rainfall is averaged across available mapped subdivisions.

### 2. Explore historical district impact

The [district clustering notebook](notebooks/FloodRelief_model2.ipynb) applies median imputation and standard scaling to corrected flooded-area percentage, DFSI, and mean flood duration. **K-means** groups districts into three historical impact profiles: **Low Flood Impact**, **High Flood Impact**, and **Long-Duration Flood Impact**.

The dashboard displays the saved assignments from [district_clusters.csv](data/district_clusters.csv). Users can search districts, filter by cluster, and choose the source record when a district has conflicting entries.

### 3. Estimate relief supplies

The [relief estimation notebook](notebooks/Relief_estimation_module.ipynb) supplies the planning formulas. The app uses the selected district’s historical `Population_Exposure` as a planning proxy. The user sets the share of that population to assist and the number of relief days. Planning is available for either flood classification.

Let **P** be historical exposure multiplied by assistance coverage and rounded up. Let **D** be the relief duration in days.

| Resource | Calculation |
| --- | --- |
| Basic water for drinking, cooking, and hygiene | P × D × 15 litres |
| Drinking water, included in basic water | P × D × 3 litres |
| Food energy | P × D × 2,100 kcal |
| Meal equivalents | P × D × 3 |
| Emergency medical kits | P ÷ 100, rounded up |

The 15-litre basic water and 2,100 kcal food baselines are drawn from the [2018 Sphere Handbook](https://spherestandards.org/wp-content/uploads/Sphere-Handbook-2018-EN.pdf). The meal and kit figures are project assumptions. Kit contents and replenishment are unspecified, and the kit calculation is independent of duration.

## Recorded results

The flood prediction notebook reports the following results for **2,529 state-month records from 2006–2015** at the saved threshold of **0.45**.

| Accuracy | Precision | Recall | F1 | F2 |
| --- | --- | --- | --- | --- |
| 82.64% | 61.23% | 77.69% | 0.685 | 0.737 |

The notebook used a 2001–2005 validation window for threshold selection. These figures come from saved notebook outputs. The district clustering artifact reports a **silhouette score of 0.442**, which measures separation among the fitted groups.

## Run locally

Use Python **3.11–3.13**. The project was previously verified with Python **3.13.2**. Run these commands from the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The app loads the bundled CSVs and model artifacts locally and requires no API keys or database. The pinned **scikit-learn 1.6.1** matches the saved artifacts.

In the sidebar, start with **Flood Prediction**, choose a state in **District Analysis**, then adjust assistance coverage and duration in **Relief Planning**. **Overview** contains the saved metrics and methodology notes.

To run the automated checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Repository guide

| Path | Contents |
| --- | --- |
| [app.py](app.py) | Streamlit entry point, overview, and navigation |
| [utils/](utils/) | Model and data loading, rainfall preprocessing, inference, and dashboard views |
| [data/](data/) | Flood inventory, rainfall observations, district impact data, and saved relief scenarios |
| [models/](models/) | Saved logistic regression and K-means artifacts |
| [notebooks/](notebooks/) | Model development, recorded evaluation, and relief calculations |
| [figures/](figures/) | Plots from model evaluation and district clustering |
| [tests/](tests/) | Model, calculation, and Streamlit workflow checks |

## Limitations and future work

- The classifier uses current-month rainfall and operates at state-month resolution. The app is designed for scenario exploration. A state result cannot establish whether a particular district will flood.
- District profiles describe historical impact. Historical exposure can support supply planning, while estimating the population affected by a specific event would require a separate, validated model. The repository contains no population regression artifact.
- The district data contain historical names, missing values, and conflicting records. The upstream exposure-data joins and the definition of DFSI need further documentation.
- Future work includes reconciling district identifiers, evaluating on an untouched temporal or geographic holdout, validating an event-specific population model, and grounding medical supply estimates in locally assessed needs.
