# FloodRelief AI

**Intelligent Flood Impact and Relief Resource Planning System** — a local Streamlit application for an M.Tech AI/ML semester project.

## Run locally (Windows PowerShell)

Use **Python 3.11–3.13**. The application was verified locally with Python **3.13.2**. The original training Python version is not recorded in the supplied notebooks.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. Use the virtual environment's Python for both installation and launch; activating it is optional. No API keys, database, or external service are required. The application itself makes no external API calls.

The pinned **scikit-learn 1.6.1** matches both artifacts, which load without version warnings in the verified environment. A different version can produce `AttributeError: Can't get attribute '_RemainderColsList'`. Install the requirements in a fresh environment rather than patching scikit-learn internals. The loader rejects mismatched scikit-learn versions with an actionable message. Only load trusted Joblib/pickle artifacts: deserialization can execute code.

## What is implemented

- **Overview:** project description, actual algorithm names, five saved classifier metrics (Accuracy, Precision, Recall, F1 and F2), and workflow. Evaluation provenance, dataset information, preprocessing, limitations and project explanations remain in optional expanders. No confusion matrix is displayed.
- **Flood Prediction:** Historical Rainfall runs the existing classifier for all supported states with valid observations for a selected year/month. Custom Rainfall accepts a state, month and three rainfall values, without requiring a year. Scores are shown only when supported and labelled uncalibrated.
- **District Analysis:** carries over the prediction state, displays all matching source districts with their original cluster labels, supports name search and cluster filtering, and allows explicit selection of conflicting source records. No cluster assignments are recalculated.
- **Relief Planning:** automatically uses the selected district, with editable duration and assistance coverage. Five resource cards show drinking/basic water, food energy, meal equivalents and emergency medical kits, with small icons and units. Calculations are available after either Flood or No Flood results.

Use the compact icon buttons in the left sidebar. Streamlit session state preserves completed predictions, relevant inputs, the selected state and district, relief duration, and coverage across navigation. Widget state is copied to separate durable keys so navigating away does not erase selections. Changing state invalidates the previous district. An unmatched state is reported explicitly; the app never silently selects a different state. District table filters do not limit the complete district dropdown.

**There is no population regression artifact in this repository.** The second model is K-means, not a regressor. The application explicitly shows the regression estimate and regression evaluation as unavailable. It uses the selected district's existing `Population_Exposure` as a labelled historical planning proxy. Neither predictions nor regression scores are fabricated.

## Files

```text
app.py                         Overview, technical explanations and navigation
utils/dashboard.py             Connected prediction, district and relief views
utils/model_loader.py          Cached models/CSVs; recorded notebook evaluation
utils/prediction.py             Validated inference and relief formulas
.streamlit/config.toml         Navy/slate dark theme with cyan accents
utils/rainfall.py              Historical monthly scenarios and calendar lags
models/                        Original saved artifacts (preserved)
data/                          Original local CSV files (preserved)
notebooks/                     Original training/relief notebooks (preserved)
tests/test_pipeline.py          Model, calculation, and interface checks
tests/test_app.py               Headless Streamlit page and workflow checks
requirements.txt               Pinned direct dependencies
```

All asset paths resolve relative to the project directory, independent of the launch directory. Model and data loads are cached. Restart Streamlit or clear its cache after replacing artifacts or datasets. Missing models and datasets produce messages; optional files do not disable unrelated pages. No model is trained or retrained by the application or tests.

## Verified artifact contracts

### Flood classifier

`models/floodrelief_model1_complete.pkl` contains a dictionary with `model`, `threshold`, `normal_rainfall_reference`, `features`, and `model_name`.

The fitted pipeline is:

1. StandardScaler for rainfall variables.
2. OneHotEncoder for State (`handle_unknown='ignore'`).
3. Seasonal features passed through.
4. LogisticRegression (`class_weight='balanced'`, `max_iter=1000`, `random_state=42`).

Input columns, in the saved order:

```text
State, Rainfall, Rainfall_Lag1, Rainfall_Lag2, Rainfall_3Month,
Rainfall_Anomaly, Month_Sin, Month_Cos, Is_Monsoon
```

`State` is a string; rainfall/seasonal continuous variables are floating point; `Is_Monsoon` is an integer. Monthly rainfall and lags are in mm. The three-month feature is their sum. Anomaly is current rainfall minus the saved state-month normal. The saved reference has 264 rows and comes from the notebook's 1970–2000 climatology. Month sine/cosine use `2π × month / 12`; June–September maps to monsoon = 1. Year and district are not model inputs.

Target `Flood`: 1 denotes at least one inventory event for a state-month; unmatched records were assigned 0 in training. Positive label is assigned when the original `predict_proba` score for class 1 is **≥ 0.45**, rather than using the estimator's default 0.50 decision. The class-weighted score has not been calibrated and must not be interpreted as a certain district forecast.

Prediction supports all states in the saved rainfall reference, independently of district availability. District lookups use exact names and do not invent geographical mappings. For example, a model state without an exact district-table match receives a clear message and no substituted district. Telangana has a reference but was absent from the training split; the prediction details explain the original encoder's unknown-category behavior.

### Historical and manual rainfall

Select **Historical Rainfall** to choose a year/month from `rainfall_subdivision_monthly_1901_2015.csv`, then run predictions for all supported states with valid inputs. The result list prioritizes Flood states, displays scores where supported, and allows any evaluated state to be selected for District Analysis. Skipped states and reasons are shown in prediction details. `utils/rainfall.py` uses the exact subdivision mapping from training notebook cell 8, the same unweighted mean of available subdivisions, the 1970–2015 window, and the notebook's state-formation filters. The provenance expander reports mapped subdivisions and observed subdivision counts. Partial subdivision coverage is averaged as in training; missing state-month values are never interpolated. Both preceding calendar months must exist, including across year boundaries. **Custom Rainfall** enables manual inputs without a year. Missing historical files or complete triplets leave custom entry available.

Historical rainfall is aggregated to state level, not measured for the selected district. Selecting an old observation runs the saved model on that observation; it does not establish a historical flood occurrence or provide a live forecast. Historical exposure stays a separate district planning proxy.

The results separate the classifier prediction and uncalibrated probability score, historical population exposure, and calculated resources. Probability is shown only when `predict_proba` is supported; otherwise inference uses the classifier's label without a probability or probability threshold. Assistance coverage is a user-chosen supply planning fraction, never a predicted impact or flood probability. The prominent missing-regression warning has been replaced with quiet, explicit explanatory text.

### District clustering

`models/floodrelief_model2_complete.pkl` contains `model` (KMeans), fitted `imputer` and `scaler`, `features`, `cluster_names`, `model_name`, `n_clusters`, and `silhouette_score`.

Input: `Corrected_Percent_Flooded_Area`, `DFSI`, `Mean_Flood_Duration`, in that order. Missing values use the saved median imputer, followed by saved StandardScaler and K-means (`n_clusters=3`, `n_init=20`, `random_state=42`). No preprocessing is refitted. Names: 0 = Low Flood Impact, 1 = High Flood Impact, 2 = Long-Duration Flood Impact. These describe historical patterns; they are not flood event probabilities.

## Integration and planning population

```text
State + month + rainfall inputs
  → notebook feature engineering → saved flood pipeline → score/label
  → selected state → district record with historical exposure
  → assistance coverage and duration → relief quantities (either prediction label)

District clustering notebook: historical area + DFSI + duration
  → fitted imputer → fitted scaler → K-means → saved impact group
Dashboard: original district_clusters.csv → display saved assignments unchanged
```

There is no learned rainfall-to-population relationship in this integration. The classifier does not output the district-level inputs required by a future population model. The app does not multiply population by the flood score to manufacture an affected count.

In all 732 supplied processed rows, `Population_Exposure` matches `max(1, round(Population × Corrected_Percent_Flooded_Area / 100))`. This is an observed dataset relationship, not a verified reconstruction of the upstream generation code. The population-exposure source CSV referenced by the clustering notebook is not supplied separately. Exposure is not a measured event-specific affected population.

Relief scenarios remain available after No Flood results, or for a manually explored state that was not in the latest prediction run. The original classifier label is preserved and is shown as context, not used to force supplies to zero. This supports preparedness planning without treating exposure as predicted affected population. Missing/invalid exposure blocks relief calculations rather than substituting a value. Zero coverage produces zero resources.

## Relief formulas and sources

Defaults reproduce `notebooks/Relief_estimation_module.ipynb` and all saved relief CSV rows. Let E be the selected district's historical exposure proxy; c is assistance coverage from 0 to 1; P = ceil(E × c); D is the positive integer number of relief days (default 7). These formulas are unchanged; only the former UI gate that forced zero resources after No Flood has been removed.

| Quantity | Formula | Unit / assumption |
|---|---|---|
| People planned for | ceil(E × c) | People |
| Basic water | P × D × 15 | Litres, covering drinking, cooking and hygiene |
| Drinking allocation | P × D × 3 | Litres, included in basic water |
| Food energy | P × D × 2,100 | kcal |
| Meal equivalents | P × D × 3 | Project convention; 700 kcal per equivalent |
| Emergency medical kits | ceil(P / 100) | Project assumption per planning operation |

The **15 L/person/day is not drinking water alone**. The [Sphere Handbook (2018)](https://spherestandards.org/wp-content/uploads/Sphere-Handbook-2018-EN.pdf), Water supply standard 2.1, printed pp. 104–105, supports a basic water baseline and gives a drinking-and-food consumption range of 2.5–3 L/person/day. The app allocates 3 L as a simplified drinking portion. Food assistance standard 6.1, printed p. 199, gives a 2,100 kcal/person/day general ration planning baseline, subject to context.

Three meal equivalents, seven days, and one kit per 100 people are project conventions from the relief notebook. Kit contents, medicine quantities, and replenishment are undefined. Kit count is independent of duration, matching that notebook; it is not a validated medical needs forecast. The [WHO IEHK 2024](https://www.who.int/emergencies/emergency-health-kits/interagency-emergency-health-kit) serves an outreach population of 10,000 for three months and is a different specification. Project kits must not be presented as WHO kits or medication prescriptions.

## Evaluation provenance

The app reads **saved notebook outputs without executing cells**. For the deployed classifier's 0.45 threshold, `notebooks/FloodRelief_AI.ipynb`, zero-based cell 32, records:

| Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| 0.8264136022 | 0.6123234917 | 0.7768729642 | 0.6848528356 | 0.7372488408 |

These results cover 2,529 test records. F2 is taken directly from saved cell 32 output; it is not fabricated or derived from training predictions. Training is through 2005 and test data span 2006–2015. Threshold selection uses a 2001–2005 validation window. Cell 28 records ROC-AUC 0.8867606161 and an earlier classification report at **0.50**; the original notebook is preserved, while the dashboard shows only the five requested classifier metrics and optional provenance/ROC-AUC. These are notebook-reported results, not a new evaluation of the saved artifact. If notebook outputs are missing, no replacement scores appear.

Clustering's **0.4420 silhouette score** is saved in the model package and recorded in `FloodRelief_model2.ipynb`. It measures fitted dataset separation, not out-of-sample classification accuracy. Population regression metrics and actual-versus-predicted values are unavailable. Arithmetic relief estimation has no accuracy, MAE, RMSE, or R² score.

## Data quality and limitations

- The processed district tables have 732 rows, 729 unique district/state keys and 37 state labels. These labels include historical names and are not a statement about present administrative boundaries.
- Both processed tables contain one missing DFSI and 11 missing mean durations.
- Raigarh/Chhattisgarh, Aurangabad/Maharashtra and Balrampur/Uttar Pradesh each have two conflicting records. The app uses explicit source-row selection and does not silently merge or delete them. Tables count source records.
- Raw inventory has 6,876 records; monthly subdivision rainfall has 4,116 rows; district rainfall normals have 641 rows; DFSI has 744 rows. The two raw district impact/area tables each have 732 rows.
- The classifier notebook's processed training table is not exported here. Notebook source data are present, but no new held-out evaluation is fabricated from them.
- Source join provenance, population vintage and DFSI's definition need further documentation. A source file's presence does not prove it was used in the fitted model.
- State-month classification cannot establish a district flood or event-specific population impact. The model requires current-month rainfall, so manually entered scenarios are not live early-warning forecasts.
- The notebook compares several models on the test period; an untouched temporal/geographic holdout would strengthen final claims. No calibrated probabilities, predictive intervals, or population regression are supplied.

## Verification

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py utils tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks load the actual artifacts, reproduce a recorded notebook example using genuine rainfall records, compare all-state historical inference with individual classifier calls, verify relief remains available after No Flood, verify original cluster assignments including imputation, compare relief arithmetic against every saved relief row, reject invalid inputs, and render all four sections using Streamlit AppTest. Interaction checks cover state/district carryover, duplicate source records, unmatched states, search/filter behavior, resource quantities, missing assets, and navigation persistence. AppTest checks rendering and widget execution headlessly; it is not a visual browser layout review.

Next substantive work is to supply and validate the teammate's population regressor, agree its schema and target units, resolve district identifiers, and document medical supply assumptions. No new training is performed here.
