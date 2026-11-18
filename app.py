"""FloodRelief AI — run with: python -m streamlit run app.py."""

import streamlit as st

from utils.model_loader import dataset_inventory, notebook_evaluation
from utils.dashboard import (
    safe_model, safe_districts, navigate, prediction_page, analysis_page, relief_page,
)

st.set_page_config(page_title="FloodRelief AI", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
[data-testid="stAppViewContainer"] {background: #0B1120; color: #F8FAFC;}
[data-testid="stSidebar"] {background: #111827;}
[data-testid="stMetric"] {background: #1E293B; padding: 1rem; border: 1px solid #334155; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #94A3B8;}
[data-testid="stMetricValue"] {color: #F8FAFC; font-size: clamp(1.3rem, 2vw, 2rem);}
[data-testid="stCaptionContainer"] {color: #94A3B8;}
[data-testid="stSidebar"] button {border-radius: 10px; text-align: left; transition: border-color .15s;}
[data-testid="stSidebar"] button:hover {border-color: #22D3EE;}
h1, h2, h3 {color: #F8FAFC;}
</style>
""", unsafe_allow_html=True)

SPHERE = "https://spherestandards.org/wp-content/uploads/Sphere-Handbook-2018-EN.pdf"
WHO = "https://www.who.int/emergencies/emergency-health-kits/interagency-emergency-health-kit"


def relief_methodology():
    st.markdown("""
Let **E** be the planning population proxy, **c** the selected assistance coverage,
**P = ceil(E × c)** the people planned for, and **D** the relief duration in days.
E is the selected district historical exposure. Planning remains available even when the state is classified as No Flood; the prediction label does not zero out this independent scenario.

| Output | Formula | Basis |
|---|---|---|
| Basic water (litres) | P × D × 15 | Drinking, cooking and hygiene together |
| Drinking allocation (litres) | P × D × 3 | Included within basic water, not additional |
| Food energy (kcal) | P × D × 2,100 | General ration planning baseline |
| Meal equivalents | P × D × 3 | Project convention: 700 kcal per meal equivalent |
| Emergency medical kits | ceil(P / 100) | Project assumption: one kit per 100 people per planning operation |
""")
    st.markdown(
        f"The [Sphere Handbook (2018)]({SPHERE}), Water supply standard 2.1 "
        "(printed pp. 104–105), supports the basic water allowance; its indicative "
        "consumption range is 2.5–3 L/person/day for drinking and food. This app "
        "allocates 3 L/person/day as a simplified drinking portion. Food assistance "
        "standard 6.1 (printed p. 199) gives the 2,100 kcal baseline, adjusted to context."
    )
    st.info(
        "Seven days, three meal equivalents per day, and one medical kit per 100 people "
        "come from the project relief notebook. Kit contents and replenishment are "
        "undefined; kits do not scale with duration in that notebook. This is a supply "
        "planning placeholder, not a medicine prescription or a validated clinical requirement."
    )
    st.markdown(
        f"The project kit assumption is distinct from the [WHO IEHK 2024]({WHO}), "
        "which is designed for 10,000 people over three months. The application "
        "does not convert its project kits into WHO kits."
    )


def explanation_page():
    with st.expander("Problem statement and objectives", expanded=False):
        st.write("Flood response requires estimating where flooding may occur and translating population needs into resource quantities. This semester project connects a state-month flood classifier with district impact profiles and transparent relief planning calculations. It aims to make the pipeline demonstrable, interpretable, and explicit about missing capabilities.")
    with st.expander("Datasets and provenance"):
        st.dataframe(dataset_inventory(), width="stretch", hide_index=True)
        st.write("FloodRelief_AI.ipynb uses the flood inventory and monthly subdivision rainfall dataset. Flood events are aggregated to state-month labels. Rainfall subdivisions are mapped to 22 states and averaged. The raw rainfall file spans 1901–2015; the classifier notebook restricts it to 1970–2015.")
        st.write("District source files contain impact, flooded area, and DFSI information. The clustering notebook starts from district_population_exposure.csv, which is not supplied separately; its processed values are present in district_clusters.csv. The exact upstream join workflow and DFSI definition are not fully documented locally. District rainfall normals are available but are not substituted for the classifier's saved state-month reference.")
    with st.expander("Preprocessing and classifier"):
        st.write("The notebook normalizes historical state names, merges rainfall and flood records, assigns zero to unmatched flood labels, filters some pre-formation state records, and constructs rainfall lags before dropping rows with missing rainfall. Absence of a recorded flood is treated as label zero, which can reflect incomplete reporting.")
        st.write("Inputs are State, current rainfall, two rainfall lags, their three-month sum, rainfall anomaly, sine/cosine of month, and a June–September monsoon indicator. The anomaly uses a 1970–2000 climatology saved in the artifact. StandardScaler handles the five rainfall features; OneHotEncoder handles State; seasonal features pass through. The saved pipeline uses class-weighted Logistic Regression and threshold 0.45.")
        st.write("Year is scenario metadata in the notebook helper, not a trained model feature. Current-month rainfall must be known or supplied as a scenario: this is not a real-time early-warning forecast.")
    with st.expander("District clustering methodology"):
        st.write("FloodRelief_model2.ipynb uses median imputation and standard scaling on corrected flooded-area percentage, DFSI, and mean flood duration. It compares K=2–8 using inertia and silhouette score, then fits K-means with K=3, random_state=42, n_init=20. Cluster names are manually assigned descriptions: Low Flood Impact, High Flood Impact, and Long-Duration Flood Impact.")
        st.write("The second saved artifact contains K-means plus its fitted imputer and scaler. It is not a population regressor. District Analysis displays the original saved assignments directly; no assignments are recalculated by the dashboard.")
    with st.expander("Complete pipeline and population impact"):
        st.write("State + month + rainfall scenario → original flood preprocessing and classifier → select state → select a historical district record → assistance coverage → relief quantities. Relief scenarios remain available for either Flood or No Flood labels.")
        st.write("Historical district features were processed using median imputation, standard scaling and K-means in the supplied notebook. District Analysis displays those saved cluster assignments; cluster IDs are not fed into a population regressor.")
        st.write("In all supplied district rows, Population_Exposure matches max(1, round(Population × Corrected_Percent_Flooded_Area / 100)). This is an observed relationship in the data, not proof of the unavailable upstream generation code. Spatial exposure does not establish actual affected people or uniform population density.")
        st.write("A future population regression model must define its feature schema, people-valued target, geographical coverage, and held-out evaluation. Flood risk scores alone cannot supply flooded area, DFSI, or duration.")
    with st.expander("Relief methodology and sources"):
        relief_methodology()
    with st.expander("Technologies and reproducibility"):
        st.write("Python, Streamlit, pandas, NumPy, scikit-learn and Joblib. Model artifacts and CSVs are loaded locally; the app makes no external API calls and performs no training. Model loading is cached. scikit-learn 1.6.1 matches the saved artifacts and avoids the _RemainderColsList version mismatch without monkey-patching.")
        st.write("Use the pinned requirements in a fresh Python 3.11–3.13 environment. Python 3.13.2 is the local verification environment; the original training Python version was not recorded in the supplied notebooks.")
    with st.expander("Limitations and future scope"):
        st.write("The classifier operates at state-month resolution, while district data describe historical impact. This scale mismatch prevents district-specific flood claims. Monthly rainfall is entered manually or loaded from historical observations; no live rainfall feed is used. Population regression and uncertainty intervals are absent. Saved model scores are not calibrated risk probabilities.")
        st.write("The processed district table contains missing DFSI/duration values and conflicting duplicate district/state keys; no source values are overwritten. Historical boundaries, population vintage, raw source units, and district joins need further validation. Missing values remain visible in analysis; missing planning population blocks relief calculation.")
        st.write("Future work: complete and validate population regression, resolve district identifiers, use an untouched temporal/geographic holdout, validate forecast rainfall inputs, quantify uncertainty, and replace the generic kit assumption with locally assessed medical supply requirements.")


def overview_page():
    st.write("FloodRelief AI connects state-level flood occurrence prediction, historical district impact analysis, and transparent resource planning.")
    st.markdown("""
- Predict state-level flood occurrence using machine learning and identify states classified as **Flood**.
- Analyze historical district flood impact using the saved clustering results.
- Estimate drinking water, food, and emergency medical kit requirements for a chosen district scenario.
""")
    flood = safe_model("floodrelief_model1_complete.pkl")
    cluster = safe_model("floodrelief_model2_complete.pkl")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Flood prediction**")
        st.subheader(flood["model_name"] if flood else "Model unavailable")
        st.write("Combines rainfall history and seasonal features to classify state-month flood occurrence.")
    with b.container(border=True):
        st.markdown("**Historical district impact**")
        st.subheader(cluster["model_name"] if cluster else "Model unavailable")
        st.write("Groups similar historical flooded area, DFSI and flood-duration profiles into impact clusters.")
    st.subheader("Recorded classifier performance")
    evaluation = notebook_evaluation()
    if evaluation:
        for col, name in zip(st.columns(5), ["Accuracy", "Precision", "Recall", "F1", "F2"]):
            col.metric({"F1": "F1-score", "F2": "F2-score"}.get(name, name), f"{evaluation['metrics'][name]:.2%}")
        st.caption("Genuine saved notebook results · 2006–2015 test period · decision threshold 0.45. F2-score gives greater weight to recall.")
    else:
        st.info("Saved evaluation outputs are unavailable; no scores are substituted.")
    st.subheader("Project workflow")
    st.write("Flood Prediction → State Selection → District Analysis → Relief Planning")
    st.caption("Historical population exposure is a planning proxy, not a population regression prediction. A state-level flood label does not establish which individual districts will flood.")
    st.button("Start Flood Prediction", type="primary", icon=":material/arrow_forward:", key="start_prediction", on_click=navigate, args=("Flood Prediction",))
    with st.expander("Model evaluation details"):
        if evaluation:
            st.write(f"Source: {evaluation['source']}, cell index {evaluation['cell']} (zero-based). These are notebook-reported results, not a fresh evaluation of the deployed artifact.")
            st.write("Training through 2005; threshold selection uses 2001–2005 validation. Several models were compared on the test period, so an untouched temporal/geographic holdout would strengthen the final claim.")
            if evaluation["roc_auc"] is not None:
                st.write(f"Notebook-reported ROC-AUC: {evaluation['roc_auc']:.4f}.")
        if cluster:
            st.write(f"Saved clusters: {cluster['n_clusters']}; saved silhouette score: {cluster['silhouette_score']:.4f}. This measures separation in the fitted dataset, not flood prediction accuracy.")
        st.write("No population regression artifact or held-out regression results are available. Arithmetic relief formulas have no ML accuracy or regression score.")
    explanation_page()


st.session_state.setdefault("active_page", "Overview")
with st.sidebar:
    st.subheader("🌊 FloodRelief AI")
    st.caption("Flood impact & relief planning")
    st.divider()
    items = [
        ("Overview", ":material/home:"),
        ("Flood Prediction", ":material/water_drop:"),
        ("District Analysis", ":material/location_on:"),
        ("Relief Planning", ":material/medical_services:"),
    ]
    for page, icon in items:
        st.button(page, icon=icon, key=f"nav_{page}", width="stretch",
                  type="primary" if st.session_state.active_page == page else "tertiary",
                  on_click=navigate, args=(page,))
    st.divider()
    st.caption("M.Tech AI/ML semester project")
    st.caption("Local models · historical data · transparent assumptions")

st.title("FloodRelief AI")
st.markdown("**Intelligent Flood Impact and Relief Resource Planning System**")
{
    "Overview": overview_page,
    "Flood Prediction": prediction_page,
    "District Analysis": analysis_page,
    "Relief Planning": lambda: relief_page(relief_methodology),
}[st.session_state.active_page]()
st.divider()
st.caption("Academic planning demonstration. Results depend on historical coverage, scenario inputs, and stated assumptions.")
