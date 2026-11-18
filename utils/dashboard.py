"""Connected dashboard views. Model preprocessing and formulas stay in prediction.py."""

import calendar

import pandas as pd
import streamlit as st

from utils.model_loader import load_model, load_districts, load_historical_catalogue, dataset_inventory
from utils.prediction import (
    build_flood_input, predict_flood, predict_historical_states,
    estimate_relief_resources, finite_number,
)
from utils.rainfall import STATE_SUBDIVISIONS


def navigate(page):
    st.session_state["active_page"] = page


def select_state(state):
    if st.session_state.get("selected_state") != state:
        for key in ("selected_district_row", "district_name", "district_record"):
            st.session_state.pop(key, None)
    st.session_state["selected_state"] = state


def _save_input(name, callback=None):
    value = st.session_state[f"_{name}"]
    st.session_state[name] = value
    if callback:
        callback(value)


def select_input(label, options, name, default=None, callback=None, **kwargs):
    """Keep durable values separate from widgets, which Streamlit cleans on navigation."""
    options = list(options)
    value = st.session_state.get(name, default)
    if value not in options:
        value = options[0]
    st.session_state[name] = value
    st.session_state[f"_{name}"] = value
    return st.selectbox(label, options, key=f"_{name}", on_change=_save_input,
                        args=(name, callback), **kwargs)


def number_input(label, name, default, **kwargs):
    st.session_state.setdefault(name, default)
    st.session_state[f"_{name}"] = st.session_state[name]
    return st.number_input(label, key=f"_{name}", on_change=_save_input, args=(name,), **kwargs)


def safe_model(filename):
    try:
        return load_model(filename)
    except Exception as exc:
        st.warning(str(exc))
        return None


def safe_districts():
    try:
        return load_districts()
    except (OSError, ValueError) as exc:
        st.warning(str(exc))
        return None


def prediction_page():
    st.header("Flood Prediction")
    st.write("Explore historical rainfall across states or assess a custom rainfall scenario.")
    package = safe_model("floodrelief_model1_complete.pkl")
    if package is None:
        return
    supported = sorted(package["normal_rainfall_reference"].State.unique())
    mode = select_input("Rainfall mode", ["Historical Rainfall", "Custom Rainfall"], "rainfall_mode")
    if mode == "Historical Rainfall":
        historical_prediction(package, supported)
    else:
        custom_prediction(package, supported)
    show_prediction_results()


def historical_prediction(package, supported):
    try:
        catalogue, unavailable = load_historical_catalogue(tuple(supported))
    except (OSError, ValueError) as exc:
        st.info(f"Historical observations unavailable: {exc} You can use Custom Rainfall instead.")
        return
    if catalogue.empty:
        st.info("No complete historical rainfall triplets are available. Custom Rainfall remains available.")
        return
    a, b = st.columns(2)
    with a:
        year = select_input("Historical year", sorted(catalogue.YEAR.astype(int).unique(), reverse=True), "historical_year")
    with b:
        months = sorted(catalogue.loc[catalogue.YEAR == year, "Month"].astype(int).unique())
        month = select_input("Month", months, "historical_month", format_func=lambda x: calendar.month_name[x])
    st.caption("Runs the saved model for every supported state with a complete current/previous/two-month-earlier rainfall triplet. These are historical state-level observations, not district measurements or observed flood labels.")
    if st.button("Predict all supported states", type="primary", key="run_historical"):
        try:
            results, skipped = predict_historical_states(package, catalogue, year, month)
            if results.empty:
                st.info("No states have valid inputs for this selection.")
            else:
                results = results.sort_values(["Flood", "State"], ascending=[False, True]).reset_index(drop=True)
                st.session_state["prediction_results"] = {
                    "mode": "Historical Rainfall", "context": f"{calendar.month_name[month]} {year}",
                    "results": results, "skipped": skipped, "threshold": package["threshold"],
                }
                # Preserve the previous selected state if it is still in these results.
                state = st.session_state.get("selected_state")
                if state not in results.State.tolist():
                    state = results.State.iloc[0]
                select_state(state)
        except Exception as exc:
            st.error(f"Historical prediction could not be completed: {exc}")
    with st.expander("Historical input provenance"):
        available = catalogue.loc[(catalogue.YEAR == year) & (catalogue.Month == month)]
        columns = ["State", "Rainfall", "Rainfall_Lag1", "Rainfall_Lag2", "Observed_subdivisions",
                   "Observed_subdivisions_Lag1", "Observed_subdivisions_Lag2"]
        st.dataframe(available[columns], hide_index=True, width="stretch")
        st.caption("Source: rainfall_subdivision_monthly_1901_2015.csv. Rainfall units: mm. Calendar lags include year boundaries. State averages, period 1970–2015 and state-formation filters match the training notebook. Missing state-month observations are not filled.")
        st.write("Subdivision mapping from the training notebook:", STATE_SUBDIVISIONS)
        if unavailable:
            st.dataframe(pd.DataFrame(unavailable), hide_index=True)


def custom_prediction(package, supported):
    a, b = st.columns(2)
    with a:
        state = select_input("State", supported, "custom_state", st.session_state.get("selected_state"))
    with b:
        month = select_input("Month", range(1, 13), "custom_month", format_func=lambda x: calendar.month_name[x])
    a, b, c = st.columns(3)
    with a:
        rainfall = number_input("Current month rainfall (mm)", "custom_rainfall", 0.0, min_value=0.0, step=1.0)
    with b:
        lag1 = number_input("Previous month rainfall (mm)", "custom_lag1", 0.0, min_value=0.0, step=1.0)
    with c:
        lag2 = number_input("Rainfall two months earlier (mm)", "custom_lag2", 0.0, min_value=0.0, step=1.0)
    st.caption("Enter state-level monthly rainfall for your scenario. No year is required. The three-month sum, saved-reference anomaly, month sine/cosine, and monsoon indicator are calculated automatically.")
    if st.button("Predict flood occurrence", type="primary", key="run_custom"):
        try:
            inputs = build_flood_input(package, state, month, rainfall, lag1, lag2)
            prediction = predict_flood(package, inputs)
            row = {"State": state, "Flood": prediction["label"],
                   "Prediction": "Flood" if prediction["label"] else "No Flood"}
            if prediction["score"] is not None:
                row["Model score (uncalibrated)"] = prediction["score"]
            st.session_state["prediction_results"] = {
                "mode": "Custom Rainfall", "context": f"{state} · {calendar.month_name[month]}",
                "results": pd.DataFrame([row]), "inputs": inputs,
                "threshold": prediction["threshold"], "skipped": [],
            }
            select_state(state)
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


def show_prediction_results():
    completed = st.session_state.get("prediction_results")
    if completed is None:
        return
    results = completed["results"]
    st.subheader("Latest completed prediction")
    st.caption(f"{completed['mode']} · {completed['context']}. Input changes take effect when you run prediction again.")
    a, b = st.columns(2)
    flood_states = results.loc[results.Flood == 1, "State"].tolist()
    if completed["mode"] == "Historical Rainfall":
        a.metric("States evaluated", str(len(results)))
        b.metric("States classified as Flood", str(len(flood_states)))
        st.write("**Classified as Flood:** " + (", ".join(flood_states) if flood_states else "None in this run."))
    else:
        a.metric("Flood prediction", results.Prediction.iloc[0])
        if "Model score (uncalibrated)" in results:
            b.metric("Model score (uncalibrated)", f"{results['Model score (uncalibrated)'].iloc[0]:.1%}")
    columns = ["State", "Prediction"]
    if "Model score (uncalibrated)" in results:
        columns.append("Model score (uncalibrated)")
        st.caption(f"Scores come from predict_proba and are uncalibrated. Saved decision threshold: {completed['threshold']:.2f}. They are not district flood probabilities.")
    st.dataframe(results[columns], width="stretch", hide_index=True)
    if completed["mode"] == "Historical Rainfall":
        # Always follows the durable workflow state; does not reset on page switches.
        state = st.session_state.get("selected_state")
        st.session_state["_result_state"] = state if state in results.State.tolist() else None
        st.selectbox("Explore a state from these results", results.State.tolist(), index=None,
                     placeholder="Choose a state from this run", key="_result_state",
                     on_change=_save_input, args=("result_state", select_state))
        if state not in results.State.tolist():
            st.caption(f"Current district exploration state: {state}. Choose a result above to switch back to an evaluated state.")
    else:
        st.caption(f"Prediction state: {results.State.iloc[0]}. Current District Analysis state: {st.session_state.get('selected_state')}.")
    st.button("Continue to District Analysis", icon=":material/arrow_forward:", key="to_districts", on_click=navigate, args=("District Analysis",))
    with st.expander("Prediction details and data availability"):
        if "inputs" in completed:
            st.dataframe(completed["inputs"], hide_index=True, width="stretch")
        if completed["skipped"]:
            st.dataframe(pd.DataFrame(completed["skipped"]), hide_index=True, width="stretch")
        st.caption("Only supported states with valid inputs are evaluated. Telangana has a saved reference but was absent from the original training split; its category is handled as unknown by the saved encoder. A No Flood label does not guarantee safety.")


DISTRICT_COLUMNS = {
    "District": "District", "Cluster": "Cluster ID", "Vulnerability_Group": "Historical cluster label",
    "Corrected_Percent_Flooded_Area": "Corrected flooded area (%)", "DFSI": "DFSI",
    "Mean_Flood_Duration": "Mean duration (days)", "Population_Exposure": "Historical exposure (people)",
}


def analysis_page():
    st.header("District Analysis")
    st.write("Explore all district records in the chosen state and select one for relief planning.")
    frame = safe_districts()
    if frame is None:
        return
    options = set(frame.State)
    if st.session_state.get("selected_state"):
        options.add(st.session_state["selected_state"])
    options = sorted(options)
    st.session_state["analysis_state"] = st.session_state.get("selected_state", options[0])
    state = select_input("State", options, "analysis_state", callback=select_state)
    select_state(state)
    records = frame.loc[frame.State == state]
    if records.empty:
        st.info(f"No matching district records are available for {state}. No other state has been substituted. Choose another state explicitly to explore.")
        return
    st.caption("Cluster labels describe historical flood impact, not district-level flood forecasts. Assignments below are read directly from district_clusters.csv.")
    a, b = st.columns(2)
    a.metric("Available district records", str(len(records)))
    b.metric("Historical impact clusters", str(records.Cluster.nunique()))
    groups = ["All clusters"] + sorted(records.Vulnerability_Group.unique())
    group = select_input("Filter by cluster label", groups, f"cluster_filter_{state}")
    search = st.text_input("Search districts", key=f"search_{state}", placeholder="Type a district name")
    shown = records if group == "All clusters" else records.loc[records.Vulnerability_Group == group]
    if search.strip():
        shown = shown.loc[shown.District.str.contains(search.strip(), case=False, regex=False)]
    st.dataframe(shown[list(DISTRICT_COLUMNS)].rename(columns=DISTRICT_COLUMNS), width="stretch", hide_index=True)
    st.caption(f"Showing {len(shown)} of {len(records)} source records. Filters affect the table only; every district remains selectable below. Duplicate district names retain their separate source records.")
    district_name = select_input("District for relief planning", sorted(records.District.unique()), "district_name")
    matches = records.loc[records.District == district_name]
    if len(matches) > 1:
        st.caption("Multiple source records share this name. Choose the population/exposure record explicitly.")
        index = select_input("Source record", matches.index.tolist(), "district_record",
                             format_func=lambda i: f"CSV row {i + 2} · population {frame.loc[i, 'Population']:,.0f} · exposure {frame.loc[i, 'Population_Exposure']:,.0f}")
    else:
        index = matches.index[0]
    st.session_state["selected_district_row"] = int(index)
    district = frame.loc[index]
    st.subheader(f"Selected district · {district.District}")
    st.write(f"**Historical cluster {district.Cluster} · {district.Vulnerability_Group}**")
    st.dataframe(district[list(DISTRICT_COLUMNS)].astype("string").to_frame("Source value"), width="stretch")
    st.button("Continue to Relief Planning", type="primary", icon=":material/arrow_forward:", key="to_relief", on_click=navigate, args=("Relief Planning",))
    with st.expander("Dataset quality and inventory"):
        st.write("Missing values remain as supplied; no cluster labels are recalculated.")
        st.dataframe(frame.isna().sum().rename("Missing cells"))
        st.dataframe(frame[frame.duplicated(["State", "District"], keep=False)], hide_index=True, width="stretch")
        st.dataframe(dataset_inventory(), hide_index=True, width="stretch")


def relief_page(methodology):
    st.header("Relief Planning")
    frame = safe_districts()
    if frame is None:
        return
    index = st.session_state.get("selected_district_row")
    state = st.session_state.get("selected_state")
    if index not in frame.index or frame.loc[index, "State"] != state:
        st.info("Select a district in District Analysis to load its exposure and historical impact profile here.")
        st.button("Choose a district", key="choose_district", on_click=navigate, args=("District Analysis",))
        return
    district = frame.loc[index]
    st.subheader(f"{district.District} · {state}")
    st.caption(f"Historical cluster {district.Cluster} · {district.Vulnerability_Group} · CSV row {index + 2}")
    st.button("Change district", key="change_district", on_click=navigate, args=("District Analysis",))
    completed = st.session_state.get("prediction_results")
    if completed is not None:
        matching = completed["results"].loc[completed["results"].State == state]
        if not matching.empty:
            st.caption(f"Latest state-level prediction: {matching.Prediction.iloc[0]} · {completed['context']}. Relief planning is available for either prediction label.")
        else:
            st.caption("The selected state was not evaluated in the latest prediction run. You can still explore this district's planning scenario.")
    a, b = st.columns(2)
    with a:
        days = number_input("Relief duration (days)", "relief_days", 7, min_value=1, max_value=365, step=1)
    with b:
        coverage = number_input("Assistance coverage (%)", "assistance_coverage", 100.0, min_value=0.0, max_value=100.0, step=5.0,
                                help="Your planning choice: the fraction of historical district exposure you intend to assist. It is not predicted by the models.")
    st.caption("Coverage is a planning assumption. For example, 50% of 1,000 historically exposed people plans supplies for 500 people; it does not estimate how many people will actually be affected.")
    try:
        exposure = finite_number(district.Population_Exposure, "Historical population exposure")
        if exposure > finite_number(district.Population, "District population", 1):
            raise ValueError("Historical exposure exceeds district population.")
        result = estimate_relief_resources(exposure, days, coverage / 100)
    except ValueError as exc:
        st.error(f"Relief requirements cannot be calculated: {exc}")
        return
    a, b, c = st.columns(3)
    a.metric("Historical exposure (people)", f"{exposure:,.0f}")
    b.metric("Planning population (people)", f"{result['People_Planned_For']:,}")
    c.metric("Relief duration (days)", str(days))
    st.subheader("Calculated resource requirements")
    resources = [
        ("💧 Drinking Water", "Drinking_Water_Litres", "litres"),
        ("🚰 Basic Water", "Water_Litres", "litres"),
        ("🍽️ Food Energy", "Total_Food_Kcal", "kcal"),
        ("🥣 Meal Equivalents", "Total_Meals", "meals"),
        ("🧰 Emergency Medical Kits", "Emergency_Medical_Kits", "kits"),
    ]
    # Two balanced rows avoid truncating large food-energy quantities.
    for row in (resources[:3], resources[3:]):
        for col, (label, key, unit) in zip(st.columns(len(row)), row):
            col.metric(f"{label} · {unit}", f"{result[key]:,.0f}")
    st.caption("Drinking water is included within basic water. Historical population exposure is a planning proxy, not a regression prediction. State-level predictions do not establish which districts will flood. These quantities are scenario-based planning estimates.")
    with st.expander("Detailed calculations and assumptions"):
        st.write(f"Planning population = ceil({exposure:,.0f} × {coverage:g}%) = {result['People_Planned_For']:,} people.")
        methodology()
