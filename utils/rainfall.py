"""Historical scenarios using the classifier notebook's state aggregation."""

import pandas as pd

# Copied from FloodRelief_AI.ipynb, cell 8; no district-to-state guesses.
STATE_SUBDIVISIONS = {
    "Andhra Pradesh": ["COASTAL ANDHRA PRADESH", "RAYALSEEMA"],
    "Arunachal Pradesh": ["ARUNACHAL PRADESH"],
    "Assam": ["ASSAM & MEGHALAYA"], "Bihar": ["BIHAR"],
    "Chhattisgarh": ["CHHATTISGARH"],
    "Gujarat": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
    "Haryana": ["HARYANA DELHI & CHANDIGARH"],
    "Himachal Pradesh": ["HIMACHAL PRADESH"],
    "Jammu and Kashmir": ["JAMMU & KASHMIR"], "Jharkhand": ["JHARKHAND"],
    "Karnataka": ["COASTAL KARNATAKA", "NORTH INTERIOR KARNATAKA", "SOUTH INTERIOR KARNATAKA"],
    "Kerala": ["KERALA"],
    "Madhya Pradesh": ["EAST MADHYA PRADESH", "WEST MADHYA PRADESH"],
    "Maharashtra": ["KONKAN & GOA", "MADHYA MAHARASHTRA", "MATATHWADA", "VIDARBHA"],
    "Meghalaya": ["ASSAM & MEGHALAYA"], "Odisha": ["ORISSA"],
    "Punjab": ["PUNJAB"], "Rajasthan": ["EAST RAJASTHAN", "WEST RAJASTHAN"],
    "Tamil Nadu": ["TAMIL NADU"], "Telangana": ["TELANGANA"],
    "Uttar Pradesh": ["EAST UTTAR PRADESH", "WEST UTTAR PRADESH"],
    "West Bengal": ["GANGETIC WEST BENGAL", "SUB HIMALAYAN WEST BENGAL & SIKKIM"],
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
FORMATION_DATES = {"Chhattisgarh": "2000-12-01", "Jharkhand": "2000-12-01", "Telangana": "2014-07-01"}


def historical_scenarios(source, state):
    """Return only real consecutive-month triplets, without filling missing data.

    Like the training notebook, state rainfall is the unweighted mean of available
    mapped subdivisions. Counts expose partial subdivision coverage to the UI.
    """
    required = {"SUBDIVISION", "YEAR", *MONTHS}
    if not required.issubset(source.columns):
        raise ValueError("Historical rainfall file is missing required subdivision/year/month columns.")
    if state not in STATE_SUBDIVISIONS:
        raise ValueError("No training-notebook rainfall mapping is available for this state.")
    subset = source.loc[source.SUBDIVISION.isin(STATE_SUBDIVISIONS[state])].copy()
    subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")
    subset = subset.loc[subset.YEAR.between(1970, 2015)]
    if subset.duplicated(["SUBDIVISION", "YEAR"]).any():
        raise ValueError("Duplicate subdivision/year records make historical rainfall ambiguous.")
    long = subset.melt(id_vars=["SUBDIVISION", "YEAR"], value_vars=MONTHS,
                       var_name="Month", value_name="Rainfall")
    long["Rainfall"] = pd.to_numeric(long.Rainfall, errors="coerce")
    long.loc[~long.Rainfall.between(0, float("inf"), inclusive="left"), "Rainfall"] = float("nan")
    long["Month"] = long.Month.map({name: i + 1 for i, name in enumerate(MONTHS)})
    monthly = long.groupby(["YEAR", "Month"]).agg(
        Rainfall=("Rainfall", "mean"), Observed_subdivisions=("Rainfall", "count")
    ).reset_index()
    monthly["Date"] = pd.to_datetime(dict(year=monthly.YEAR, month=monthly.Month, day=1))
    monthly = monthly.set_index("Date").sort_index()
    if state in FORMATION_DATES:
        monthly = monthly.loc[monthly.index >= FORMATION_DATES[state]]
    # Calendar-index lookups prevent skipping missing months or years.
    for lag in (1, 2):
        dates = monthly.index - pd.DateOffset(months=lag)
        monthly[f"Rainfall_Lag{lag}"] = monthly.Rainfall.reindex(dates).to_numpy()
        monthly[f"Observed_subdivisions_Lag{lag}"] = monthly.Observed_subdivisions.reindex(dates).to_numpy()
    return monthly.dropna(subset=["Rainfall", "Rainfall_Lag1", "Rainfall_Lag2"]).reset_index()


def historical_catalogue(source, states):
    """Collect available scenarios for every supported state, including unmapped districts."""
    frames, unavailable = [], []
    for state in sorted(states):
        try:
            frame = historical_scenarios(source, state)
            if frame.empty:
                unavailable.append({"State": state, "Reason": "No complete historical triplets"})
            else:
                frames.append(frame.assign(State=state))
        except ValueError as exc:
            unavailable.append({"State": state, "Reason": str(exc)})
    catalogue = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return catalogue, unavailable
