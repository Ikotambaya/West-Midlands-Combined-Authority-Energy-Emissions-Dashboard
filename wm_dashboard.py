import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("westmidlands.xlsx", sheet_name="WML")
    return df

df = load_data()

# --- Constants ---
la_list = ['Birmingham', 'Coventry', 'Dudley', 'Sandwell', 'Solihull', 'Walsall', 'Wolverhampton']
years = np.arange(2016, 2042)
dates = pd.to_datetime(years.astype(str) + "-01-01")

# --- Sidebar ---
st.sidebar.title("Controls")
use_per_capita = st.sidebar.checkbox("Per Capita Emissions", value=False)
scenario = st.sidebar.selectbox("Scenario", ["Business-as-Usual", "Accelerated"])

st.title("🌍 West Midlands Emissions Forecast vs WM2041 Target")

metric = "Grand Total"
if use_per_capita:
    df[metric] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    metric_label = "Emissions (tCO2e per person)"
else:
    metric_label = "Emissions (kt CO2e)"

# --- Forecasting Function ---
@st.cache_data
def forecast_la(la, metric):
    la_df = df[df["Local Authority"] == la]
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(la_df["Calendar Year"].astype(str) + "-01-01")
    prophet_df["y"] = la_df[metric]

    model = Prophet(yearly_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=2041 - prophet_df["ds"].dt.year.max(), freq="Y")
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].rename(columns={"yhat": la})

# --- Forecast All LAs ---
forecast_df = pd.DataFrame({"ds": pd.date_range(start="2005", end="2042", freq="Y")})

for la in la_list:
    try:
        forecast = forecast_la(la, metric)
        forecast_df = forecast_df.merge(forecast, on="ds", how="left")
    except Exception as e:
        st.warning(f"Forecast failed for {la}: {e}")

forecast_df["Total Forecast"] = forecast_df[la_list].sum(axis=1)

# --- WM2041 Target Calculation ---
baseline_year = 2016
baseline_val = df[(df["Calendar Year"] == baseline_year) & (df["Local Authority"].isin(la_list))][metric].sum()
target_vals = []

for year in years:
    if year <= 2026:
        target = baseline_val - (baseline_val * 0.33 * (year - 2016) / 10)
    else:
        target = (baseline_val * 0.67) * (1 - (year - 2026) / 15)
    if scenario == "Business-as-Usual":
        target = baseline_val  # No reduction
    target_vals.append(target)

target_df = pd.DataFrame({"ds": dates, "WM2041 Target": target_vals})

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(forecast_df["ds"], forecast_df["Total Forecast"], label="Forecasted Total", linewidth=2)
ax.plot(target_df["ds"], target_df["WM2041 Target"], "k--", label=f"{scenario} Target", linewidth=2)
ax.fill_between(target_df["ds"], forecast_df["Total Forecast"], target_df["WM2041 Target"], color="red", alpha=0.1, label="Gap")

ax.set_title("Emissions Forecast vs. WM2041 Target")
ax.set_ylabel(metric_label)
ax.set_xlabel("Year")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- Summary Metrics ---
latest_year = forecast_df["ds"].dt.year.max()
latest_forecast = forecast_df.loc[forecast_df["ds"].dt.year == latest_year, "Total Forecast"].values[0]
target_2041 = target_df.loc[target_df["ds"].dt.year == 2041, "WM2041 Target"].values[0]
gap = latest_forecast - target_2041

st.subheader("📊 Summary")
st.metric("Latest Forecast (2041)", f"{latest_forecast:,.1f}")
st.metric("WM2041 Target", f"{target_2041:,.1f}")
st.metric("Forecast Overshoot", f"{gap:,.1f}", delta_color="inverse")
