import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
st.sidebar.title("Dashboard Controls")
use_per_capita = st.sidebar.checkbox("Per Capita Emissions", value=False)
scenario = st.sidebar.selectbox("Scenario", ["Business-as-Usual", "Accelerated"])

# --- Metric ---
metric = "Grand Total"
if use_per_capita:
    df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    metric = "Per Capita Emissions (tCO2e)"
    metric_label = "Emissions (tCO2e per person)"
else:
    metric_label = "Emissions (kt CO2e)"

# --- Title ---
st.title("🌍 West Midlands Emissions Dashboard")

# --- Forecasting ---
@st.cache_data
def forecast_la(la, metric):
    la_df = df[df["Local Authority"] == la]
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(la_df["Calendar Year"].astype(str) + "-01-01")
    prophet_df["y"] = la_df[metric]

    model = Prophet(yearly_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=2041 - prophet_df["ds"].dt.year.max(), freq="YE")
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].rename(columns={"yhat": la})

# --- Forecast All LAs ---
forecast_df = pd.DataFrame({"ds": pd.date_range(start="2005", end="2042", freq="YE")})
for la in la_list:
    try:
        forecast = forecast_la(la, metric)
        forecast_df = forecast_df.merge(forecast, on="ds", how="left")
    except Exception as e:
        st.warning(f"Forecast failed for {la}: {e}")

forecast_df["Total Forecast"] = forecast_df[la_list].sum(axis=1)

# --- WM2041 Target ---
baseline_year = 2016
baseline_val = df[(df["Calendar Year"] == baseline_year) & (df["Local Authority"].isin(la_list))][metric].sum()
target_vals = []
for year in years:
    if year <= 2026:
        target = baseline_val - (baseline_val * 0.33 * (year - 2016) / 10)
    else:
        target = (baseline_val * 0.67) * (1 - (year - 2026) / 15)
    if scenario == "Business-as-Usual":
        target = baseline_val
    target_vals.append(target)

target_df = pd.DataFrame({"ds": dates, "WM2041 Target": target_vals})

# --- Forecast Plot ---
st.subheader("📈 Forecast vs. WM2041 Target")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(forecast_df["ds"], forecast_df["Total Forecast"], label="Forecasted Total", linewidth=2)
ax.plot(target_df["ds"], target_df["WM2041 Target"], "k--", label=f"{scenario} Target", linewidth=2)

if len(forecast_df["Total Forecast"]) == len(target_df["WM2041 Target"]):
    ax.fill_between(forecast_df["ds"], forecast_df["Total Forecast"], target_df["WM2041 Target"], color="red", alpha=0.1, label="Gap")

ax.set_title("Emissions Forecast vs WM2041 Target")
ax.set_ylabel(metric_label)
ax.set_xlabel("Year")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Summary ---
latest_forecast = forecast_df.loc[forecast_df["ds"].dt.year == 2041, "Total Forecast"].values[0]
target_2041 = target_df.loc[target_df["ds"].dt.year == 2041, "WM2041 Target"].values[0]
gap = latest_forecast - target_2041

st.subheader("📊 Summary Metrics")
st.metric("Latest Forecast (2041)", f"{latest_forecast:,.1f}")
st.metric("WM2041 Target", f"{target_2041:,.1f}")
st.metric("Forecast Overshoot", f"{gap:,.1f}", delta_color="inverse")

# --- Time Series Trends ---
st.subheader("📊 Time Series Trends")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df[df["Local Authority"].isin(la_list)], x="Calendar Year", y="Grand Total", hue="Local Authority", marker='o', ax=ax)
ax.set_title('Total Emissions by Local Authority (2005–2022)')
ax.set_ylabel("kt CO2e")
ax.grid(True)
st.pyplot(fig)

# --- Faceted Trends ---
st.subheader("📈 Local Authority Breakdown")
g = sns.FacetGrid(df[df["Local Authority"].isin(la_list)], col="Local Authority", col_wrap=3, height=3.5, sharey=False)
g.map_dataframe(sns.lineplot, x="Calendar Year", y="Grand Total", marker='o')
g.set_titles("{col_name}")
g.set_axis_labels("Year", "kt CO2e")
st.pyplot(g.fig)

# --- Per Capita ---
st.subheader("👤 Per Capita Emissions")
df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df[df["Local Authority"].isin(la_list)], x="Calendar Year", y="Per Capita Emissions (tCO2e)", hue="Local Authority", ax=ax)
ax.set_title("Per Capita Emissions by LA (2005–2022)")
ax.set_ylabel("tCO2e per person")
ax.grid(True)
st.pyplot(fig)

# --- Emissions per km² ---
st.subheader("📏 Emissions Intensity by Area")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df[df["Local Authority"].isin(la_list)], x="Calendar Year", y="Emissions per km2 (kt CO2e)", hue="Local Authority", ax=ax)
ax.set_title("Emissions per km² by Local Authority (2005–2022)")
ax.set_ylabel("kt CO2e per km²")
ax.grid(True)
st.pyplot(fig)

# --- Avg Emissions Heatmap ---
st.subheader("🔥 Avg Emissions Intensity Heatmap")
pivot = df.pivot_table(index="Local Authority", values="Emissions per km2 (kt CO2e)", aggfunc="mean")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot, annot=True, cmap="Reds", ax=ax)
ax.set_title("Average Emissions per km² (2005–2022)")
st.pyplot(fig)

# --- Sectoral Emissions ---
st.subheader("🏭 Sectoral Emissions by LA (2022)")
sector_cols = [
    'Local Authority', 'Calendar Year',
    'Industry Total', 'Transport Total', 'Domestic Total',
    'Commercial Total', 'Public Sector Total',
    'Agriculture Total', 'Waste Total', 'LULUCF Net Emissions'
]
df_sector = df[sector_cols]
df_melted = df_sector.melt(id_vars=["Local Authority", "Calendar Year"],
                           var_name="Sector", value_name="Emissions")

latest_year = df["Calendar Year"].max()
df_latest = df_melted[df_melted["Calendar Year"] == latest_year]
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=df_latest[df_latest["Local Authority"].isin(la_list)],
            x="Local Authority", y="Emissions", hue="Sector", ax=ax)
ax.set_title(f"Sectoral Emissions by Local Authority ({latest_year})")
ax.set_ylabel("kt CO2e")
plt.xticks(rotation=45)
st.pyplot(fig)
