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
    df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    df["Emissions per km2 (kt CO2e)"] = df["Grand Total"] / df["Area (km2)"]
    return df

df = load_data()

# --- Constants ---
la_list = df["Local Authority"].unique().tolist()
years = np.arange(2016, 2042)
dates = pd.to_datetime(years.astype(str) + "-01-01")

# --- Sidebar Controls ---
st.sidebar.title("🔧 Controls")
use_per_capita = st.sidebar.checkbox("Per Capita Emissions", value=False)
scenario = st.sidebar.selectbox("Emission Scenario", ["Business-as-Usual", "Accelerated"])
selected_vis = st.sidebar.multiselect("Select visualizations", [
    "Forecast vs Target",
    "Total Emissions Trend",
    "Per Capita Emissions",
    "Emissions per km²",
    "Average Emissions Heatmap",
    "Sectoral Emissions"
], default=["Forecast vs Target"])

# --- Title ---
st.title("🌍 West Midlands Emissions Intelligence Dashboard")

metric = "Grand Total"
metric_label = "Emissions (kt CO2e)"
if use_per_capita:
    metric = "Per Capita Emissions (tCO2e)"
    metric_label = "Emissions (tCO2e per person)"

# --- Forecast Function ---
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

# --- Forecasting All LAs ---
if "Forecast vs Target" in selected_vis:
    st.header("📈 Forecast vs WM2041 Target")

    forecast_df = pd.DataFrame({"ds": pd.date_range(start="2005", end="2042", freq="Y")})
    for la in la_list:
        try:
            forecast = forecast_la(la, metric)
            forecast_df = forecast_df.merge(forecast, on="ds", how="left")
        except Exception as e:
            st.warning(f"Forecast failed for {la}: {e}")

    forecast_df["Total Forecast"] = forecast_df[la_list].sum(axis=1)

    # --- WM2041 Target ---
    baseline_val = df[(df["Calendar Year"] == 2016) & (df["Local Authority"].isin(la_list))][metric].sum()
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
    merged_df = pd.merge(forecast_df, target_df, on="ds", how="left")

    # --- Plot Forecast vs Target ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged_df["ds"], merged_df["Total Forecast"], label="Forecasted Total", linewidth=2)
    ax.plot(merged_df["ds"], merged_df["WM2041 Target"], "k--", label=f"{scenario} Target", linewidth=2)
    ax.fill_between(merged_df["ds"],
                    merged_df["Total Forecast"],
                    merged_df["WM2041 Target"],
                    color="red", alpha=0.1, label="Gap")
    ax.set_title("Emissions Forecast vs. WM2041 Target")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Emissions Trend Line Plot ---
if "Total Emissions Trend" in selected_vis:
    st.header("📊 Emissions Trend by Local Authority")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Calendar Year', y='Grand Total', hue='Local Authority', marker='o')
    plt.title('Total Emissions by Local Authority (2005–2022)')
    plt.ylabel("kt CO2e")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# --- Per Capita Emissions Trend ---
if "Per Capita Emissions" in selected_vis:
    st.header("👥 Per Capita Emissions by Local Authority")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Calendar Year', y='Per Capita Emissions (tCO2e)', hue='Local Authority')
    plt.title('Per Capita Emissions (2005–2022)')
    plt.ylabel('tCO2e per person')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# --- Emissions per km² ---
if "Emissions per km²" in selected_vis:
    st.header("📏 Emissions per km² by Local Authority")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Calendar Year', y='Emissions per km2 (kt CO2e)', hue='Local Authority')
    plt.title('Emissions per km² (2005–2022)')
    plt.ylabel('kt CO2e per km²')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# --- Heatmap of Average Emissions per km² ---
if "Average Emissions Heatmap" in selected_vis:
    st.header("🌡️ Average Emissions per km² (2005–2022)")
    pivot = df.pivot_table(index='Local Authority', values='Emissions per km2 (kt CO2e)', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, cmap='Reds', ax=ax)
    ax.set_title('Average Emissions per km² by LA')
    st.pyplot(fig)

# --- Sectoral Emissions Barplot ---
if "Sectoral Emissions" in selected_vis:
    st.header("🏭 Sectoral Emissions by Local Authority (2022)")
    sector_cols = [
        'Industry Total', 'Transport Total', 'Domestic Total',
        'Commercial Total', 'Public Sector Total', 'Agriculture Total',
        'Waste Total', 'LULUCF Net Emissions'
    ]
    df_sector = df[df["Calendar Year"] == 2022][["Local Authority"] + sector_cols]
    df_melted = df_sector.melt(id_vars="Local Authority", var_name="Sector", value_name="Emissions")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_melted, x='Local Authority', y='Emissions', hue='Sector')
    plt.title("Sectoral Emissions Breakdown (2022)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# --- Footer ---
st.caption("Built for West Midlands Net Zero Strategy — Research Analyst Dashboard")
