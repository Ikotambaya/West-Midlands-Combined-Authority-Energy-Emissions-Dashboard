import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import geopandas as gpd

st.set_page_config(layout="wide", page_title="West Midlands Emissions Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("westmidlands.xlsx", sheet_name="WML")
    df['Per Capita Emissions (tCO2e)'] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    df['Emissions per km2 (kt CO2e)'] = df["Grand Total"] / df["Area (km²)"]
    return df

df = load_data()
la_list = df['Local Authority'].unique().tolist()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
use_per_capita = st.sidebar.checkbox("Per Capita View", value=False)
selected_la = st.sidebar.selectbox("Select Local Authority", la_list)
scenario = st.sidebar.radio("Scenario", ["Business-as-Usual", "Accelerated"])
reduction_slider = st.sidebar.slider("Reduction in Transport by 2030 (%)", 0, 100, 40)

# --- Main Title ---
st.title("🌍 West Midlands Emissions Forecast & Trends Dashboard")

# --- Forecasting ---
@st.cache_data
def forecast_la(la, metric):
    la_df = df[df["Local Authority"] == la]
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(la_df["Calendar Year"].astype(str) + "-01-01"),
        "y": la_df[metric]
    })
    model = Prophet(yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=18, freq="Y")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].rename(columns={"yhat": "Forecast"})

metric = "Per Capita Emissions (tCO2e)" if use_per_capita else "Grand Total"
forecast = forecast_la(selected_la, metric)

st.subheader(f"📈 Emissions Forecast: {selected_la}")
fig1, ax1 = plt.subplots(figsize=(10, 4))
la_df = df[df["Local Authority"] == selected_la]
ax1.plot(pd.to_datetime(la_df["Calendar Year"].astype(str) + "-01-01"), la_df[metric], label="Historical", marker='o')
ax1.plot(forecast["ds"], forecast["Forecast"], label="Forecast", linestyle='--')
ax1.set_title("Forecast vs Historical")
ax1.set_ylabel(metric)
ax1.legend()
st.pyplot(fig1)

# --- Time Series Trends ---
st.subheader("📊 Emissions Trends by Local Authority")
fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.lineplot(data=df, x="Calendar Year", y="Grand Total", hue="Local Authority", marker="o", ax=ax2)
ax2.set_title("Total Emissions by Local Authority")
ax2.set_ylabel("kt CO2e")
st.pyplot(fig2)

# --- Per Capita Trend ---
st.subheader("👤 Per Capita Emissions by Local Authority")
fig3, ax3 = plt.subplots(figsize=(12, 5))
sns.lineplot(data=df, x="Calendar Year", y="Per Capita Emissions (tCO2e)", hue="Local Authority", ax=ax3)
ax3.set_title("Per Capita Emissions (tCO2e)")
st.pyplot(fig3)

# --- Emissions per km2 ---
st.subheader("🗺️ Emissions Intensity (kt CO2e / km²)")
fig4, ax4 = plt.subplots(figsize=(12, 5))
sns.lineplot(data=df, x="Calendar Year", y="Emissions per km2 (kt CO2e)", hue="Local Authority", ax=ax4)
ax4.set_title("Emissions per km²")
st.pyplot(fig4)

# --- Sectoral Breakdown (2022) ---
st.subheader("🏭 Sectoral Emissions by Local Authority (2022)")
sector_cols = ['Industry Total', 'Transport Total', 'Domestic Total',
               'Commercial Total', 'Public Sector Total', 'Agriculture Total', 
               'Waste Total', 'LULUCF Net Emissions']
df_2022 = df[df["Calendar Year"] == 2022]
df_melted = df_2022.melt(id_vars="Local Authority", value_vars=sector_cols, var_name="Sector", value_name="Emissions")
fig5, ax5 = plt.subplots(figsize=(14, 6))
sns.barplot(data=df_melted, x="Local Authority", y="Emissions", hue="Sector", ax=ax5)
ax5.set_title("Sectoral Emissions by LA (2022)")
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
st.pyplot(fig5)

# --- Choropleth ---
st.subheader("🗺️ Average Emissions per km² (Heatmap)")
@st.cache_data
def load_geo():
    return gpd.read_file("westmidlands_shapefile.geojson")

gdf = load_geo()
avg_emissions = df.groupby("Local Authority")["Emissions per km2 (kt CO2e)"].mean().reset_index()
merged = gdf.merge(avg_emissions, left_on="LA_Name", right_on="Local Authority", how="left")

fig6, ax6 = plt.subplots(1, 1, figsize=(8, 8))
merged.plot(column="Emissions per km2 (kt CO2e)", cmap="Reds", linewidth=0.8, ax=ax6, edgecolor='0.8', legend=True)
ax6.set_title("Average Emissions per km² by LA")
ax6.axis("off")
st.pyplot(fig6)

# --- Scenario Analysis ---
st.subheader("🔄 Scenario Impact Calculator")
base_2022 = df_2022[df_2022["Local Authority"].isin(la_list)]["Grand Total"].sum()
transport_2022 = df_2022["Transport Total"].sum()
reduction_amount = transport_2022 * (reduction_slider / 100)
new_total = base_2022 - reduction_amount

st.metric("2022 Total Emissions (kt CO2e)", f"{base_2022:,.0f}")
st.metric("Scenario-Adjusted Emissions (kt CO2e)", f"{new_total:,.0f}", delta=f"{-reduction_amount:,.0f}")

st.success("📢 Dashboard completed – ready for deployment or report integration.")
