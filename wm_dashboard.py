import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap

# --- Configuration ---
st.set_page_config(
    page_title="West Midlands Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("westmidlands.xlsx", sheet_name="WML")
    df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    df["Emissions per km2 (kt CO2e)"] = df["Grand Total"] / df["Area (km2)"]
    
    # Coordinates for West Midlands local authorities
    la_coords = {
        "Birmingham": [52.4862, -1.8904],
        "Coventry": [52.4068, -1.5197],
        "Dudley": [52.5087, -2.0873],
        "Sandwell": [52.5069, -2.0016],
        "Solihull": [52.4118, -1.7776],
        "Walsall": [52.5862, -1.9829],
        "Wolverhampton": [52.5873, -2.1294]
    }
    
    df["lat"] = df["Local Authority"].map(lambda x: la_coords.get(x, [0, 0])[0])
    df["lon"] = df["Local Authority"].map(lambda x: la_coords.get(x, [0, 0])[1])
    
    return df

df = load_data()

# --- Constants ---
la_list = df["Local Authority"].unique().tolist()
years = np.arange(2016, 2042)
dates = pd.to_datetime(years.astype(str) + "-01-01")  # Fixed the missing parenthesis here

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
    "Sectoral Emissions",
    "Interactive Map"  # Added the new map visualization option
], default=["Forecast vs Target", "Interactive Map"])

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

# --- Interactive Map Visualization ---
if "Interactive Map" in selected_vis:
    st.header("🗺️ Interactive Emissions Map")
    
    # Prepare data for the map
    map_year = st.sidebar.selectbox(
        "Select year for map",
        sorted(df["Calendar Year"].unique(), reverse=True)
    
    map_df = df[df["Calendar Year"] == map_year].groupby("Local Authority").agg({
        "Grand Total": "sum",
        "Per Capita Emissions (tCO2e)": "mean",
        "lat": "first",
        "lon": "first",
        "Area (km2)": "first",
        "Population ('000s, mid-year estimate)": "sum"
    }).reset_index()
    
    # Create color scale
    max_per_capita = map_df["Per Capita Emissions (tCO2e)"].max()
    colormap = LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=0,
        vmax=max_per_capita
    )
    
    # Create map centered on West Midlands
    m = folium.Map(
        location=[52.4862, -1.8904],  # Birmingham coordinates
        zoom_start=9,
        tiles="cartodbpositron"
    )
    
    # Add markers for each local authority
    for idx, row in map_df.iterrows():
        popup_text = f"""
        <b>{row['Local Authority']}</b><br>
        Total Emissions: {row['Grand Total']:,.0f} kt CO2e<br>
        Per Capita: {row['Per Capita Emissions (tCO2e)']:,.1f} tCO2e<br>
        Area: {row['Area (km2)']:,.0f} km²<br>
        Population: {row["Population ('000s, mid-year estimate)"]:,.0f}k
        """
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["Grand Total"] / 100,  # Scale circle size
            popup=popup_text,
            color=colormap(row["Per Capita Emissions (tCO2e)"]),
            fill=True,
            fill_color=colormap(row["Per Capita Emissions (tCO2e)"]),
            fill_opacity=0.7
        ).add_to(m)
    
    # Add color scale to map
    colormap.caption = "Per Capita Emissions (tCO2e)"
    colormap.add_to(m)
    
    # Display the map
    folium_static(m, width=1000, height=600)

# [Rest of your existing visualizations...]

# --- Footer ---
st.markdown("---")
st.caption("Built for West Midlands Net Zero Strategy — Research Analyst Dashboard")
