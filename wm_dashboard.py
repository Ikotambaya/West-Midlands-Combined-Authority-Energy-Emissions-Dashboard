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

# --- Load Data from GitHub ---
@st.cache_data
def load_data():
    # Load data directly from GitHub (replace with your raw GitHub URL)
    data_url = "https://github.com/yourusername/yourrepo/raw/main/westmidlands.xlsx"
    
    try:
        df = pd.read_excel(data_url, sheet_name="WML")
    except:
        # Fallback to local file if GitHub load fails
        df = pd.read_excel("westmidlands.xlsx", sheet_name="WML")
    
    # Calculate metrics
    df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    df["Emissions per km2 (kt CO2e)"] = df["Grand Total"] / df["Area (km2)"]
    
    # Coordinates for West Midlands LAs
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

# Load the data
df = load_data()

# --- Constants ---
la_list = df["Local Authority"].unique().tolist()
years = np.arange(2016, 2042)
dates = pd.to_datetime(years.astype(str) + "-01-01"

# --- Sidebar Controls ---
st.sidebar.title("🔧 Dashboard Controls")

# Visualization selection
selected_vis = st.sidebar.multiselect(
    "Select visualizations", 
    [
        "Forecast vs Target",
        "Total Emissions Trend",
        "Per Capita Emissions",
        "Emissions per km²",
        "Average Emissions Heatmap",
        "Sectoral Emissions",
        "Interactive Map"
    ],
    default=["Forecast vs Target", "Interactive Map"]
)

# Metric selection
metric_options = {
    "Grand Total": "Total Emissions (kt CO2e)",
    "Per Capita Emissions (tCO2e)": "Per Capita Emissions (tCO2e per person)",
    "Emissions per km2 (kt CO2e)": "Emissions per km² (kt CO2e)"
}
selected_metric = st.sidebar.selectbox(
    "Primary Metric", 
    options=list(metric_options.keys()),
    format_func=lambda x: metric_options[x]
)

scenario = st.sidebar.selectbox(
    "Emission Scenario", 
    ["Business-as-Usual", "Accelerated", "Net Zero by 2041"]
)

# --- Main Dashboard ---
st.title("🌍 West Midlands Emissions Dashboard")
st.markdown("""
    *Tracking progress toward the WM2041 Net Zero target*  
    [![GitHub](https://img.shields.io/badge/View%20on%20GitHub-blue?logo=github)](https://github.com/yourusername/yourrepo)
""")

# --- Interactive Map Visualization ---
if "Interactive Map" in selected_vis:
    st.header("🗺️ Interactive Emissions Map")
    
    # Prepare map data
    map_year = st.sidebar.selectbox(
        "Map Year",
        sorted(df["Calendar Year"].unique(), reverse=True)
    )
    
    map_df = df[df["Calendar Year"] == map_year].groupby("Local Authority").agg({
        "Grand Total": "sum",
        "Per Capita Emissions (tCO2e)": "mean",
        "lat": "first",
        "lon": "first",
        "Area (km2)": "first"
    }).reset_index()
    
    # Create color scale
    colormap = LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=map_df["Per Capita Emissions (tCO2e)"].min(),
        vmax=map_df["Per Capita Emissions (tCO2e)"].max()
    )
    
    # Create map
    m = folium.Map(
        location=[52.4862, -1.8904],  # Centered on West Midlands
        zoom_start=9,
        tiles="cartodbpositron"
    )
    
    # Add markers
    for idx, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["Grand Total"] / 100,
            popup=f"""
            <b>{row['Local Authority']}</b><br>
            Total: {row['Grand Total']:,.0f} kt CO2e<br>
            Per Capita: {row['Per Capita Emissions (tCO2e)']:,.1f} tCO2e<br>
            Area: {row['Area (km2)']:,.0f} km²
            """,
            color=colormap(row["Per Capita Emissions (tCO2e)"]),
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    # Add color scale
    colormap.caption = "Per Capita Emissions (tCO2e)"
    colormap.add_to(m)
    
    # Display map
    folium_static(m, width=1000, height=600)

# [Include your other visualizations here...]

# --- Footer ---
st.markdown("---")
st.caption("""
    Built with Streamlit | Data Source: West Midlands Combined Authority  
    [Report Issues](https://github.com/yourusername/yourrepo/issues) | 
    [View Source Code](https://github.com/yourusername/yourrepo)
""")
