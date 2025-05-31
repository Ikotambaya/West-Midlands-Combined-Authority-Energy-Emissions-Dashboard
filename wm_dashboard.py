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
    
    # Calculate metrics
    df["Per Capita Emissions (tCO2e)"] = df["Grand Total"] / df["Population ('000s, mid-year estimate)"]
    df["Emissions per km2 (kt CO2e)"] = df["Grand Total"] / df["Area (km2)"]
    
    # Load geographic data (you'll need to provide this or use a different approach)
    # This is a placeholder - you should replace with actual coordinates for each LA
    la_coords = {
        "Birmingham": [52.4862, -1.8904],
        "Coventry": [52.4068, -1.5197],
        "Dudley": [52.5087, -2.0873],
        "Sandwell": [52.5069, -2.0016],
        "Solihull": [52.4118, -1.7776],
        "Walsall": [52.5862, -1.9829],
        "Wolverhampton": [52.5873, -2.1294]
    }
    
    df["lat"] = df["Local Authority"].map(lambda x: la_coords.get(x, [0, 0])[0]
    df["lon"] = df["Local Authority"].map(lambda x: la_coords.get(x, [0, 0])[1]
    
    return df

df = load_data()

# --- Constants ---
la_list = df["Local Authority"].unique().tolist()
years = np.arange(2016, 2042)
dates = pd.to_datetime(years.astype(str) + "-01-01")

# --- Sector Definitions ---
sector_cols = {
    "Industry": ['Industry Total'],
    "Transport": ['Transport Total'],
    "Domestic": ['Domestic Total'],
    "Commercial": ['Commercial Total'],
    "Public Sector": ['Public Sector Total'],
    "Agriculture": ['Agriculture Total'],
    "Waste": ['Waste Total'],
    "LULUCF": ['LULUCF Net Emissions']
}

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
        "Interactive Map",
        "Emissions Reduction Progress",
        "Sector Contribution Analysis"
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

# Year filter for some visualizations
min_year, max_year = int(df["Calendar Year"].min()), int(df["Calendar Year"].max())
selected_year = st.sidebar.slider(
    "Focus Year", 
    min_value=min_year, 
    max_value=max_year, 
    value=max_year
)

# --- Dashboard Header ---
st.title("🌍 West Midlands Emissions Intelligence Dashboard")
st.markdown("""
    *Tracking progress toward the WM2041 Net Zero target across local authorities*
""")

# --- Key Metrics ---
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_emissions = df[df["Calendar Year"] == selected_year]["Grand Total"].sum()
    st.metric("Total Emissions (2022)", f"{total_emissions:,.0f} kt CO2e")

with col2:
    pop = df[df["Calendar Year"] == selected_year]["Population ('000s, mid-year estimate)"].sum()
    st.metric("Total Population", f"{pop:,.0f} thousand")

with col3:
    avg_per_capita = df[df["Calendar Year"] == selected_year]["Per Capita Emissions (tCO2e)"].mean()
    st.metric("Avg Per Capita Emissions", f"{avg_per_capita:,.1f} tCO2e")

with col4:
    baseline = df[df["Calendar Year"] == 2016]["Grand Total"].sum()
    current = df[df["Calendar Year"] == selected_year]["Grand Total"].sum()
    reduction = ((baseline - current) / baseline) * 100
    st.metric("Reduction since 2016", f"{reduction:,.1f}%")

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
    
    with st.expander("About this visualization"):
        st.write("""
        This chart compares the projected emissions trajectory (based on historical trends) 
        with the WM2041 target pathway. The shaded area represents the gap between current 
        trajectory and the target.
        """)
    
    forecast_df = pd.DataFrame({"ds": pd.date_range(start="2005", end="2042", freq="Y")})
    for la in la_list:
        try:
            forecast = forecast_la(la, selected_metric)
            forecast_df = forecast_df.merge(forecast, on="ds", how="left")
        except Exception as e:
            st.warning(f"Forecast failed for {la}: {e}")

    forecast_df["Total Forecast"] = forecast_df[la_list].sum(axis=1)

    # --- WM2041 Target Calculation ---
    baseline_val = df[(df["Calendar Year"] == 2016) & (df["Local Authority"].isin(la_list))][selected_metric].sum()
    target_vals = []
    for year in years:
        if scenario == "Business-as-Usual":
            target = baseline_val
        elif scenario == "Accelerated":
            if year <= 2026:
                target = baseline_val - (baseline_val * 0.33 * (year - 2016) / 10)
            else:
                target = (baseline_val * 0.67) * (1 - (year - 2026) / 15)
        else:  # Net Zero by 2041
            target = baseline_val * (1 - (year - 2016) / (2041 - 2016))
        target_vals.append(target)

    target_df = pd.DataFrame({"ds": dates, "WM2041 Target": target_vals})
    merged_df = pd.merge(forecast_df, target_df, on="ds", how="left")

    # --- Plot Forecast vs Target ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged_df["ds"], merged_df["Total Forecast"], label="Forecasted Total", linewidth=2)
    ax.plot(merged_df["ds"], merged_df["WM2041 Target"], "k--", label=f"{scenario} Target", linewidth=2)
    ax.fill_between(
        merged_df["ds"],
        merged_df["Total Forecast"],
        merged_df["WM2041 Target"],
        where=(merged_df["Total Forecast"] > merged_df["WM2041 Target"]),
        color="red", alpha=0.2, label="Gap to Target"
    )
    ax.set_title(f"{metric_options[selected_metric]} Forecast vs. WM2041 Target")
    ax.set_ylabel(metric_options[selected_metric])
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Interactive Map Visualization ---
if "Interactive Map" in selected_vis:
    st.header("🗺️ Interactive Emissions Map")
    
    with st.expander("About this map"):
        st.write("""
        This interactive map shows emissions data by local authority. 
        - Circle size represents total emissions
        - Color intensity represents per capita emissions
        - Click on markers for detailed information
        """)
    
    # Prepare data for the map
    map_df = df[df["Calendar Year"] == selected_year].groupby("Local Authority").agg({
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
        Population: {row['Population ('000s, mid-year estimate)']:,.0f}k
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

# --- Emissions Trend Line Plot ---
if "Total Emissions Trend" in selected_vis:
    st.header("📊 Emissions Trend by Local Authority")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df, 
            x='Calendar Year', 
            y=selected_metric, 
            hue='Local Authority', 
            marker='o',
            linewidth=2
        )
        plt.title(f'{metric_options[selected_metric]} by Local Authority (2005–2022)')
        plt.ylabel(metric_options[selected_metric])
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
    
    with col2:
        st.markdown("**Top 3 Local Authorities**")
        latest_data = df[df["Calendar Year"] == max_year]
        top3 = latest_data.sort_values(by=selected_metric, ascending=False).head(3)
        for idx, row in top3.iterrows():
            st.metric(
                row["Local Authority"],
                f"{row[selected_metric]:,.1f}",
                delta=f"{((row[selected_metric] - row[selected_metric])/row[selected_metric]*100):.1f}%"
            )

# --- Sectoral Emissions Analysis ---
if "Sectoral Emissions" in selected_vis:
    st.header("🏭 Sectoral Emissions Breakdown")
    
    tab1, tab2 = st.tabs(["By Local Authority", "Regional Trends"])
    
    with tab1:
        st.subheader(f"Sectoral Emissions by Local Authority ({selected_year})")
        sector_cols = [
            'Industry Total', 'Transport Total', 'Domestic Total',
            'Commercial Total', 'Public Sector Total', 'Agriculture Total',
            'Waste Total', 'LULUCF Net Emissions'
        ]
        df_sector = df[df["Calendar Year"] == selected_year][["Local Authority"] + sector_cols]
        df_melted = df_sector.melt(id_vars="Local Authority", var_name="Sector", value_name="Emissions")
        
        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=df_melted, 
            x='Local Authority', 
            y='Emissions', 
            hue='Sector',
            palette="viridis"
        )
        plt.title(f"Sectoral Emissions Breakdown ({selected_year})")
        plt.xticks(rotation=45)
        plt.ylabel("kt CO2e")
        plt.tight_layout()
        st.pyplot(plt)
    
    with tab2:
        st.subheader("Sector Contribution Over Time")
        
        # Calculate regional totals by sector over time
        sector_trends = df.groupby("Calendar Year")[sector_cols].sum().reset_index()
        sector_trends_melted = sector_trends.melt(
            id_vars="Calendar Year", 
            var_name="Sector", 
            value_name="Emissions"
        )
        
        plt.figure(figsize=(14, 6))
        sns.lineplot(
            data=sector_trends_melted,
            x="Calendar Year",
            y="Emissions",
            hue="Sector",
            style="Sector",
            markers=True,
            dashes=False,
            linewidth=2
        )
        plt.title("Regional Sectoral Emissions Trends (2005-2022)")
        plt.ylabel("kt CO2e")
        plt.grid(True)
        st.pyplot(plt)

# --- Emissions Reduction Progress ---
if "Emissions Reduction Progress" in selected_vis:
    st.header("📉 Emissions Reduction Progress")
    
    # Calculate progress for each LA
    progress_df = df.pivot_table(
        index="Local Authority",
        columns="Calendar Year",
        values="Grand Total"
    ).reset_index()
    
    # Calculate reduction percentages
    for year in range(2017, max_year + 1):
        progress_df[f"Reduction_since_2016_{year}"] = (
            (progress_df[2016] - progress_df[year]) / progress_df[2016] * 100
        )
    
    # Melt for visualization
    progress_melted = progress_df.melt(
        id_vars="Local Authority",
        value_vars=[f"Reduction_since_2016_{y}" for y in range(2017, max_year + 1)],
        var_name="Year",
        value_name="Reduction %"
    )
    progress_melted["Year"] = progress_melted["Year"].str.extract(r'(\d+)').astype(int)
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=progress_melted,
        x="Year",
        y="Reduction %",
        hue="Local Authority",
        marker="o"
    )
    plt.axhline(y=33, color='r', linestyle='--', label="2026 Target (33%)")
    plt.axhline(y=100, color='g', linestyle='--', label="Net Zero (100%)")
    plt.title("Emissions Reduction Progress Since 2016")
    plt.ylabel("Reduction (%)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# --- Sector Contribution Analysis ---
if "Sector Contribution Analysis" in selected_vis:
    st.header("🔍 Sector Contribution Analysis")
    
    # Calculate sector contributions
    latest_sectors = df[df["Calendar Year"] == max_year][list(sector_cols.keys())].sum()
    latest_sectors = latest_sectors.reset_index()
    latest_sectors.columns = ["Sector", "Emissions"]
    latest_sectors["Percentage"] = latest_sectors["Emissions"] / latest_sectors["Emissions"].sum() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Sector Contribution ({max_year})")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            latest_sectors["Emissions"],
            labels=latest_sectors["Sector"],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("viridis", len(latest_sectors))
        )
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Change Since 2016")
        # Calculate change since 2016
        sectors_2016 = df[df["Calendar Year"] == 2016][list(sector_cols.keys())].sum()
        change = ((latest_sectors.set_index("Sector")["Emissions"] - sectors_2016) / sectors_2016) * 100
        
        plt.figure(figsize=(8, 6))
        change.sort_values().plot(kind="barh", color="steelblue")
        plt.title("Percentage Change in Sector Emissions (2016-2022)")
        plt.xlabel("Percentage Change")
        plt.grid(True)
        st.pyplot(plt)

# --- Footer ---
st.markdown("---")
st.caption("""
    Built for West Midlands Net Zero Strategy — Research Analyst Dashboard  
    Data Source: West Midlands Combined Authority | Last Updated: 2023
""")
