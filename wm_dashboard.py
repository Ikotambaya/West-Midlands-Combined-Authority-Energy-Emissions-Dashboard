import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("your_data.csv")  # Replace with your dataset
la_list = ['Birmingham', 'Coventry', 'Dudley', 'Sandwell', 'Solihull', 'Walsall', 'Wolverhampton']
wm_df = df[df['Local Authority'].isin(la_list)]
grouped = wm_df.groupby('Calendar Year').sum().reset_index()

st.title("🌍 WM2041 Emissions Scenario Dashboard")
st.markdown("Adjust sectoral reductions to explore emissions pathways against the WM2041 net-zero target.")

# Sidebar: Sliders for sectoral reductions
st.sidebar.header("📉 Reduction Assumptions")

sectors = {
    'Transport Total': '🚗 Transport',
    'Industry Total': '🏭 Industry',
    'Domestic Total': '🏠 Domestic',
    'Commercial Total': '🏢 Commercial',
    'Public Sector Total': '🏛 Public Sector',
    'Agriculture Total': '🌾 Agriculture',
    'Waste Total': '🗑 Waste'
}

reduction_settings = {}
for key, label in sectors.items():
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        pct = st.slider(f"{label} Reduction %", 0, 100, 30 if "Transport" in key else 10, key=key)
    with col2:
        year = st.number_input(f"Start Year", min_value=2024, max_value=2040, value=2030, key=f"{key}_year")
    reduction_settings[key] = (pct / 100, year)

# Scenario Simulation
scenario = grouped.copy()

for year in scenario['Calendar Year']:
    idx = scenario['Calendar Year'] == year
    for sector, (reduction_pct, start_year) in reduction_settings.items():
        if year >= start_year:
            scenario.loc[idx, sector] *= (1 - reduction_pct)

    scenario.loc[idx, 'Grand Total'] = scenario.loc[idx, [
        'Industry Total', 'Commercial Total', 'Public Sector Total',
        'Domestic Total', 'Transport Total', 'Agriculture Total', 'Waste Total'
    ]].sum(axis=1)

# --- WM2041 Target Line ---
baseline_2016 = grouped[grouped['Calendar Year'] == 2016]['Grand Total'].values[0]
years = np.arange(2016, 2042)
target_vals = []

target_2026 = baseline_2016 * (1 - 0.33)
for y in years:
    if y <= 2026:
        v = baseline_2016 - (baseline_2016 - target_2026) * (y - 2016) / (2026 - 2016)
    else:
        v = target_2026 * (1 - (y - 2026) / (2041 - 2026))
    target_vals.append(v)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(grouped['Calendar Year'], grouped['Grand Total'], label='Original Emissions', linewidth=2)
ax.plot(scenario['Calendar Year'], scenario['Grand Total'], label='Scenario Emissions', linestyle='--', linewidth=2)
ax.plot(years, target_vals, 'k--', label='WM2041 Target')

ax.set_title("Projected Emissions with Custom Reductions")
ax.set_xlabel("Year")
ax.set_ylabel("Total Emissions (kt CO2e)")
ax.grid(True)
ax.legend()
st.pyplot(fig)
