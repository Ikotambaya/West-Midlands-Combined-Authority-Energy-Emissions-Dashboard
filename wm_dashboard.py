import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filter for selected LAs and use yearly group
la_list = ['Birmingham', 'Coventry', 'Dudley', 'Sandwell', 'Solihull', 'Walsall', 'Wolverhampton']
wm_df = df[df['Local Authority'].isin(la_list)]

grouped = wm_df.groupby('Calendar Year').sum().reset_index()

# Store original totals for reference
original = grouped.copy()

# Clone to apply scenario
scenario = grouped.copy()

# Apply reductions (example reductions over time)
for year in scenario['Calendar Year']:
    idx = scenario['Calendar Year'] == year
    if year >= 2030:
        scenario.loc[idx, 'Transport Total'] *= 0.6  # 40% reduction
    if year >= 2035:
        scenario.loc[idx, 'Industry Total'] *= 0.6
    if year >= 2030:
        scenario.loc[idx, 'Domestic Total'] *= 0.7

    # Recalculate Grand Total
    scenario.loc[idx, 'Grand Total'] = (
        scenario.loc[idx, 'Industry Total'].values +
        scenario.loc[idx, 'Commercial Total'].values +
        scenario.loc[idx, 'Public Sector Total'].values +
        scenario.loc[idx, 'Domestic Total'].values +
        scenario.loc[idx, 'Transport Total'].values +
        scenario.loc[idx, 'Agriculture Total'].values +
        scenario.loc[idx, 'Waste Total'].values
    )

# Plot original vs scenario
plt.figure(figsize=(12, 6))
plt.plot(original['Calendar Year'], original['Grand Total'], label='Original Emissions', linewidth=2)
plt.plot(scenario['Calendar Year'], scenario['Grand Total'], label='Scenario Emissions', linewidth=2, linestyle='--')

# Optional: add WM2041 Target line
baseline_2016 = original[original['Calendar Year'] == 2016]['Grand Total'].values[0]
years = np.arange(2016, 2042)
target_vals = []

target_2026 = baseline_2016 * (1 - 0.33)
for y in years:
    if y <= 2026:
        v = baseline_2016 - (baseline_2016 - target_2026) * (y - 2016) / (2026 - 2016)
    else:
        v = target_2026 * (1 - (y - 2026) / (2041 - 2026))
    target_vals.append(v)
plt.plot(years, target_vals, 'k--', label='WM2041 Target')

plt.title('Scenario Analysis: Emissions Reduction by Sector')
plt.xlabel('Year')
plt.ylabel('Total Emissions (kt CO2e)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
