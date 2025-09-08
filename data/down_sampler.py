"""
this script downsamples a CSV file by taking every 3rd row and 
saves the result to a new CSV file.
"""

import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("zurich_flights.csv")

# # Take every 3rd row
# df_downsampled = df.iloc[::3].reset_index(drop=True)

# # Save to CSV
# df_downsampled.to_csv("zurich_flights_downsampled.csv", index=False)

# print("Downsampled data saved as zurich_flights_downsampled.csv")

# Convert timestamp from µs to seconds
df["time_s"] = df["Timpstemp"] / 1e6

# Build target times (0.1s intervals from start to end)
start_time = df["time_s"].iloc[0]
end_time = df["time_s"].iloc[-1]
target_times = np.arange(start_time, end_time, 0.1)

# For each target time, pick the closest row
selected_rows = []
i = 0  # pointer for raw data
n = len(df)

for t in target_times:
    # Move pointer forward until we pass target time
    while i < n - 1 and abs(df["time_s"].iloc[i+1] - t) < abs(df["time_s"].iloc[i] - t):
        i += 1
    selected_rows.append(df.iloc[i])

# Create downsampled DataFrame
df_downsampled = pd.DataFrame(selected_rows).reset_index(drop=True)

# Save to CSV
df_downsampled.to_csv("zurich_flights_downsampled_2.csv", index=False)

print(f"Downsampled data saved as zurich_flights_downsampled_2.csv with {len(df_downsampled)} rows")
