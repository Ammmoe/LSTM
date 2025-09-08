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
df["time_s"] = df["Timpstemp"].astype(np.float64) / 1e6

# Build target times (0.1s intervals from start to end)
start_time = float(df["time_s"].iloc[0])
end_time = float(df["time_s"].iloc[-1])
target_times = np.arange(start_time, end_time, 0.1, dtype=np.float64)

# Ensure numpy array is float64
time_array = df["time_s"].to_numpy(dtype=np.float64)

# Use searchsorted
indices = np.searchsorted(time_array, target_times)

# Correct indices if previous row is closer
for j, idx in enumerate(indices):
    if idx > 0 and abs(time_array[idx] - target_times[j]) > abs(
        time_array[idx - 1] - target_times[j]
    ):
        indices[j] = idx - 1

# Select rows
df_downsampled = df.iloc[indices].reset_index(drop=True)

# Save CSV
df_downsampled.to_csv("zurich_flights_downsampled_2.csv", index=False)
print(f"Downsampled data saved with {len(df_downsampled)} rows")
