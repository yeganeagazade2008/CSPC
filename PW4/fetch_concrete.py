import os
from ucimlrepo import fetch_ucirepo

print("1. Fetching the Concrete Compressive Strength dataset from UCI...")
concrete = fetch_ucirepo(id=165)

# Extract features and target
X = concrete.data.features.copy()
y = concrete.data.targets.iloc[:, 0].rename("strength")

# Print metadata and variables table for README/inspection
print("\n--- METADATA ---")
print(concrete.metadata)

print("\n--- VARIABLES TABLE ---")
print(concrete.variables)

# Rename columns by position to match short, standard names
X.columns = [
    "cement",
    "slag",
    "ash",
    "water",
    "superplastic",
    "coarse_agg",
    "fine_agg",
    "age",
]

# Create 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Combine features and target into a single DataFrame and save as CSV
df = X.assign(strength=y)
df.to_csv("data/concrete.csv", index=False)

print("\nSuccess! 'data/concrete.csv' has been created and saved.")