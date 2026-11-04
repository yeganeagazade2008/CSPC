import os
import pandas as pd
from sklearn.datasets import load_wine

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

print("--- STEP B1: LOAD WINE DATASET ---")
# 1. Load dataset with pandas DataFrames
wine = load_wine(as_frame=True)
X = wine.data  # 178 rows x 13 measurements
cultivar = wine.target  # 0, 1, 2 labels (hidden for now)

# Print dataset description source summary
print("Dataset Source:")
print(
    "The data is the results of a chemical analysis of wines grown in the same"
    " region in Italy but derived from three different cultivars."
)
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {list(X.columns)}")

# 2. Print descriptive statistics
print("\nFeature Descriptive Statistics (X.describe()):")
print(X.describe())

# Identify measurements with largest and smallest ranges/standard deviations
print("\nAnalysis of ranges and standard deviations:")
print(
    "- Largest numerical range and std: 'proline' (ranges from hundreds up to"
    " 1680, high std)."
)
print(
    "- Smallest numerical range and std: 'nonflavanoid_phenols' or 'ash'"
    " (very small values and low std)."
)
from sklearn.decomposition import PCA
import numpy as np

print("\n--- STEP B2: PCA ON RAW DATA ---")
# 1. Fit PCA on raw data (without scaling)
pca_raw = PCA()
pca_raw.fit(X)

# 2. Explained variance ratio of the first principal component (PC1)
ev_ratio_raw = pca_raw.explained_variance_ratio_
print(f"Explained Variance Ratio of PC1 (Raw): {ev_ratio_raw[0]:.4f}")
print(f"Total Explained Variance (first 2 PCs): {np.sum(ev_ratio_raw[:2]):.4f}")

# 3. Sort PC1 loadings in descending order
pc1_loadings = pd.Series(
    pca_raw.components_[0], index=X.columns
).sort_values(ascending=False)
print("\nPC1 Loadings (Raw Data, Sorted):")
print(pc1_loadings)

print("\nAnalysis of PC1 (Raw Data):")
print(
    "- 'proline' completely dominates PC1 because its numerical range is much"
    " larger than other features."
)
print("- PCA subtracts the mean, but it does not rescale variances.")
print(
    "- Therefore, we cannot accept raw PC1 as 'the main chemical difference'"
    " between wines; it's just an artifact of measurement scales."
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("\n--- STEP B3: SCALE, THEN PCA AGAIN ---")
# 1. Scale the data so all features have mean=0 and std=1
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# 2. Fit PCA on scaled data
pca_scaled = PCA()
pca_scaled.fit(Z)

ev_ratio_scaled = pca_scaled.explained_variance_ratio_
cum_variance = np.cumsum(ev_ratio_scaled)

print("\nExplained Variance Ratios (Scaled PCA):")
for i, (ev, cum) in enumerate(zip(ev_ratio_scaled, cum_variance), 1):
    print(f"PC{i}: Individual = {ev:.4f}, Cumulative = {cum:.4f}")

# 3. Table of PC1 and PC2 loadings
loadings_df = pd.DataFrame(
    pca_scaled.components_[:2].T, 
    index=X.columns, 
    columns=["PC1", "PC2"]
)
print("\nPC1 and PC2 Loadings (Scaled Data):")
print(loadings_df)

# 4. Save Scree Plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(ev_ratio_scaled) + 1), ev_ratio_scaled, alpha=0.7, label='Individual')
plt.step(range(1, len(cum_variance) + 1), cum_variance, where='mid', label='Cumulative', color='red')
plt.axhline(y=0.9, color='blue', linestyle='--', label='90% Threshold')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot (Scaled Wine PCA)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('results/B3_scree_plot.png')
plt.close()
print("Scree plot saved to 'results/B3_scree_plot.png'")
from sklearn.cluster import KMeans

print("\n--- STEP B4 & B5: K-MEANS CLUSTERING ---")
# 1. Fit K-Means with k=3 on scaled data (Z)
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km.fit(Z)

# 2. Cluster sizes
cluster_counts = np.bincount(km.labels_)
print(f"Cluster sizes: {cluster_counts}")

# 3. Bring cluster centers back to real units
centers_real = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers_real, columns=X.columns)
print("\nCluster Centers in Real Units:")
print(centers_df)
print("\n--- STEP B6: THE REVEAL & AGREEMENT ---")
# Crosstab between true cultivars and K-Means clusters
ct = pd.crosstab(cultivar, km.labels_)
print("Crosstab (Cultivar vs Clusters):")
print(ct)

# Calculate agreement score
agreement = ct.max(axis=0).sum() / len(cultivar)
print(f"Agreement Score (Scaled K-Means): {agreement:.4f}")

# Save agreement result to CSV for safety
agreement_df = pd.DataFrame({"Model": ["Scaled K-Means"], "Agreement": [agreement]})
agreement_df.to_csv("results/B_agreement.csv", index=False)
print("Agreement saved to 'results/B_agreement.csv'")