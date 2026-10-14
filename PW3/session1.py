import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# 1. Load the data with pandas
df = pd.read_csv("Heart_disease_cleveland_new.csv")  #
print("Dataset shape (Rows, Columns):", df.shape)  #[cite: 1]
print("\nFirst rows:")
print(df.head())  #[cite: 1]

# Isolate the four continuous columns: age, chol, trestbps, thalach
continuous_vars = ["age", "chol", "trestbps", "thalach"]  #[cite: 1]
df_continuous = df[continuous_vars]

print("\nSummary statistics:")
print(df_continuous.describe())  #[cite: 1]

# 2. Visualise the distribution of each continuous variable (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, var in enumerate(continuous_vars):
  axes[i].hist(df[var], bins=20, edgecolor="black", alpha=0.7)  #[cite: 1]
  axes[i].set_title(f"Distribution: {var}")
  axes[i].set_xlabel(var)
  axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("distributions.png")
print("\nDistributions plot saved successfully as 'distributions.png'.")

# 3. Check whether they are normally distributed (Shapiro-Wilk test)
print("\n--- Shapiro-Wilk Normality Test ---")
for var in continuous_vars:
  stat, p_value = stats.shapiro(df[var].dropna())
  print(f"{var}: Statistic = {stat:.4f}, p-value = {p_value:.4e}")  #[cite: 1, 2]
  if p_value > 0.05:
    print(f"  -> {var} is approximately normally distributed (p > 0.05).\n")
  else:
    print(
        f"  -> {var} deviates from normal distribution (p < 0.05, hypothesis"
        " rejected).\n"
    )