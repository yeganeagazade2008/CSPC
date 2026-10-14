import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# ==========================================
# PART 1: HEART DISEASE DATASET (SESSION 2)
# ==========================================
df_heart = pd.read_csv("Heart_disease_cleveland_new.csv")

# Task 4: Do heart-disease patients reach a different maximum heart rate (thalach) than healthy ones?
print("--- Task 4: Thalach Comparison by Heart Disease Status ---")
disease_group = df_heart[df_heart["target"] == 1][
    "thalach"
]  # Disease group (target=1)[cite: 1, 2]
healthy_group = df_heart[df_heart["target"] == 0][
    "thalach"
]  # Healthy group (target=0)[cite: 1, 2]

# Independent t-test (or Mann-Whitney U if non-normal, but let's run t-test for demonstration)
t_stat, p_val = stats.ttest_ind(disease_group, healthy_group)
print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")

# Mean and uncertainty (Standard Error of the Mean - SEM)
mean_disease = disease_group.mean()
sem_disease = stats.sem(disease_group)

mean_healthy = healthy_group.mean()
sem_healthy = stats.sem(healthy_group)

print(
    f"Disease Group - Mean Thalach: {mean_disease:.2f} ± {sem_disease:.2f}"
    ""
)
print(
    f"Healthy Group - Mean Thalach: {mean_healthy:.2f} ± {sem_healthy:.2f}"
    ""
)

# Plotting means with uncertainty
plt.figure(figsize=(6, 5))
groups = ["Disease", "Healthy"]
means = [mean_disease, mean_healthy]
sems = [sem_disease, sem_healthy]

plt.bar(groups, means, yerr=sems, capsize=5, color=["red", "blue"], alpha=0.7)
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Thalach Comparison with Uncertainty (SEM)")
plt.savefig("thalach_comparison.png")
plt.close()
print("Plot saved as 'thalach_comparison.png'.\n")


# Task 5: Do age and thalach move together?
print("--- Task 5: Age and Thalach Correlation ---")
correlation, corr_p_val = stats.pearsonr(df_heart["age"], df_heart["thalach"])
print(
    f"Correlation coefficient (r): {correlation:.4f}, p-value:"
    f" {corr_p_val:.4e}"
)

plt.figure(figsize=(6, 5))
plt.scatter(
    df_heart["age"], df_heart["thalach"], alpha=0.7, edgecolor="k"
)  #
plt.xlabel("Age")
plt.ylabel("Thalach")
plt.title("Age vs Max Heart Rate (Thalach)")
plt.savefig("age_thalach_corr.png")
plt.close()
print("Correlation plot saved as 'age_thalach_corr.png'.\n")


# ==========================================
# PART 2: CHEMICAL-EXPOSURE MYSTERY (DATASET 2)
# ==========================================
df_chem = pd.read_csv("chemicals_cancer.csv")  #

# Task 6: Naive analysis (Association between chemicals and malignancy)
print("--- Task 6: Naive Analysis (Chemicals vs Malignancy) ---")
corr_benzene, _ = stats.pearsonr(df_chem["benzene"], df_chem["malignancy"])
corr_cadmium, _ = stats.pearsonr(df_chem["cadmium"], df_chem["malignancy"])

print(f"Naive Correlation - Benzene vs Malignancy: {corr_benzene:.4f}")
print(f"Naive Correlation - Cadmium vs Malignancy: {corr_cadmium:.4f}")
print(
    "Notice: One chemical looks far more strongly associated than the other"
    "!"
)

# Task 7: Look again (Accounting for confounding variable - pollution_index)
print(
    "\n--- Task 7: Controlling for Confounding Variable (pollution_index) ---"
)
# Restrict data to patients who are similar on pollution_index (e.g., between 40 and 60)[cite: 2]
restricted_df = df_chem[
    (df_chem["pollution_index"] > 40) & (df_chem["pollution_index"] < 60)
]

corr_benzene_restricted, _ = stats.pearsonr(
    restricted_df["benzene"], restricted_df["malignancy"]
)
corr_cadmium_restricted, _ = stats.pearsonr(
    restricted_df["cadmium"], restricted_df["malignancy"]
)

print(
    "Restricted Correlation (Controlled for pollution) - Benzene vs Malignancy:"
    f" {corr_benzene_restricted:.4f}"
)
print(
    "Restricted Correlation (Controlled for pollution) - Cadmium vs Malignancy:"
    f" {corr_cadmium_restricted:.4f}"
)
print("Conclusion: The association changes once pollution is accounted for[cite: 2]!")