# CSPC Coursework

Repository for CSPC practical work assignments.

## PW1 --- Lab A

- **Setup & Environment:** Configured conda virtual environment and directory structure.
- **Git Version Control:** Set up repository with appropriate `.gitignore` and version-controlled scripts.

## PW1 --- Lab B

- **Data Analysis:** The observed data displays an exponential decay trend over time.
- **Model Match:** The observed scatter plot matches the analytical decay law (\(N_0 e^{-\lambda t}\) with \(\lambda = 0.3\)).
- **Snakemake Pipeline:** Automates figure generation by executing `plot.py` only when input files are modified.

## PW2 Lab A - Motion from Tracking Data

### Results
- **Mean Acceleration:** -8.5797 m/s²
- **Standard Deviation of Acceleration:** 28.7161 m/s²
- **Max Difference in Recovered Position:** 0.7846 m

### Bonus Task: Noise Reduction (Savitzky-Golay Filter)
- **Smoothed Mean Acceleration:** -9.4595 m/s²
- **Smoothed Standard Deviation of Acceleration:** 3.4375 m/s²

*Note: Applying the Savitzky-Golay filter significantly reduced the high-frequency noise in acceleration, bringing the standard deviation down from 28.72 m/s² to 3.44 m/s² and yielding a mean acceleration much closer to theoretical gravity (-9.81 m/s²).*

### Observations on Noise
Differentiation amplifies high-frequency noise because it compares nearby measurements where small errors become large rates of change; repeating this twice for acceleration creates severe noise (std ~28.72 m/s²). Conversely, integration acts as a cumulative summation where random noise tends to cancel itself out, successfully recovering the original position curve within less than a meter (0.7846 m).

### Visualisation
![Motion Analysis](PW2/Lab%20A/motion.png)

## PW2 - Lab B: Optimization in Chemistry

### Part 2: Optimization Methods
- **Convex Function:** All three methods (Gradient Descent, Newton, SLSQP) converged to x ≈ 3.
- **Complex Landscape:** Starting points heavily affect the result. Newton's method finds stationary points (where f'(x)=0), which can be local maxima or minima. Checking the second derivative (f''(x) > 0) is necessary to confirm a minimum.

### Part 3: Reaction Rate Fitting
- Fitted rate constant \(k \approx 0.25 \text{ s}^{-1}\).

### Part 4: Chemical Equilibrium
- Both Newton and SLSQP yielded \(x \approx 0.66\), giving equilibrium amounts: \(n_{H_2} = n_{I_2} \approx 0.34 \text{ mol}\), \(n_{HI} \approx 1.33 \text{ mol}\).

### Part 5: Titration Equivalence Point
- Peak slope identified equivalence point at \(V \approx 50 \text{ mL}\).

# Practical Work 3 (PW3) Report

## Session 1: Characterising Distributions
- **Data Loading & Isolation**: Loaded `heart.csv` using pandas, extracted continuous variables (`age`, `chol`, `trestbps`, `thalach`).
- **Visualisation**: Generated distribution plots (`distributions.png`).
- **Normality Tests**: Conducted Shapiro-Wilk tests. `age` and `thalach` are approximately normal, whereas `chol` and `trestbps` deviate from normality due to right-skewness.

## Session 2: Conclusions & Causation
- **Heart Rate Comparison**: Compared maximum heart rates (`thalach`) between disease and healthy groups using statistical testing and visualised with SEM uncertainty (`thalach_comparison.png`).
- **Correlation**: Analysed the relationship between age and max heart rate (`age_thalach_corr.png`).
- **Chemical Mystery**: 
  - *Naive Analysis*: Benzene appeared strongly associated with malignancy.
  - *Controlled Analysis*: After controlling for the confounding variable (`pollution_index`), the false association was revealed, showing the true nature of correlation vs. causation.
  # Practical Work 4 (PW4): Machine Learning Pipelines (Session A & Session B)

This repository contains the complete implementation and automation for **Practical Work 4 (PW4)**, covering both supervised regression/classification (Session A) and unsupervised machine learning / dimensionality reduction (Session B).

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Session A: Concrete Compressive Strength Analysis](#session-a-concrete-compressive-strength-analysis)
3. [Session B: UCI Wine Dataset & PCA Clustering](#session-b-uci-wine-dataset--pca-clustering)
4. [Pipeline Automation with Snakemake](#pipeline-automation-with-snakemake)
5. [Repository Structure](#repository-structure)

---

## 🔎 Overview
- **Student:** Yegana Aghazada
- **Course:** CSPC (Computer Science)
- **Environment:** Conda environment (`cspc`) with Python 3.13, Pandas, Scikit-Learn, Matplotlib, and Snakemake.

---

## 🏗️ Session A: Concrete Compressive Strength Analysis
- **Dataset:** Concrete compressive strength measurements with multiple chemical and physical features (cement, slag, ash, water, age, etc.).
- **Preprocessing:** Cleaned dataset by checking for missing values, handling duplicates, and scaling features.
- **Models Applied:** 
  - Linear and Polynomial Regression for strength prediction.
  - KNN and Decision Tree models with hyperparameter sweeps ($K$ values and tree depth).
- **Key Artifacts:** 
  - `results/A2_plots.png`
  - `results/A4_residuals.png`
  - `results/A6_knn_sweep.png`
  - `results/A6_tree_sweep.png`
  - `results/A_metrics.csv`

---

## 🍷 Session B: UCI Wine Dataset & PCA Clustering
- **Dataset:** 178 wine samples with 13 chemical measurements derived from 3 distinct Italian cultivars.
- **Step B1 (Loading & Exploration):** Inspected dataset shape `(178, 13)` and descriptive statistics. Identified extreme numerical scale variations (e.g., `proline` ranging up to 1680 vs. smaller scales like `ash` and `nonflavanoid_phenols`).
- **Step B2 (Raw PCA):** Demonstrated that unscaled PCA is completely dominated by `proline` due to its large numerical range, proving that raw PCA cannot reflect true chemical variance.
- **Step B3 (Scaled PCA & Scree Plot):** Applied `StandardScaler` (mean=0, std=1) before PCA. 
  - PC1 explains **36.20%** and PC2 explains **19.21%** of the variance (Cumulative 55.41% for first 2 PCs).
  - Saved Scree Plot to `results/B3_scree_plot.png`.
- **Step B4 & B5 (K-Means Clustering):** Performed K-Means with $k=3$ on the scaled space. Extracted cluster sizes (`[65, 51, 62]`) and inverse-transformed cluster centers back to real measurement units.
- **Step B6 (The Reveal & Agreement):** Compared unsupervised clusters against hidden true cultivars, achieving an outstanding **Agreement Score of 0.9663**.
  - Saved agreement results to `results/B_agreement.csv`.

---

## ⚙️ Pipeline Automation with Snakemake
All data processing steps, machine learning scripts, and result generations are fully automated using a DAG-based `Snakefile`.
- **Dry Run Verification:** `snakemake -n`
- **Full Execution:** `snakemake --cores 1`

---

## 📂 Repository Structure
```text
PW4/
├── sessionA.py
├── sessionB.py
├── Snakefile
├── fetch_concrete.py
├── README.md
└── results/
    ├── A2_plots.png
    ├── A4_residuals.png
    ├── A6_knn_sweep.png
    ├── A6_tree_sweep.png
    ├── A_metrics.csv
    ├── B3_scree_plot.png
    └── B_agreement.csv