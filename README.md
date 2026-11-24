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


    # PW5: Where was the earthquake, and how strong?

All results below come from `data/seismogram_A.csv` (steps 1-8) and from both records (steps 9-10).
Sampling rate: fs = 100 Hz, record length 300 s, 30000 samples.

> Note: my first copy of `seismogram_A.csv` was truncated (only 1061 lines, the last line was cut in the middle of a number).
> I downloaded it again; the final file has 30000 samples like seismogram_B.

## 1. Look at the record

![three channels](figures/step1.png)
![first 60 s](figures/step1_zoom.png)

The record starts quiet. At about 108 s a weaker shaking begins (P-wave), and at about 144 s a much stronger one begins (S-wave).
After that the amplitude slowly decays until the end of the record. The P-wave is clearest on the vertical channel Z.
The S-wave is largest on the horizontal channels N and E (about 10 mm/s, against about 4 mm/s on Z).

In the first 60 s the station is not really still: all three channels show a continuous slow oscillation of about ±0.005 mm/s.
This is the background noise of the Earth (microseism), visible only when zooming in, because the earthquake is about 1000 times larger.

## 2. The spectrum

![index vs frequency](figures/step2a.png)
![log-log spectra](figures/step2b.png)

(a) The FFT returns one number per sample, ordered by array index, which has no physical meaning.
`fftfreq` tells which frequency in Hz belongs to each index, so the x-axis becomes a real frequency axis from -50 to +50 Hz (fs/2 is the maximum).
The spectrum is mirrored around 0 Hz because the signal is real: the value at -f is the complex conjugate of the value at +f, so the negative half carries no new information.

(b) The quiet part (first 60 s) has its maximum at about 0.1-0.3 Hz, which is the microseism caused by ocean waves.
Above about 1 Hz it falls to a flat low level of about 1e-8 m/s.
The whole record is higher than the quiet part at all frequencies, by two to three orders of magnitude above 1 Hz.
The earthquake adds most of its energy between about 0.1 and 10 Hz, strongest below about 1 Hz, and it fades quickly above 10 Hz.

## 3. Inverse transform and band-pass filter

![raw and filtered Z](figures/step3b.png)
![Z - clean](figures/step3c.png)

(a) `np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z)` is `True`.
Filtering in the frequency domain means: transform to frequencies, delete some of them, and go back to time.
This only works because the inverse transform brings back exactly the original signal when nothing is deleted. Without that, the filtered result would be distorted by the transform itself.

(b) I used `bandpass(x, fs, 0.5, 10)`.
Below 0.5 Hz the spectrum is dominated by the microseism (peak at 0.1-0.3 Hz), which hides the weak P arrival.
Above 10 Hz the earthquake energy is already small and falls off, so only noise would be kept.
The band 0.5-10 Hz keeps the part where the earthquake is much stronger than the noise.

(c) `Z - clean` shows what was removed: almost nothing before the earthquake (the microseism is tiny)
and, after about 108 s, slow large oscillations, strongest around 150-170 s.
So the removed part is mainly the low-frequency part of the earthquake itself (below 0.5 Hz), not just noise.
This is fine for finding arrival times, which is why PGV is computed from the raw signal (step 8).

## 4. The envelope

![envelopes, 0.5 s](figures/step4a.png)
![windows 0.05 s, 0.5 s, 5 s](figures/step4b.png)

With a window of 0.05 s the envelope is very jagged: it follows every single oscillation, and one random spike can cross a threshold.
With 5 s the envelope is very smooth, but the sharp start of the P and S waves is spread out.
With "same" mode the window looks forward and backward, so the envelope starts rising before the real onset, and the detected arrival would be too early.
0.5 s is a compromise: smooth enough to be stable, short enough to keep a sharp onset.

## 5. Arrival times

![arrivals](figures/step5.png)

- tP = 108.42 s (first time the Z envelope exceeds 5 % of its maximum)
- tS = 144.08 s (first time after tP + 2 s that the horizontal envelope exceeds 10 % of its maximum)
- the maximum of the horizontal envelope is at 152.03 s (find_peaks), that is 7.95 s after tS

The arrival time is when the shaking starts, not when it is strongest.
Using the maximum would make tS - tP too large by about 7.95 s, and since R is proportional to tS - tP (8.6 km per second), R would be too large by about 68 km.

## 6. Distance

R = (tS - tP) / (1/vS - 1/vP) = 35.66 s x 8.585 km/s, which gives **R ≈ 306 km**.

If tS is uncertain by 1 s, R changes by 8.6 km (about 2.8 %).
Origin time = tP - R/vP = **59.0 s** after the start of the record.

## 7. Spectrogram

![spectrogram of Z](figures/step7.png)

Before about 105 s the spectrogram is dark (quiet), with only a thin bright line near 0 Hz (the microseism).
At about 108 s there is a sudden jump in power over the whole 0-10 Hz range: this is the P-wave.
At about 144 s a second, stronger jump appears, brightest at low frequencies (0-4 Hz): the S-wave.
After that the energy slowly decays and stays longer at low frequencies, which is where the later, slower waves are.
In this record the P-wave is the one that reaches the highest frequencies, and the later part is dominated by low frequencies.
The surface waves are not a clearly separated band in this synthetic record.

A single spectrum (step 2) gives the frequency content of the whole record but loses the time information.
The spectrogram shows which frequencies are present at which time, so the different wave types can be separated.

## 8. Magnitude

PGV (maximum of sqrt(N² + E²) of the raw record) = **1.51 cm/s**, and

M = (log10(1.51) + log10(306.1) + 1.0) / 0.55 = **6.66**

Why log10 R: waves get weaker with distance (the energy spreads over a larger volume and is absorbed by the rock).
The same earthquake recorded farther away gives a smaller PGV. Without the log10 R term the farther station would underestimate the magnitude.
At twice the distance the correction is log10(2)/0.55 ≈ 0.55 magnitude units.

One magnitude unit more at the same distance means log10 PGV increases by 0.55, so PGV is 10^0.55 ≈ **3.55 times larger** with this formula.
(The task text says "about 10 times larger ground motion"; with this formula the factor is 3.55.)

## 9. The whole system

`analyse.py` contains the function `analyse(csv_file)` (filter, envelopes, tP, tS, R, origin time, PGV, M) and was run on both records without changing anything.
`report.py` reads `results/events.csv` and writes `results/report.txt`.

| Record | tP (s) | tS (s) | R (km) | PGV (cm/s) | M |
|---|---|---|---|---|---|
| A | 108.42 | 144.08 | 306.1 | 1.51 | 6.66 |
| B | 84.25 | 102.43 | 156.1 | 0.571 | 5.36 |

Earthquake B was **closer** (156 km against 306 km), and earthquake A was **stronger** (M 6.66 against 5.36).
Being closer does not mean being stronger: B is nearer, but its PGV is smaller, and the magnitude, which corrects for distance, is also smaller.
Both records give almost the same origin time (59.0 s and 59.1 s), which is a good consistency check of the method.

## 10. Snakemake

**(a)** After `rm -rf results` and `snakemake --cores 1`, all three rules ran (analyse, report, all).
Reason shown: `Missing output files: results/events.csv`.

**(b)** Running it again: `Nothing to be done (all requested files are present and up to date).`
The outputs exist and are newer than all their inputs, so nothing needs to be recomputed.

**(c)** After changing only the wording in `report.py`, only `report` and `all` ran (reason: `Updated input files: report.py`), `analyse` did not.
The inputs of `analyse` did not change, so `events.csv` is still valid.

**(d)** After changing the cut-off b in `analyse.py`, `analyse`, `report` and `all` all ran
(reasons: `Updated input files: analyse.py`, then `Input files updated by another job: results/events.csv`).
With b = 8 Hz the results changed slightly (for A: R = 306.6 km, M = 6.67).
The difference with (c) is what I want from a pipeline: a change at the start of the chain makes everything that depends on it outdated, so it is recomputed,
while a change at the end only reruns the last step. Results always match the current code, and nothing is recomputed for no reason.
Afterwards I set b back to 10 Hz and reran, so the final `results/` use 0.5-10 Hz.

## Bonus

Not done.