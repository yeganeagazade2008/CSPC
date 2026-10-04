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