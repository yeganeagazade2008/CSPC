"""
PW2 Lab B Part 5 (Bonus) -- Titration equivalence point
"""
import numpy as np
import matplotlib.pyplot as plt

# 1. Veriyi yükle
data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# 2. Sayısal türev hesabı (dpH/dV)
slope = np.gradient(pH, V)

# 3. Maksimum türevin olduğu indeksi (eşdeğerlik noktası) bul
max_idx = np.argmax(slope)
eq_volume = V[max_idx]
eq_pH = pH[max_idx]

print(f"=== Bonus Exercise Result ===")
print(f"Equivalence point volume: {eq_volume:.2f} mL")
print(f"pH at equivalence point: {eq_pH:.2f}")

# 4. Grafikleri çiz ve kaydet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# pH Eğrisi
ax1.plot(V, pH, color="purple", label="pH Curve")
ax1.axvline(eq_volume, color="red", linestyle="--", label=f"Eq Point ({eq_volume:.1f} mL)")
ax1.set_title("Titration Curve")
ax1.set_xlabel("Volume of Base (mL)")
ax1.set_ylabel("pH")
ax1.legend()

# Türev (Eğim) Eğrisi
ax2.plot(V, slope, color="green", label="dpH/dV")
ax2.axvline(eq_volume, color="red", linestyle="--", label=f"Max Slope ({eq_volume:.1f} mL)")
ax2.set_title("First Derivative (Eğim)")
ax2.set_xlabel("Volume of Base (mL)")
ax2.set_ylabel("dpH/dV")
ax2.legend()

plt.tight_layout()
plt.savefig("titration.png")
print("titration.png was created successfully!")