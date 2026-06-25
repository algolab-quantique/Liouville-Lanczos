import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
out_dir = script_dir / "outputs" / "five_sites"
out_dir.mkdir(exist_ok=True)
# Load saved data
kmax = 3

df = pd.read_csv(out_dir / f"{kmax}_iterations.csv")

# -----------------------------
# Reconstruct energy data
# -----------------------------

energy_df = df.dropna(subset=["energy_iteration"])

omega = energy_df["energy_iteration"].to_numpy()
true_gs_energy = energy_df["true_gs_energy"].to_numpy()[0]
true_energy_line = energy_df["true_gs_energy"].to_numpy()
spin_gm_energy = energy_df["spin_gm_energy"].to_numpy()

real = omega
imag = true_energy_line

# -----------------------------
# Reconstruct Green's function data
# -----------------------------

green_df = df.dropna(subset=["w_real"])

w = green_df["w_real"].to_numpy() + 1j * green_df["w_imag"].to_numpy()
x = np.real(w)

green1_w = green_df["green1_real"].to_numpy() + 1j * green_df["green1_imag"].to_numpy()
green2_w = green_df["green2_real"].to_numpy() + 1j * green_df["green2_imag"].to_numpy()

analytical0_w = green_df["analytical0_real"].to_numpy() + 1j * green_df["analytical0_imag"].to_numpy()
analytical1_w = green_df["analytical1_real"].to_numpy() + 1j * green_df["analytical1_imag"].to_numpy()

# -----------------------------
# Plot energies exactly like before
# -----------------------------

plt.figure()

plt.plot(real, imag, label="true", color="black")
plt.plot(spin_gm_energy, color="red", label="sum")
plt.ylabel("energy")
plt.xlabel("iteration")

plt.legend()
plt.show()

# -----------------------------
# Plot Greens exactly like before
# -----------------------------

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(
    x,
    np.imag(green1_w) + np.imag(green2_w),
    color="blue",
    label="Lanczos",
)
axs[0].plot(
    x,
    np.imag(analytical0_w) + np.imag(analytical1_w),
    label="GM",
    color="red",
    linestyle="--",
)
axs[0].set_title("green1 + green2")
axs[0].set_ylabel("Im G")
axs[0].legend()

axs[1].plot(
    x,
    np.imag(green1_w),
    color="blue",
    label="Lanczos",
)
axs[1].plot(
    x,
    np.imag(analytical1_w),
    label="GM",
    color="red",
    linestyle="--",
)
axs[1].set_title("green1")
axs[1].set_ylabel("Im G")
axs[1].legend()

axs[2].plot(
    x,
    np.imag(green2_w),
    color="blue",
    label="Lanczos",
)
axs[2].plot(
    x,
    np.imag(analytical0_w),
    label="GM",
    color="red",
    linestyle="--",
)
axs[2].set_title("green2")
axs[2].set_ylabel("Im G")
axs[2].set_xlabel(r"$\omega$")
axs[2].legend()

plt.tight_layout()
plt.show()