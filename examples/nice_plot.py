import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Enable LaTeX rendering in matplotlib
# plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 12

# Parameters
mu_0 = 20  # Mean of f_0
mu_1 = 30  # Mean of f_1
sigma = 10  # Standard deviation (scale)
tau = 25  # Separator variable

# x values for plotting
x = np.linspace(0, 50, 500)

# Gaussian functions
f_0 = norm.pdf(x, mu_0, sigma)
f_1 = norm.pdf(x, mu_1, sigma)

# Ratio of the two functions
r = f_0 / f_1

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

### Plot 1: f_0, f_1, and r on the same plot with right axis for r ###
ax1 = axs[0]
ax1.plot(x, f_0, label=r"$P_0$", color="blue")
ax1.plot(x, f_1, label=r"$P_1$", color="red")

# Create a twin y-axis to plot r
ax1_r = ax1.twinx()
ax1_r.plot(x, r, "g--", label=r"$P_0 / P_1$")

# Add tau as vertical dashed line
ax1.axvline(x=tau, color="k", linestyle="--")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=4)

ax1.set_xlabel("")
ax1_r.set_ylabel("")
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1_r.set_yticklabels([])

# Remove tick marks
ax1.tick_params(axis="both", which="both", length=0)
ax1_r.tick_params(axis="y", which="both", length=0)

### Plot 2: f_0 and r with shaded area up to tau ###
ax2 = axs[1]
ax2_r = ax2.twinx()
ax2.plot(x, f_0, label=r"$f_0(x)$", color="blue")
ax2_r.plot(x, r, "g--", label=r"$r(x)$")

# Shade area under f_0 up to tau
x_fill = x[x > tau]
f0_fill = f_0[x > tau]
ax2.fill_between(x_fill, f0_fill, color="blue", alpha=0.3)

# Add tau as vertical dashed line
ax2.axvline(x=tau, color="k", linestyle="--")

# Add text to the plot
ax2.text(tau + 1, max(f_0) / 3, r"False Positive", fontsize=12)


ax2.set_xlabel("")
ax2.set_ylabel("")
ax2_r.set_ylabel("")
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2_r.set_yticklabels([])
ax2.tick_params(axis="both", which="both", length=0)
ax2_r.tick_params(axis="y", which="both", length=0)

### Plot 3: f_1 and r with shaded area up to tau ###
ax3 = axs[2]
ax3.plot(x, f_1, label=r"$f_1(x)$", color="red")
ax3_r = ax3.twinx()
ax3_r.plot(x, r, "g--", label=r"$r(x)$")

# Shade area under f_1 up to tau
x_fill = x[x <= tau]
f1_fill = f_1[x <= tau]
ax3.fill_between(x_fill, f1_fill, color="red", alpha=0.3)

# Add tau as vertical dashed line
ax3.axvline(x=tau, color="k", linestyle="--")

# Add text to the plot
ax3.text(tau - 9, max(f_1) / 3, r"False Negative", fontsize=12)

ax3.set_xlabel("")
ax3.set_ylabel("")
ax3_r.set_ylabel("")
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3_r.set_yticklabels([])
ax3.tick_params(axis="both", which="both", length=0)
ax3_r.tick_params(axis="y", which="both", length=0)

# Add common labels
fig.text(0.5, 0.04, r"$\tau$", ha="center", fontsize=14)
fig.text(
    0.04, 0.5, "Probability Density", va="center", rotation="vertical", fontsize=14
)
fig.text(0.25, 0.96, r"$H_0$", va="center", fontsize=18)
fig.text(0.75, 0.96, r"$H_1$", va="center", fontsize=18)

# Adjust layout and display the plot
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=0)
plt.show()
