import numpy as np
import matplotlib
matplotlib.use('TkAgg')
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)

from matplotlib import pyplot as plt
from framework import file_m2k
from framework.post_proc import envelope


root = "../data/collimation/"
filename = ["no_collimation", "20mm_collimation", "5mm_collimation"]

# Read the ultrassound data:
data = [file_m2k.read(root + filename + '.m2k', freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0) for filename in filename]

# Define the wedge aperture of 90 degrees:
delta_alpha = .1 # degree
alpha_max = 45
angle_span = np.arange(-alpha_max, alpha_max + delta_alpha, delta_alpha)

# Minor adjusment for enhanced plotting:
roll_idx = [20, 0, 0]

# Extract relevant data from ultrasound file:
time_span = data[0].time_grid
sscans = [np.sum(data.ascan_data[..., 0], axis=2) for data in data] # sum all channels to obtain S-Scan
ascans = [np.roll(sscan[:, 92], shift=roll, axis=0) for sscan, roll in zip(sscans, roll_idx)] # extract the a-scan allign with alpha = 0 degrees.
ascans_db = [20 * np.log10(envelope(ascan, axis=0) + 1e-6) for ascan in ascans]

#%% Plots the results:

fig, ax = plt.subplots(figsize=(6.5 , 4))
plt.plot(time_span, ascans_db[0], linestyle=":", color='k', label="No collimation.", linewidth=2)
plt.plot(time_span, ascans_db[1], linestyle="--", color='b', label="20 mm wide.", linewidth=2)
plt.plot(time_span, ascans_db[2], linestyle="-", color='r', label="5 mm wide "
                                                                  ".", linewidth=2)
plt.xlim([52.5, 62.5])
plt.xlabel(r"Time in $\mu$s")
plt.ylabel("Amplitude in dB")
plt.grid(alpha=.5)
ax.annotate('Scatterer', xy=(59.8, 100), xytext=(57.7, 112.4),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Inner wall', xy=(60.8, 84.1), xytext=(59.0, 53.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Outer wall', xy=(54.3, 84.1), xytext=(52.4, 53.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.43), ncol=2, fancybox=True, shadow=True)
plt.xticks(np.arange(52, 64, 2))
plt.tight_layout()
fig.subplots_adjust(bottom=0.3)  # Increase right margin

plot_fig = True
if plot_fig:
    plt.show()
else:
    plt.savefig("../figures/collimation_effects.pdf")