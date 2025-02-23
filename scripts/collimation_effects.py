import numpy as np
import matplotlib
matplotlib.use('TkAgg')
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

from matplotlib import pyplot as plt
from framework import file_m2k
from framework.post_proc import envelope
from bisect import bisect

from pipe_lens.surf_estimation import image_correction, surfaces
from pipe_lens.imaging_utils import linear2db

root = "../data/collimation/"
filename = ["no_collimation", "20mm_collimation", "5mm_collimation"]

# Read the ultrassound data:
data = [file_m2k.read(root + filename + '.m2k', freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0) for filename in filename]

# Define the wedge aperture of 90 degrees:
delta_alpha = .1 # degree
alpha_max = 45
angle_span = np.arange(-alpha_max, alpha_max + delta_alpha, delta_alpha)

# Extract relevant data from ultrasound file:
time_span = data[0].time_grid
sscans = [np.sum(data.ascan_data[..., 0], axis=2) for data in data] # sum all channels to obtain S-Scan
ascans = [sscan[:, 92] for sscan in sscans] # extract the a-scan allign with alpha = 0 degrees.
ascans_db = [20 * np.log10(envelope(ascan, axis=0) + 1e-6) for ascan in ascans]

#%% Plots the results:

plt.figure(figsize=(6 * .95, 4 * .85))
plt.plot(time_span, ascans_db[0], color='k', label="No collimation.", linewidth=2)
plt.plot(time_span, ascans_db[1], color='b', label="20 mm collimation window.", linewidth=2)
plt.plot(time_span, ascans_db[2], color='r', label="5 mm collimation window.", linewidth=2)
# plt.legend()
plt.xlim([52.5, 62.5])
plt.xlabel(r"Time in $\mu$s")
plt.ylabel("Amplitude in dB")
plt.grid()
plt.tight_layout()
plt.gca().annotate('Scatterer', xy=(59.5, 97.4), xytext=(56.5, 112.4),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.gca().annotate('Inner wall', xy=(60.8, 84.1), xytext=(59.56, 53.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.gca().annotate('Outer wall', xy=(54.3, 84.1), xytext=(54.3, 53.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.xticks(np.arange(52, 64, 2))

plot_fig = False
if plot_fig:
    plt.show()
else:
    plt.savefig("../figures/collimation_effects.pdf")