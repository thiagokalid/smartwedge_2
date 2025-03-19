import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, AutoMinorLocator

matplotlib.use('TkAgg')
font = {
    'weight' : 'normal',
    'size'   : 9
}
linewidth = 6.3091141732 # LaTeX linewidth

# Set the default font to DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rc('font', **font)


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

# Estimation of hardware maximum amplitude:
hardware_ceiling = 2**16 # its a 16bit channel
maximum_amplitude = hardware_ceiling * data[0].probe_params.num_elem

# Extract relevant data from ultrasound file:
time_span = data[0].time_grid
sscans = [np.sum(data.ascan_data[..., 0], axis=2) for data in data] # sum all channels to obtain S-Scan
ascans = [np.roll(sscan[:, 92], shift=roll, axis=0) for sscan, roll in zip(sscans, roll_idx)] # extract the a-scan allign with alpha = 0 degrees.
envelopes = [envelope(ascan, axis=0) for ascan in ascans]
relative_amplitudes = [envelope / maximum_amplitude * 100 for envelope in envelopes]

#%% Plots the results:

fig, ax = plt.subplots(figsize=(linewidth * .65 , 2.5))
plt.plot(time_span, relative_amplitudes[1], linestyle="-", color='b', label="20 mm", linewidth=1)
plt.plot(time_span, relative_amplitudes[2], linestyle="-", color='r', label="5 mm", linewidth=2)
plt.plot(time_span, relative_amplitudes[0], linestyle="--", color='k', label="No collimation", linewidth=.5)

plt.xlim([52.5, 62.5])
plt.xlabel(r"Time / [$\mu$s]")
plt.ylabel("Relative Amplitude / [%]")
plt.grid(axis='x', alpha=.25)
plt.grid(axis='y', alpha=.75)
ax.annotate('Scatterer', xy=(59.6, 3.5), xytext=(58.1, 7.4),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2))
ax.annotate('Back wall', xy=(60.2, 11.6), xytext=(58.1, 15),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2))
ax.annotate('Front wall', xy=(54.86, 10), xytext=(55.59, 11.5),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2))
ax.legend(loc='upper center', ncol=3, fancybox=False, shadow=False, columnspacing=1, framealpha=.5)
plt.xticks(np.arange(52, 64, 2))
plt.yticks(np.arange(0, 27.5, 5))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
plt.tight_layout()

plt.savefig("../figures/collimation_effects.pdf")
plt.show()