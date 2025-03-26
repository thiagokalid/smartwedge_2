from framework import file_m2k
import numpy as np
import time

from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm
import gc

# Modules from matplotlib for advanced plotting:
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
matplotlib.use('TkAgg')
font = {
    'weight' : 'normal',
    'size'   : 9
}
# Set the default font to DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rc('font', **font)

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/resolution/active_dir/'

delay_law_type = ["focused", "planewave"]

width_dict = {"focused": [], "planewave": []}
height_dict = {"focused": [], "planewave": []}
maximums_dict = {"focused": [], "planewave": []}

for ii in range(2):
    print("ii = ", ii)

    for jj in range(2):
        print("jj = ", jj)
        delay_law = delay_law_type[ii]
        run = jj + 1

        data = file_m2k.read(root + f"active_dir_xl_{delay_law}_1degree_v{run}.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
        data_ref = file_m2k.read(root + f"ref_xl_{delay_law}.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

        if delay_law == "planewave":
            data.ascan_data = data.ascan_data[:, ::-1, :, :]
            data_ref.ascan_data = data_ref.ascan_data[:, ::-1, :, :]

        n_shots = 40
        log_cte = 1e-6
        theoretical_maximum = 2**16 * data.probe_params.num_elem
        time_grid = data.time_grid[:, 0]
        ang_span = np.linspace(-45, 45, 181)
        vmax, vmin = 0, -120
        sweep_angs = np.linspace(-6, 44, n_shots)

        # It was manually identified where the front and back wall was (in microseconds):
        t_outer, t_inner = 55.27, 60.75
        rtop, rbottom = 62.0, 58.0


        r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

        # Define
        m = 0
        mmax = 4
        plot_position = np.linspace(0, 36, mmax)


        # Seta os vetores dos dos dados
        widths, heights, maximums, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)

        plt.subplots(figsize=(15.5,9))
        for i in tqdm(range(n_shots)):
            channels = data.ascan_data[:, :, :, i]
            channels_ref = np.mean(data_ref.ascan_data, axis=3)
            sscan = np.sum(channels - channels_ref, axis=2)
            sscan_db = 20 * np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

            # Aplica API para descobrir a área acima de -6 dB
            current_angle = 1 * i
            corners = [(rtop, -10 + current_angle), (rbottom, 10 + current_angle)]

            widths[i], heights[i], maximums[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

        width_dict[delay_law].append(widths)
        height_dict[delay_law].append(heights)
        maximums_dict[delay_law].append(maximums)

        del data, data_ref, sscan, sscan_db, pixels_above_threshold, channels, channels_ref
        gc.collect()
        time.sleep(30)
        gc.collect()

#%% Plotting the results:

# ############################################
# Figure 1:
fig, ax = plt.subplots(figsize=(6,4))


# Width y-axis:
plt.plot(sweep_angs, (width_dict["focused"][0] + width_dict["focused"][1])/2, 'o:', color='k', markersize=5, label="Focused waves")
plt.plot(sweep_angs, (width_dict["planewave"][0] + width_dict["planewave"][1])/2, 's-', color='k', markersize=3, label="Plane waves")

# Axes:
plt.xticks(sweep_angs)
plt.ylabel("Width / [mm]")
plt.xlabel(r"$\alpha$-axis / [degrees]")
plt.grid(alpha=.5)
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 47])
plt.tight_layout()
plt.ylim([0.0, 11])
# plt.legend(loc="upper left")

# Formating plot:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.0f}$"))

# Height y-axis:
twin = ax.twinx()
twin.plot(sweep_angs, (height_dict["focused"][0] + height_dict["focused"][1])/2, 'o:', color='r', markersize=5)
twin.plot(sweep_angs, (height_dict["planewave"][0] + height_dict["planewave"][1])/2, 's-', color='r', markersize=3)
twin.yaxis.label.set_color("r")
twin.tick_params(axis='y', colors='r')

plt.xticks(sweep_angs)
plt.ylabel("Height / (mm)")
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.grid(alpha=.5)
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 47])
plt.tight_layout()
plt.ylim([0.0, 11])

# Formating plot:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.0f}$"))

plt.savefig("../figures/active_dir_resolution.pdf")
plt.tight_layout()
plt.show()

# ############################################
# Figure 2:
fig, ax = plt.subplots(figsize=(6,4))

# Curves:
plt.plot(sweep_angs, (maximums_dict["focused"][0] + maximums_dict["focused"][1])/(theoretical_maximum * 2) * 100, 'o:', color='k', label="Focused waves", markersize=5)
plt.plot(sweep_angs, (maximums_dict["planewave"][0] + maximums_dict["planewave"][1])/(theoretical_maximum * 2) * 100, 's-', color='k', label="Plane waves", markersize=5)

# Axes:
plt.xticks(sweep_angs)
plt.ylabel("Relative Amplitude")
plt.xlabel(r"$\alpha$-axis / [degrees]")
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 45])
plt.ylim([0, 6])
plt.grid(alpha=.5)
plt.legend()

# Formating axes:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.1f}$ %"))

plt.tight_layout()
plt.savefig("../figures/amplitude_comparisson.pdf")
plt.show()
