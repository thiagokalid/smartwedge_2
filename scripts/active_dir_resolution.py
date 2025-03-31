import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from framework.post_proc import envelope
from framework import file_m2k
import time
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm
import gc

linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/resolution/active_dir/'

delay_law_type = ["focused", "planewave"]

width_dict = {"focused": [], "planewave": [], "single-element": []}
height_dict = {"focused": [], "planewave": [], "single-element": []}
maximums_dict = {"focused": [], "planewave": [], "single-element": []}

num_versions = 2
num_laws = len(delay_law_type)

#%% Single-element:
print("Single-element acquisition:")
for v in tqdm(range(3)):
    data = file_m2k.read(root + f"active_dir_single_element_v{v+1}.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
    time_grid = data.time_grid
    pseudo_sscan = data.ascan_data[:, 0, 0, :40]
    # It was manually identified where the front and back wall was (in microseconds):
    t_outer, t_inner = 17, 22.48
    rtop, rbottom = 65.0, 55.0
    r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)
    ang_span = np.arange(-10, 10, .5)
    corners = [(rtop, -4), (rbottom, 4)]

    width_single, height_single, maximum_single, pixels_above_threshold, = fwhm(pseudo_sscan, r_span, ang_span, corners, drawSquare=False)

    # if False:
    #     plt.figure()
    #     plt.imshow(np.log10(envelope(pseudo_sscan, axis=0) + 1e-6), extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]])
    #     plt.show()
    #
    #     plt.figure()
    #     plt.imshow(pixels_above_threshold, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]])
    #     plt.show()

    width_dict["single-element"].append(width_single)
    height_dict["single-element"].append(height_single)
    maximums_dict["single-element"].append(maximum_single)

#%% Acoustic lens with Phased array:

print("")
print("Acoustic lens + Phased Array acquisition:")
for idx in tqdm(range(num_laws * num_versions)):
    ii = idx // 2
    jj = idx % 2

    delay_law = delay_law_type[ii]
    run = jj + 1

    data = file_m2k.read(root + f"active_dir_xl_{delay_law}_1degree_v{run}.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
    data_ref = file_m2k.read(root + f"ref_xl_{delay_law}.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
    channels_ref = np.mean(data_ref.ascan_data, axis=3)

    if delay_law == "planewave":
        data.ascan_data = data.ascan_data[:, ::-1, :, :]
        data_ref.ascan_data = data_ref.ascan_data[:, ::-1, :, :]

    n_shots = 40
    log_cte = 1e-6
    time_grid = data.time_grid[:, 0]
    ang_span = np.linspace(-45, 45, 181)
    sweep_angs = np.linspace(-6, 44, n_shots)

    # It was manually identified where the front and back wall was (in microseconds):
    t_outer, t_inner = 55.27, 60.75
    rtop, rbottom = 62.0, 58.0
    r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

    # Seta os vetores dos dos dados
    widths, heights, maximums, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)

    plt.subplots(figsize=(15.5,9))
    for i in range(n_shots):
        channels = data.ascan_data[:, :, :, i]

        sscan = np.sum(channels - channels_ref, axis=2)

        # Aplica API para descobrir a área acima de -6 dB
        current_angle = 1 * i
        corners = [(rtop, -10 + current_angle), (rbottom, 10 + current_angle)]

        widths[i], heights[i], maximums[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

    width_dict[delay_law].append(widths)
    height_dict[delay_law].append(heights)
    maximums_dict[delay_law].append(maximums)

    del data, data_ref, sscan, pixels_above_threshold, channels, channels_ref
    gc.collect()
    time.sleep(5)
    gc.collect()

#%%

focused_maximum = np.mean(maximums_dict["focused"], axis=0)
plane_maximum = np.mean(maximums_dict["planewave"], axis=0)
focused_amplitude_std = np.std(maximums_dict["focused"], axis=0)
plane_amplitude_std = np.std(maximums_dict["planewave"], axis=0)
absolute_maximum = np.max([focused_maximum, plane_maximum])

focused_width = np.mean(width_dict["focused"], axis=0)
plane_width = np.mean(width_dict["planewave"], axis=0)
focused_width_std = np.std(width_dict["focused"], axis=0)
plane_width_std = np.std(width_dict["planewave"], axis=0)

focused_height = np.mean(height_dict["focused"], axis=0)
plane_height = np.mean(height_dict["planewave"], axis=0)
focused_height_std = np.std(height_dict["focused"], axis=0)
plane_height_std = np.std(height_dict["planewave"], axis=0)

single_height = np.ones_like(sweep_angs) * np.mean(height_dict["single-element"], axis=0)
single_width = np.ones_like(sweep_angs) * np.mean(width_dict["single-element"], axis=0)
single_width_std = np.ones_like(sweep_angs) * np.std(width_dict["single-element"], axis=0)
single_height_std = np.ones_like(sweep_angs) * np.std(height_dict["single-element"], axis=0)

#%% Plotting the results:

#%% Figure 1:
fig, ax = plt.subplots(figsize=(linewidth * .485, 2.5))


# Width y-axis:
plt.plot(sweep_angs, focused_width, 'o-', linewidth=2, color='k', markersize=4, label="Focused waves")
ax.fill_between(sweep_angs, focused_width - focused_width_std, focused_width + focused_width_std, alpha=0.2, color='k')

plt.plot(sweep_angs, plane_width, 'x-', linewidth=1, color='#FF1F5B', markersize=4, label="Plane waves")
ax.fill_between(sweep_angs, plane_width - plane_width_std, plane_width + plane_width_std, alpha=0.2, color='#FF1F5B')

plt.plot(sweep_angs, single_width, color="#00CD6C", linewidth=2, label="Single element")
ax.fill_between(sweep_angs, single_width - single_width_std, single_width + single_width_std, alpha=0.2, color='#00CD6C')
plt.legend()

# Axes:
plt.xticks(sweep_angs)
plt.ylabel(r"FWHM along $\alpha$-axis / (degrees)")
plt.xlabel(r"Scatterer position / (degrees)")
plt.grid(alpha=.5)
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 47])
plt.tight_layout()
plt.ylim([0.0, 7])
plt.yticks(np.arange(0, 7 + 1, 1))
# plt.legend(loc="upper left")

# Formating plot:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.0f}$"))

plt.savefig("../figures/active_dir_width.pdf")
plt.tight_layout()
plt.show()

#%% Figure 2:
fig, ax = plt.subplots(figsize=(linewidth * .485, 2.5))


# Width y-axis:
plt.plot(sweep_angs, focused_height, 'o-', linewidth=2, color='k', markersize=4, label="Focused waves")
ax.fill_between(sweep_angs, focused_height - focused_height_std, focused_height + focused_height_std, alpha=0.2, color='k')

plt.plot(sweep_angs, plane_height, 'x-', linewidth=1, color='#FF1F5B', markersize=4, label="Plane waves")
ax.fill_between(sweep_angs, plane_height - plane_height_std, plane_height + plane_height_std, alpha=0.2, color='#FF1F5B')

plt.plot(sweep_angs, single_height, color="#00CD6C", linewidth=2, label="Single element")
ax.fill_between(sweep_angs, single_height - single_height_std, single_height + single_height_std, alpha=0.2, color='#00CD6C')
# plt.legend()

# Axes:
plt.xticks(sweep_angs)
plt.ylabel(r"FWHM along time axis / ($\mathrm{\mu s}$)")
plt.xlabel(r"Scatterer position / (degrees)")
plt.grid(alpha=.5)
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 47])
plt.tight_layout()
plt.ylim([0.0, 11])
plt.yticks(np.arange(0, 10 + 1, 1))
# plt.legend(loc="upper left")

# Formating plot:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.0f}$"))

plt.savefig("../figures/active_dir_height.pdf")
plt.tight_layout()
plt.show()

#%%
# Figure 3:
fig, ax = plt.subplots(figsize=(linewidth * .485, 2.5))

focused = focused_maximum/absolute_maximum * 100
plane = plane_maximum/absolute_maximum * 100

focused_std = focused_amplitude_std/absolute_maximum * 100
plane_std = plane_amplitude_std/absolute_maximum * 100

# Curves:
plt.plot(sweep_angs, focused, 'o-', color='k', linewidth=2, label="Focused waves", markersize=4)
ax.fill_between(sweep_angs, focused - focused_std, focused + focused_std, alpha=0.2, color='k')

plt.plot(sweep_angs, plane, 'x-', color='#FF1F5B', linewidth=1, label="Plane waves", markersize=4)
ax.fill_between(sweep_angs, plane - plane_std, plane + plane_std, alpha=0.2, color='#FF1F5B')
# plt.legend()

# Axes:
plt.xticks(sweep_angs)
plt.yticks(np.arange(0, 125, 25))
plt.ylabel(r"Normalized Amplitude / (\%)")
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.xticks(np.arange(-6, 44 + 5, 5))
plt.xlim([-8, 45])
plt.ylim([-5, 110])
plt.grid(alpha=.5)

# Formating axes:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.0f}$ %"))

plt.tight_layout()
plt.savefig("../figures/amplitude_comparisson.pdf")
plt.show()
