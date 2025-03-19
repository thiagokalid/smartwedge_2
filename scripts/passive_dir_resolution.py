from framework import file_m2k
import numpy as np
import matplotlib, scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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

from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/2025-03-12 - BH/'
acoustic_lens_type = "xl"
plot_fig = True # Chooses to either plot or save the output.

#%%
data = file_m2k.read(root + f"passive_dir_{acoustic_lens_type}_0degree.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian')[1]
data_ref = file_m2k.read(root + f"ref.m2k", type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian')[1]
#%%
n_shots = data.ascan_data.shape[-1]
log_cte = 1e-6
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)
vmax, vmin = 0, -120

# It was manually identified where the outer and inner surface was (in microseconds):
t_outer, t_inner = 56.13, 63.22
r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

# Seta os vetores dos dos dados
widths, heights, peaks, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)

shot_range = [
    (0, 11), # 0 degree
    (0, 0), # 10 degree
    (0, 13), # 20 degree
    ()  # 30 degree
]

angular_location = [
    0., # degree
    10., # degree
    16.,
    30
]

for i in tqdm(range(13)):
    channels = data.ascan_data[:, :, :, i]
    channels_ref = np.mean(data_ref.ascan_data, axis=3)
    sscan = np.sum(channels - channels_ref, axis=2)
    sscan_db = 20 * np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

    # Aplica API para descobrir a área acima de -6 dB
    corners = [(60, -10 + 0), (52, +10 + 0  )]

    widths[i], heights[i], peaks[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

    if False:
        plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='magma', aspect='auto', interpolation="none")
        plt.imshow(pixels_above_threshold, extent=[ang_span[0], ang_span[-1], r_span[-1], r_span[0]], cmap='magma', aspect='auto', interpolation="none")

#%% Plots results:
peaks_dB = 20 * np.log10(peaks + log_cte)
theoretical_maximum = 2**16 * 64
peaks_relative = peaks / theoretical_maximum * 100
half_peak = (peaks.max() - peaks.min())/2 + peaks.min()
xspan = np.arange(0, n_shots) - np.where(peaks==peaks.max())[0]
peaks_interp = lambda x: np.interp(x, xspan, peaks)

cost_fun = lambda x: np.power(peaks_interp(x) - half_peak, 2)
half_peak_loc_left = scipy.optimize.minimize(cost_fun, xspan[0], bounds=[(xspan[0], xspan[peaks.argmax()])]).x
half_peak_loc_right = scipy.optimize.minimize(cost_fun, xspan[0], bounds=[(xspan[peaks.argmax()], xspan[-1])]).x
passive_flaw_width = half_peak_loc_right[0] - half_peak_loc_left[0]


fig, ax = plt.subplots(figsize=(linewidth * .49, 3))
plt.plot(xspan, peaks_relative, ":o", color="k")
plt.xlabel("Passive direction movement / [mm]")
plt.ylabel("Relative amplitude / [%]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.2f}"))
plt.ylim([-0.01, 0.25])
plt.yticks(np.arange(-0.01, 0.25, 0.05))
plt.xlim([-4.5, 4.5])
ytemp = np.arange(0, 0.15, 1e-2)
plt.xticks(np.linspace(-4, 4, 5))
plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
plt.grid(alpha=.5)

ax.annotate("", xy=(half_peak_loc_left , 0.04), xytext=(half_peak_loc_right, 0.04),
            arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(rf"${passive_flaw_width:.2f}$ mm", xy=(0.22, .045), xytext=(0.22, .045),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )




plt.tight_layout()
plt.savefig("../figures/passive_dir_resolution.pdf")
plt.show()
