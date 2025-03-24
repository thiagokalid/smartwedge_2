from framework import file_m2k
import numpy as np
import time
from numpy import rad2deg
linewidth = 6.3091141732

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

data = file_m2k.read(root + f"active_dir_xl_focused_1degree_v1.m2k",
                     freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=20)
data_ref = file_m2k.read(root + f"ref_xl_focused.m2k",
                         freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

log_cte = 1e-6
theoretical_maximum = 2 ** 16 * data.probe_params.num_elem
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)

# It was manually identified where the front and back wall was (in microseconds):
t_outer, t_inner = 55.27, 60.75
rtop, rbottom = 62.0, 58.0

r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)


channels = data.ascan_data[..., 0]
channels_ref = np.mean(data_ref.ascan_data, axis=3)
sscan = np.sum(channels - channels_ref, axis=2)
sscan_envelope = envelope(sscan, axis=0)
sscan_db = 20 * np.log10(sscan_envelope + log_cte)


#%%

plt.figure(figsize=(linewidth*.48, 3))
plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='inferno', aspect='auto', interpolation='None', vmin=30, vmax=120)
ytemp = np.arange(40, 65, 1)
plt.plot(np.ones_like(ytemp) * 21, ytemp, "-", color='C0', linewidth=1)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / $\mu$s")
plt.colorbar()
plt.ylim([63, 53])

plt.tight_layout()
plt.savefig("../figures/sscan_backwall.pdf")
plt.show()


#%%

fig, ax = plt.subplots(figsize=(linewidth*.48, 3))
plt.plot(time_grid, sscan_envelope[:, 132] / theoretical_maximum * 100, color='C0')
plt.xlim([53, 63])
plt.ylim([-5, 60])
plt.xlabel(r"Time / $\mu$s")
plt.ylabel("Amplitude / (%)")
plt.grid(alpha=.5)

ax.annotate("", xy=(55.7, 48), xytext=(58.92, 48),
            arrowprops=dict(arrowstyle="|-|", color='k', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r"$\tau_w$", xy=(55.7,50), xytext=(57.31, 50),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate("Backwall", xy=(58.58, 3.5), xytext=(58, 10),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

plt.tight_layout()
plt.savefig("../figures/ascan_backwall.pdf")
plt.show()