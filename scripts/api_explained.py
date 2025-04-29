import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pipe_lens.imaging_utils import convert_time2radius
from framework import file_m2k
from framework.post_proc import envelope
from bisect import bisect
import scipy
from pipe_lens_imaging.utils import api_func_polar

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
root = '../data/echoes/'
data = file_m2k.read(root + f"pit_inspection.m2k",
                     freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

log_cte = 1e-6
theoretical_maximum = 2 ** 16 * data.probe_params.num_elem
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)

# It was manually identified where  the front and back wall was (in microseconds):
t_outer, t_inner = 55.27, 60.75
rtop, rbottom = 62.0, 58.0

r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)


channels = data.ascan_data[..., 0]
sscan = np.sum(channels, axis=2)
sscan_envelope = envelope(sscan, axis=0)
sscan_envelope = (sscan_envelope - sscan_envelope.min()) / (sscan_envelope.max() - sscan_envelope.min())
sscan_db = np.log10(sscan_envelope + log_cte)

tbeg, tend = bisect(time_grid, 58.5), bisect(time_grid, 60.5)
ang_beg, ang_end = bisect(ang_span, -10), bisect(ang_span, 20)
sscan_windowed = sscan_envelope[tbeg:tend, ang_beg:ang_end]

corners = [(rtop, -10), (rbottom, 10)]
api, maxAbsoluteLocation, pixels_above_threshold, maximumAmplitude, estimated_width, estimated_height = api_func_polar(sscan, r_span, ang_span, corners, drawSquare=False)

#%%

plt.figure(figsize=(linewidth*.48, 2))
plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='YlGnBu', aspect='auto', interpolation='None', vmin=-6, vmax=-2)
ytemp = np.arange(40, 65, 1)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
plt.ylim([60.5, 58.5])
plt.xlim([-10, 20])
plt.colorbar()

plt.tight_layout()
plt.savefig("../figures/api_sscan.pdf")
plt.show()

#%%

fig, ax = plt.subplots(figsize=(linewidth*.48, 2))
cm = plt.imshow(pixels_above_threshold, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='binary', aspect='auto', interpolation='None')
ytemp = np.arange(40, 65, 1)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
plt.ylim([60.5, 58.5])
plt.xlim([-10, 20])

fig.colorbar(cm, orientation='vertical', shrink=1)


plt.tight_layout()
plt.savefig("../figures/api_binary_sscan.pdf")
plt.show()
