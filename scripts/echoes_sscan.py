from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi

from matplotlib.ticker import FuncFormatter, MultipleLocator, AutoMinorLocator
from matplotlib.patches import Polygon
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline, pol2cart

import matplotlib
matplotlib.use('TkAgg')
font = {
    'weight' : 'normal',
    'size'   : 9
}


# Set the default font to DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rc('font', **font)

from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/echoes/'
acoustic_lens_type = "xl"
plot_fig = True # Chooses to either plot or save the output.

#%%
data = file_m2k.read(root + f"only_tube.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

#
time_grid = data.time_grid
c1 = 6300 # m/s
c2 = 1480 # m/s
d = 170e-3 # mm
tau = 78.97e-3/c1 + 91.03e-3/c2 # seconds

acoustic_lens = AcousticLens(d, c1, c2, tau)

#
radius = 70e-3
wall_width = 18e-3
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3)


#
time_offset = 2.25e-6
delta_ang = 1e-1
angs = np.deg2rad(np.arange(-45, 45 + delta_ang, delta_ang))

xlens, zlens = acoustic_lens.h(angs) * np.sin(angs), acoustic_lens.h(angs) * np.cos(angs)
lens_xz = np.vstack([xlens, zlens]).T

tof_lens = np.linalg.norm(lens_xz - np.array([0, acoustic_lens.d]), axis=1) / acoustic_lens.c1
tof_front_wall = tof_lens + (acoustic_lens.h(angs) - pipeline.outer_radius) / acoustic_lens.c2
tof_back_wall = tof_front_wall + pipeline.wall_width / pipeline.c

#
linewidth = 6.3091141732
fig, ax = plt.subplots(figsize=(linewidth * .5, 3))

# ROI
# plt.plot(np.rad2deg(angs), 2 * tof_front_wall + time_offset, "--", color='lime', linewidth=2)
# plt.plot(np.rad2deg(angs), 2 * tof_back_wall + time_offset, "--", color='lime', linewidth=2)

log_cte = 1e-6
channels = data.ascan_data[..., 0]
channels_envelope = envelope(channels, axis=0)
sscan = np.sum(channels_envelope, axis=2)
sscan_db = 20 * np.log10(sscan + log_cte)


# Lens multiple reflections:
delta_ang = 1e-1
angs_restricted = np.deg2rad(np.arange(-9, 9 + delta_ang, delta_ang))
xlens, zlens = acoustic_lens.h(angs_restricted) * np.sin(angs_restricted), acoustic_lens.h(angs_restricted) * np.cos(angs_restricted)
lens_xz_restricted = np.vstack([xlens, zlens]).T
tof_lens_restricted = np.linalg.norm(lens_xz_restricted - np.array([0, acoustic_lens.d]), axis=1) / acoustic_lens.c1

plt.imshow(sscan_db, extent=[np.rad2deg(angs[0]), np.rad2deg(angs[-1]), time_grid[-1] * 1e-6, time_grid[0] * 1e-6],
           cmap='inferno', aspect='auto', interpolation='None', vmin=30, vmax=120)
plt.colorbar()

plt.ylim([80e-6, 40e-6])
plt.xticks(np.arange(-40, 40 + 20, 20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim([-45, 45])
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / ($\mu$s)")
plt.grid(which="major", alpha=.5)
plt.grid(axis='x', which="minor", alpha=.2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e6:.0f}"))

# Annotations:
ax.annotate(r'$\tau_{lens}(\alpha,1)$',
            xy=(5.9, 25e-6),
            xytext=(17, 19e-6),
            color="lime",
            arrowprops=dict(arrowstyle="-|>", color='lime', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{lens}(\alpha,2)$',
            xy=(5.9, 50e-6),
            xytext=(17, 44e-6),
    color="lime",
            arrowprops=dict(arrowstyle="-|>", color='lime', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{lens}(\alpha,3)$',
            xy=(5.9, 76e-6),
            xytext=(17, 70e-6),
            color="lime",
            arrowprops=dict(arrowstyle="-|>", color='lime', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate(r'$\tau_{front}$',
            xy=(-40, 55e-6),
            xytext=(-40 + 11, 49e-6),
            color="lime",
            arrowprops=dict(arrowstyle="-|>", color='lime', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate(r'$\tau_{back}$',
            xy=(-40, 62e-6),
            xytext=(-40 + 11, 73e-6),
            color='lime',
            arrowprops=dict(arrowstyle="-|>", color='lime', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )


plt.tight_layout()
plt.savefig("../figures/sscan_echoes_avoidance.pdf")
plt.show()