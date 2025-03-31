import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from framework import file_m2k
from framework.post_proc import envelope
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline
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
acoustic_lens_type = "xl"
plot_fig = True # Chooses to either plot or save the output.

#%%
data = file_m2k.read(root + f"only_tube.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

#
time_grid = data.time_grid
c1 = 6332.93 # m/s
c2 = 1430.00 # m/s
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

# ROI
# plt.plot(np.rad2deg(angs), 2 * tof_front_wall + time_offset, "--", color='black', linewidth=2)
# plt.plot(np.rad2deg(angs), 2 * tof_back_wall + time_offset, "--", color='black', linewidth=2)

log_cte = 1e-6
channels = data.ascan_data[..., 0]
sscan = np.sum(channels, axis=2)
sscan_envelope = envelope(sscan, axis=0)
sscan_envelope = (sscan_envelope - sscan_envelope.min()) / (sscan_envelope.max() - sscan_envelope.min())
sscan_db = np.log10(sscan_envelope + log_cte)


# Lens multiple reflections:
delta_ang = 1e-1
angs_restricted = np.deg2rad(np.arange(-9, 9 + delta_ang, delta_ang))
xlens, zlens = acoustic_lens.h(angs_restricted) * np.sin(angs_restricted), acoustic_lens.h(angs_restricted) * np.cos(angs_restricted)
lens_xz_restricted = np.vstack([xlens, zlens]).T
tof_lens_restricted = np.linalg.norm(lens_xz_restricted - np.array([0, acoustic_lens.d]), axis=1) / acoustic_lens.c1


fig, ax = plt.subplots(figsize=(linewidth * .5, 3))
plt.imshow(sscan_db, extent=[np.rad2deg(angs[0]), np.rad2deg(angs[-1]), time_grid[-1] * 1e-6, time_grid[0] * 1e-6],
           cmap='YlGnBu', aspect='auto', interpolation='None', vmin=-6, vmax=0)
plt.colorbar()

plt.ylim([70e-6, 20e-6])
plt.xticks(np.arange(-40, 40 + 20, 20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim([-45, 45])
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
plt.grid(which="major", alpha=.25, color='k')
plt.grid(axis='x', which="minor", alpha=.2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e6:.0f}"))

# Annotations:
ax.annotate(r'$\tau_{lens}(\alpha,1)$',
            xy=(5.9, 25e-6),
            xytext=(17, 34e-6),
            color="black",
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'${\tau_{lens}(\alpha,2)}$',
            xy=(5.9, 49e-6),
            xytext=(17, 44e-6),
            color="black",
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom",  # Position text below arrow
            weight='bold',
            style='italic'
            )
ax.annotate(r'${\tau_{lens}(\alpha,3)}$',
            xy=(5.9, 74e-6),
            xytext=(17, 69e-6),
            color="black",
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom",  # Position text below arrow
            weight='bold',
            style='italic'
            )

ax.annotate(r'${\tau_{front}}$',
            xy=(-20, 56e-6),
            xytext=(-40 + 11, 51e-6),
            color="black",
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1.0),
            ha="center",  # Center text horizontally
            va="bottom",  # Position text below arrow
            weight='bold',
            style='italic'
            )

ax.annotate(r'${\tau_{back}}$',
            xy=(-20, 63e-6),
            xytext=(-40 + 11, 70e-6),
            color='black',
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1.0),
            ha="center",  # Center text horizontally
            va="bottom",  # Position text below arrow
            weight='bold',
            style='italic'
            )


plt.tight_layout()
plt.savefig("../figures/sscan_echoes_avoidance.pdf")
plt.show()