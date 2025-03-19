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



#
c1 = 6300 # m/s
c2 = 1480 # m/s
d = 170e-3 # mm
tau = 78.97e-3/c1 + 91.03e-3/c2 # seconds

acoustic_lens = AcousticLens(d, c1, c2, tau)

#
radius = 70e-3
wall_width = 20e-3
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3)


#
delta_ang = 1e-1
angs = np.deg2rad(np.arange(-45, 45 + delta_ang, delta_ang))

xlens, zlens = acoustic_lens.h(angs) * np.sin(angs), acoustic_lens.h(angs) * np.cos(angs)
lens_xz = np.vstack([xlens, zlens]).T

tof_lens = np.linalg.norm(lens_xz - np.array([0, acoustic_lens.d]), axis=1) / acoustic_lens.c1
tof_front_wall = tof_lens + (acoustic_lens.h(angs) - pipeline.outer_radius) / acoustic_lens.c2
tof_back_wall = tof_front_wall + pipeline.wall_width / pipeline.c

#
linewidth = 6.3091141732
fig, ax = plt.subplots(figsize=(linewidth * .45, 3))

# ROI
plt.plot(np.rad2deg(angs), 2 * tof_front_wall, "r", linewidth=2)
plt.plot(np.rad2deg(angs), 2 * tof_back_wall, "r", linewidth=2)

rectangle = plt.Rectangle((-45, 2*tof_back_wall[0]), 90, -2*(tof_back_wall[0]-tof_front_wall[0]), fc='red', ec='red', alpha=.25)
plt.gca().add_patch(rectangle)

# Lens multiple reflections:
delta_ang = 1e-1
angs_restricted = np.deg2rad(np.arange(-9, 9 + delta_ang, delta_ang))
xlens, zlens = acoustic_lens.h(angs_restricted) * np.sin(angs_restricted), acoustic_lens.h(angs_restricted) * np.cos(angs_restricted)
lens_xz_restricted = np.vstack([xlens, zlens]).T
tof_lens_restricted = np.linalg.norm(lens_xz_restricted - np.array([0, acoustic_lens.d]), axis=1) / acoustic_lens.c1
for k in range(1, 4):
    plt.plot(np.rad2deg(angs_restricted), 2 * k * tof_lens_restricted, "k", linewidth=3)


plt.ylim([85e-6, 10e-6])
plt.xticks(np.arange(-40, 40 + 20, 20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim([-45, 45])
plt.xlabel(r"$\alpha$-axis / [degree]")
plt.ylabel(r"Time / [$\mu$s]")
plt.grid(which="major", alpha=.5)
plt.grid(axis='x', which="minor", alpha=.2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e6:.0f}"))

# Annotations:
ax.annotate(r'$\tau_{lens}(\alpha,1)$',
            xy=(5.9, 25e-6),
            xytext=(17, 19e-6),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{lens}(\alpha,2)$',
            xy=(5.9, 50e-6),
            xytext=(17, 44e-6),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{lens}(\alpha,3)$',
            xy=(5.9, 76e-6),
            xytext=(17, 70e-6),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate(r'$\tau_{front}$',
            xy=(-40, 53e-6),
            xytext=(-40 + 11, 47e-6),
            arrowprops=dict(arrowstyle="-|>", color='red', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate(r'$\tau_{back}$',
            xy=(-40, 62e-6),
            xytext=(-40 + 11, 73e-6),
            arrowprops=dict(arrowstyle="-|>", color='red', alpha=1, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )


plt.tight_layout()
plt.savefig("../figures/echoes_avoidance_computation.pdf")
plt.show()