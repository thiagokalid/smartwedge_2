import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline, pol2cart

import matplotlib
matplotlib.use('TkAgg')

#
c1 = 6300 # m/s
c2 = 1480 # m/s
d = 186e-3 # mm
tau = 74.80e-6 # us

acoustic_lens = AcousticLens(d, c1, c2, tau)

#
delta_ang = 1e-1
angs = np.deg2rad(np.arange(-45, 45 + delta_ang, delta_ang))

#
z = acoustic_lens.z(angs)
xlens, zlens = pol2cart(z, angs)

#
radius = 50e-3
wall_width = 10e-3
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3)

# Plotting the results:
fig, ax = plt.subplots(figsize=(6, 10))
ax.plot(xlens, zlens, color='k', linewidth=3, label="Acoustic Lens")
ax.plot(pipeline.xout, pipeline.zout, color='b', linewidth=2, label="Outer Surface")
ax.plot(pipeline.xint, pipeline.zint, color='b', linewidth=2, label="Inner Surface")
ax.plot(0, d, 's', color='k', label="Transducer")
ax.plot(0, 0, 'o', color='r', label="Pipeline center")
ax.plot([0, 0], [0, d], '--', linewidth=1, color='k')
ax.plot([0, xlens[-1]], [0, zlens[-1]], '--', linewidth=1, color='k')
ax.plot([0, xlens[-1]], [d, zlens[-1]], '--', linewidth=1, color='k')
# ax.set_xlabel("x-axis / [mm]")
# ax.set_ylabel("z-axis / [mm]")
ax.set_ylim([-1.1e-3, d*1.1])
ax.axis("off")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e3:.2f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e3:.2f}"))
ax.set_aspect("equal")
plt.show()