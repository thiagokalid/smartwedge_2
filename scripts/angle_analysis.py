import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pipe_lens.acoustic_lens import AcousticLens
linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})


#%%
c1 = 6332.93 # m/s
c2 = 1430.00 # m/s
d = 170e-3 # mm
tau = 78.97e-3/c1 + 91.03e-3/c2 # seconds

acoustic_lens = AcousticLens(d, c1, c2, tau)

alpha = np.arange(0, np.pi/4, 1e-3)
h = acoustic_lens.h(alpha)
theta = acoustic_lens.pipeline2steering_angle(alpha)


#%% Figure 1:

plt.figure(figsize=(linewidth * .45, 2.5))
plt.plot(np.rad2deg(theta), np.rad2deg(alpha), color='k', linewidth=1.5)
mask = (np.deg2rad(4) <= theta) * (theta <= np.deg2rad(5))
plt.fill_between(np.rad2deg(theta[mask]),
                 np.rad2deg(alpha[mask]), color='lightgray')
plt.fill_betweenx(np.rad2deg(alpha[mask]),
                 np.rad2deg(theta[mask]), color='lightgray')

mask = (np.deg2rad(25.4) <= theta) * (theta <= np.deg2rad(26.4))
plt.fill_between(np.rad2deg(theta[mask]),
                 np.rad2deg(alpha[mask]), color='lightgray')
plt.fill_betweenx(np.rad2deg(alpha[mask]),
                 np.rad2deg(theta[mask]), color='lightgray')

plt.xlabel(r'$\theta$ / (degrees)')
plt.ylabel(r'$\alpha$ / (degrees)')
plt.axis([0, plt.axis()[1], 0, plt.axis()[3]])
plt.xticks(np.arange(0, 30, 5))
plt.yticks(np.arange(0, 50, 10))
plt.ylim([0, 45])
plt.grid(alpha=.2)

plt.tight_layout()
plt.savefig('../figures/theta_alpha.pdf')
plt.show()

#%% Figure 2:

dalpha = np.diff(alpha) / np.diff(theta)
plt.figure(figsize=(linewidth * .45, 2.5))
plt.plot(np.rad2deg(alpha[:-1]), dalpha, color='k', linewidth=1.5)

plt.xlabel(r'$\alpha$ / (degrees)')
plt.ylabel(r'$\mathrm{d\alpha/d\theta}$')
plt.grid(alpha=.2)
plt.xticks(np.arange(0, 60, 15))
plt.xlim([0, 45])

plt.tight_layout()
plt.savefig('../figures/dalpha_dtheta.pdf')
plt.show()
