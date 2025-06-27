import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from framework import file_m2k
from framework.post_proc import envelope
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos

import gc
#
from pipe_lens_imaging.simulator import Simulator
from pipe_lens_imaging.simulator_utils import dist

linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

# Acoustic lens parameters:
c1 = 6332.93 # in (m/s)
c2 = 1430.00 # in (m/s)q
d = 170e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 91.03e-3 + 1e-3 # in (m)
rho_aluminium = 2.710 #kg / m
rho_water = 1.000
rho_steel = 7.850

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water)

# Pipeline-related parameters:
radius = 139.82e-3/2
wall_width = 16.23e-3
inner_radius = (radius - wall_width)
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3, rho_steel)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

#%%
# Delay law related parameters:
delta_alpha = np.deg2rad(.5)
alpha_max = pi/4
alpha_min = -pi/4
alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

focus_angle = np.copy(alpha_span)
focus_radius = inner_radius + 10e-3

chosen_angle = [0.5, 15.5, 29.5]
idxs = [np.where(np.abs(np.degrees(focus_angle) - float(chosen_angle[i])) < 1e-6)[0][0] for i in range(len(chosen_angle))]

#%%
root = "../data/api/"
data_insp = file_m2k.read(root + f"active_dir_xl_focused_1degree_v1.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
data_ref = file_m2k.read(root + f"ref_xl_focused.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
t_span = data_insp.time_grid

sscans_envelope = envelope(np.sum(data_insp.ascan_data - np.mean(data_ref.ascan_data, axis=3)[..., np.newaxis], axis=2), axis=0)
sscans_envelope /= 218990.3

del data_insp, data_ref
gc.collect()

#%%

ang_positions = np.arange(-1.5, 45 * 1, 1)
idxs_positions = [np.where(np.abs(ang_positions - float(chosen_angle[i])) < 1e-6)[0] for i in range(len(chosen_angle))]

filled_marker_style = dict(marker='o', linestyle=':', markersize=20,
                           color='none',
                           markerfacecolor='none',
                           markerfacecoloralt='none',
                           markeredgecolor='red',
                           label="Expected reflector position")


ncols = len(idxs)
fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=ncols)
for i in range(len(idxs)):
    plt.subplot(1, ncols, i + 1)

    # plt.imshow(sscans_log[0], extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim.tspan[-1] * 1e6, sim.tspan[0] * 1e6],
    #            aspect='auto', interpolation='none', vmin=-6, vmax=0, cmap='jet')

    #
    # sscans_envelope[:489, :, :] = 0
    # sscans_envelope[948:, :, :] = 0

    im = plt.imshow(sscans_envelope[..., idxs_positions[i]],
                    extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), t_span[-1], t_span[0]],
                    aspect='auto', interpolation='none', cmap='jet', vmin=0, vmax=1)
    # TOFs of interest:
    outer_surf_time = ((d - h0) / c1 + (h0 - pipeline.outer_radius) / c2) * 2 * 1e6
    inner_surf_time = outer_surf_time + 2 * pipeline.wall_width / c3 * 1e6
    scatterer_time = outer_surf_time + 2 * 1e6 * (pipeline.outer_radius - focus_radius) / c3

    # Plots:
    plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * outer_surf_time, color='r', linewidth=3)
    plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time, color='r', linewidth=3)
    # plt.plot(np.rad2deg(focus_angle[idxs[i]]), scatterer_time, **filled_marker_style)
    plt.ylim([64.2, 53])
    plt.grid(alpha=.25)
    plt.xlabel(r"$\alpha$-axis / (degrees)")
    plt.ylabel(r"time-axis / ($\mathrm{\mu s}$)")
    plt.xticks(np.arange(-45, 45 + 15, 15))

    if i + 1 == 1:
        plt.suptitle("SDH experiment")
        # plt.colorbar()
        # plt.legend()
        cax = plt.gca().inset_axes([0.15, .125, 0.7, .025])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=.7)
        cb.ax.tick_params(axis='x', colors='white')  # set colorbar tick color
        cb.outline.set_edgecolor('white')  # set colorbar edgecolor
        cb.ax.xaxis.label.set_color('white')  # set colorbar label color
        for t in cb.ax.get_xticklabels():  # set colorbar tick labels
            t.set_color('white')

plt.tight_layout()
plt.savefig("../figures/sdh_experiment.pdf")
plt.show()
