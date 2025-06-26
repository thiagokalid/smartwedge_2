import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from framework import file_m2k
from framework.post_proc import envelope
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens_imaging.raytracer import RayTracer
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos

#
from pipe_lens_imaging.simulator import Simulator

linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})


filled_marker_style = dict(marker='o', linestyle=':', markersize=20,
                           color='none',
                           markerfacecolor='none',
                           markerfacecoloralt='none',
                           markeredgecolor='red',
                           label="Expected reflector position")

# Acoustic lens parameters:
c1 = 6460.00 # in (m/s)
c2 = 1430.00 # in (m/s)q
d = 2 * 85.e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 85.06e-3 # in (m)
rho_aluminium = 2.710 #kg / m
rho_water = 1.000
rho_steel = 7.850

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water)

# Pipeline-related parameters:
radius = 67.15e-3
inner_radius = 47.15e-3
wall_width = radius - inner_radius
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3, rho_steel, z_center=0.09e-3)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = RayTracer(acoustic_lens, pipeline, transducer, transmission_loss=True, reflection_loss=True, directivity=True)

#%%

# Delay law related parameters:
delta_alpha = np.deg2rad(.5)
alpha_max = pi/4
alpha_min = -pi/4
alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

focus_radius = inner_radius
focus_angle = np.copy(alpha_span)
xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)


#%%

outer_surf_time = ((d - h0) / c1 + (h0 - pipeline.outer_radius) / c2) * 2


#%%
tofs_roi, _ = raytracer.solve(xf, zf)
delay_law_focused = tofs_roi[transducer.num_elem // 2, :] - tofs_roi
configs = {
    "surface_echoes": True,
    "gate_end": 70e-6,
    "gate_start": 40e-6,
    "fs": 32.25e6 * 2, # Hz
    "response_type": "s-scan",
    "emission_delaylaw": delay_law_focused,
    "reception_delaylaw": delay_law_focused
}


chosen_angle = [25]

ang_ideal_law, thickness_ideal_law = list(), list()
ang_non_ideal_law, thickness_non_ideal_law = list(), list()


# reflector_radii = inner_radius + 0.44209488888888515e-3
reflector_radii = inner_radius + 5e-3
x_reflector, z_reflector = reflector_radii * sin(np.radians(chosen_angle)), reflector_radii * cos(np.radians(chosen_angle))

# Without displacement:
sim1 = Simulator(configs, raytracer)
sim1.add_reflector(x_reflector, z_reflector, different_instances=True)

sscan = sim1.get_response()

sscans_envelope = envelope(sscan[..., 0], axis=0)

sscans_log = np.log10(sscans_envelope/sscans_envelope.max() + 1e-6)

#%% Plot:
time_shift = 0

fig, ax = plt.subplots(figsize=(17, 5), nrows=1, ncols=1)
plt.subplot(1, 1, 1)
im = plt.imshow(sscans_log,
            extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim1.tspan[-1] * 1e6, sim1.tspan[0] * 1e6],
            aspect='auto', interpolation='none', vmin=-6, vmax=0, cmap='jet')


# TOFs of interest:

inner_surf_time = (outer_surf_time + time_shift) + 2 * pipeline.wall_width / c3
scatterer_time = (outer_surf_time + time_shift) + 2 * (pipeline.wall_width - 5e-3) / c3

# Plots:
plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * (outer_surf_time + time_shift) * 1e6, color='r', linewidth=3)
plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time * 1e6, color='r', linewidth=3)
plt.plot(chosen_angle, scatterer_time * 1e6, **filled_marker_style)

plt.ylim([64.2, 45])

plt.grid(alpha=.25)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"time-axis / ($\mathrm{\mu s}$)")
plt.xticks(np.arange(-45, 45 + 15, 15))


plt.suptitle("Spatial Impulse Response simulation")
# plt.colorbar()
plt.legend()
cax = plt.gca().inset_axes([0.15, .125, 0.7, .025])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=.7)
cb.ax.tick_params(axis='x', colors='white')  # set colorbar tick color
cb.outline.set_edgecolor('white')  # set colorbar edgecolor
cb.ax.xaxis.label.set_color('white')  # set colorbar label color
for t in cb.ax.get_xticklabels():  # set colorbar tick labels
    t.set_color('white')
plt.tight_layout()
plt.show()