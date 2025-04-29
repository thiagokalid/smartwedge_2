import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from framework import file_m2k
from framework.post_proc import envelope
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline
from pipe_lens_imaging.raytracer import RayTracer
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos
from matplotlib.patches import Circle

#
from pipe_lens_imaging.simulator import Simulator
from pipe_lens_imaging.simulator_utils import dist

matplotlib.use("TkAgg")

# Acoustic lens parameters:
c1 = 6332.93 # in (m/s)
c2 = 1430.00 # in (m/s)q
d = 170e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 91.03e-3 # in (m)

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0)

# Pipeline-related parameters:
radius = 70e-3
wall_width = 20e-3
inner_radius = (radius - wall_width)
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = RayTracer(acoustic_lens, pipeline, transducer, transmission_loss=False, directivity=True)

#%%
# Delay law related parameters:
delta_alpha = np.deg2rad(.5)
alpha_max = pi/4
alpha_min = -pi/4
alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

focus_radius = inner_radius
focus_angle = np.copy(alpha_span)
xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)

tofs_roi, amp = raytracer.solve(xf, zf)
delay_law_focused = tofs_roi.max() - tofs_roi

#%%

sim_parameters = {
    "gate_end": 70e-6,
    "gate_start": 50e-6,
    "fs": 62.5e6, # Hz
    "response_type": "s-scan",
    "emission_delaylaw": delay_law_focused,
    "reception_delaylaw": delay_law_focused
}

chosen_angle = 30
idx = np.argmin(np.power(np.radians(chosen_angle) - alpha_span, 2))

sim = Simulator(sim_parameters, raytracer)
sim.add_reflector(xf[idx:idx + 1], zf[idx:idx + 1], different_instances=True)
sscans = sim.get_response()

sscans_envelope = [envelope(sscans[..., i], axis=0) for i in range(sscans.shape[-1])]
max_sscan = np.max(sscans_envelope)
sscans_normalized = [sscans_envelope[i]/max_sscan for i in range(sscans.shape[-1])]
sscans_log = [np.log10(sscans_normalized[i] + 1e-6) for i in range(sscans.shape[-1])]
sscans_log = np.array(sscans_log)[0, ...]

#%%

filled_marker_style = dict(marker='o', linestyle=':', markersize=20,
                           color='none',
                           markerfacecolor='none',
                           markerfacecoloralt='none',
                           markeredgecolor='red',
                           label="Expected reflector position")


plt.figure()
plt.imshow(np.asarray(sscans_envelope)[0, ...], extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim.tspan[-1] * 1e6, sim.tspan[0] * 1e6], aspect='auto', interpolation='none', vmin=0, vmax=1000)
plt.plot(np.rad2deg(focus_angle[idx]), 63.59, **filled_marker_style)
plt.colorbar()
plt.legend()
plt.show()
#
# plt.figure()
# plt.plot(sscans[1700/2, :, 0])
