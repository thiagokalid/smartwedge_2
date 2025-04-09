import numpy as np

from numpy import cos, sin

from pipe_lens.raytracer import RayTracer
from pipe_lens.transducer import Transducer
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline
from pipe_lens.simulator import Simulator
from pipe_lens.delay_law import FocusedWave
from pipe_lens.imaging_utils import fwhm

# All the variables are express in SI units, e.g., speed in m/s, time in seconds, so on.

#%% User-input fields:

# Acoustic lens design related parameters:
c1 = 6300
c2 = 1480
acoustic_lens = AcousticLens(d=186e-3, c1=c1, c2=c2, tau=74.8e-6)

# Specimen related parameters:
c3 = 5900  # Longitudinal speed
pipeline = Pipeline(outer_radius=70e-3, wall_thickness=20e-3, c=5900)

# Transducer related parameters:
transducer = Transducer(pitch=0.4e-3, num_elem=64, fc=5e6)

# Delay law (beam forming) related parameters:
delay_law = FocusedWave(acoustic_lens, pipeline, transducer)
delay_law.compute(focusing_radius=65e-3, alpha_min=-45, alpha_max=45, delta_alpha=0.5)

## Simulation-related parameters:
# Gate:
gate_start = 0e-6  # s
gate_end = 30e-6  # s
sampling_frequency = 125e6  # in Hz

# Creates the solver for the ray-tracing problem:
solver = RayTracer(acoustic_lens, pipeline, transducer)

# Creates the Spatial Impulse Response simulator:
simulator = Simulator(solver, directivity=True, transmission_loss=True)

#%% Perform N-simulations. Each iteration consider a single Punctual Reflector in a different location.

# locati
baseline_radius = 65e-6
angular_locations = np.arange(0, 45 + 1, 1)
Nsimulations = len(angular_locations)

widths, heights, max_amplitudes = np.zeros(Nsimulations), np.zeros(Nsimulations), np.zeros(Nsimulations)
for n in range(Nsimulations):
    reflector_radius, reflector_angle = baseline_radius, np.deg2rad(angular_locations[n])
    xr, zr = reflector_radius * cos(reflector_angle), reflector_radius * sin(reflector_angle)
    channels = simulator.simulate(xr, zr, delay_law)
    sscan = np.sum(channels, axis=2)
    # widths[n], heights[n], max_amplitudes[n], binary_mask = fwhm(sscan)


