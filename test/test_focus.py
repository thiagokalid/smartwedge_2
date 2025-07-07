import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos
import matplotlib.ticker as ticker
#
from pipe_lens_imaging.simulator import Simulator

def rotate_point(xy, theta_rad):
    x, y = xy
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return (x_rot, y_rot)

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
pipeline = Pipeline(radius, wall_width, c3, rho_steel, xcenter=-5e-3, zcenter=-5e-3)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer, transmission_loss=False, directivity=True)

arg = (
    0,
    inner_radius + 10e-3,
)
arg = rotate_point(arg, theta_rad=0)

arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

tofs, amps = raytracer.solve(*arg)

sol = raytracer._solve(*arg)

#%% Extract refraction/reflection points:

extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

xlens, zlens = extract_pts(sol, 'xlens'), extract_pts(sol, 'zlens')
xpipe, zpipe = extract_pts(sol, 'xpipe'), extract_pts(sol, 'zpipe')
xf, zf = arg

#%% Debug plots:

firing_angles = extract_pts(sol, 'firing_angle')
pipe_incidence_angles = extract_pts(sol, 'interface_23')

plt.figure(figsize=(8,4))
plt.plot(np.arange(transducer.num_elem), np.degrees(pipe_incidence_angles[0::2]), 'o')
plt.grid()
plt.show()
#%%
plt.figure(figsize=(5, 7))
plt.title("Case A: Focusing on point inside the pipe wall.")
# Plot fix components:
plt.plot(transducer.xt, transducer.zt, 'sk')
plt.plot(0, 0, 'or')
plt.plot(0, acoustic_lens.d, 'or'   )
plt.plot(pipeline.xout, pipeline.zout, 'k')
plt.plot(pipeline.xint, pipeline.zint, 'k')
plt.plot(acoustic_lens.xlens, acoustic_lens.zlens, 'k')
plt.axis("equal")

plt.grid()
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.xlabel("x-axis / (mm)")
plt.ylabel("y-axis / (mm)")

# Plot rays:
for n in range(transducer.num_elem):
    plt.plot(
        [transducer.xt[n], xlens[n], xpipe[n], xf],
        [transducer.zt[n], zlens[n], zpipe[n], zf],
        linewidth=.5, color='lime', zorder=1
    )
plt.plot(xf, zf, 'xr', label='Focus')
plt.legend()
plt.tight_layout()
plt.ylim(-5e-3, acoustic_lens.d + 5e-3)
plt.show()