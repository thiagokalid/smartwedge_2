import numpy as np
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from pipe_lens_imaging import file_law
from numpy import sin, cos
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pipe_lens_imaging.utils import convert_parameters

LAW_PATH = "../data/laws/"

temperatures = [25] # Celsius
zshift_span = [0]
delta_h0_span = [0]
pitch_span = [.5e-3]
delta_radius_span = [0]

total = len(temperatures) * len(zshift_span) * len(delta_h0_span) * len(pitch_span) * len(delta_radius_span)

transd_type = "olympus"


for T, zshift, delta_h0, pitch, delta_radius in tqdm(product(temperatures, zshift_span, delta_h0_span, pitch_span, delta_radius_span), total=total):
    # Acoustic lens parameters:
    c1_project = 6370.878
    c2_project = 1483.000
    c2 = c2_project
    c3_project = 5900
    d = 186.00000e-3  # in (m)
    alpha_0 = 0  # in (rad)
    h0, z0, max_alpha = convert_parameters(d/2, 14e-3, 70e-3, c1_project, c2)
    alpha_max = np.radians(45)
    rho_aluminium = 2.710  # kg / m
    rho_water = 1.000
    rho_steel = 7.850

    acoustic_lens = AcousticLens(c1_project, c2_project, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water)


    # Pipeline-related parameters:
    radius = 70e-3
    wall_width = 20e-3
    inner_radius = (radius - wall_width)

    pipeline = Pipeline(radius, wall_width, c3_project, rho_steel, zcenter=zshift, xcenter=0)

    # Ultrasound phased array transducer specs:
    transducer = Transducer(pitch=pitch, bw=.4, num_elem=64, fc=5e6)
    transducer.zt += acoustic_lens.d

    # Raytracer engine to find time of flight between emitter and focus:
    raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer)


    # match T:
    #     case 15:
    #         c2 = 1400
    #     case 20:
    #         c2 = 1430
    #     case 25:
    #         c2 = 1483
    #     case 30:
    #         c2 = 1500
    # c2 = compute_cl_water(T)
    # raytracer.set_speeds(c1_project, c2, c3_project) # non-project propagation speeds

    # %% Delay law related parameters:
    delta_alpha = np.deg2rad(.5)
    alpha_max = np.radians(45)
    alpha_min = -alpha_max
    alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

    focus_radius = inner_radius
    focus_angle = np.copy(alpha_span)
    xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)

    tofs_roi, amp = raytracer.solve(xf, zf)
    delay_law = tofs_roi[transducer.num_elem // 2, :] - tofs_roi
    delay_law = delay_law - delay_law.min() # Ensure positive delays

    # Plot setup:
    plt.figure(figsize=(8, 8))
    plt.plot(xf, zf, 'o', color='orange', markersize=2)
    plt.plot(transducer.xt, transducer.zt, 's', color='k', markersize=2)
    plt.plot(acoustic_lens.xlens, acoustic_lens.zlens, '-', color='k')
    plt.plot(pipeline.xout, pipeline.zout, '-', color='k')
    plt.plot(pipeline.xint, pipeline.zint, '-', color='k')
    plt.axis("equal")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, y: f'{x * 1000:.1f}'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, y: f'{x * 1000:.1f}'))
    plt.ylim([0, d + 5e-3])
    plt.show()

    mirror_law = np.load("../temp/law_LL/time_emission.npy")

    full_law = np.vstack([mirror_law[:90], delay_law.T, mirror_law[90:]])

    SAVE_LAW = True
    if SAVE_LAW:
        file_law.write(
            LAW_PATH + f"T_{T}_zshift_{zshift * 1e3:.2f}_delta_radius_{delta_radius * 1e3:.2f}_cwater_{c2:.2f}",
            emission=delay_law.T,
            reception=delay_law.T,
            delay_time_unit="s"
        )

