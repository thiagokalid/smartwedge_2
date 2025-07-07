import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos
import matplotlib.ticker as ticker
from pipe_lens_imaging.reflection_raytracer import ReflectionRayTracer
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.simulator import Simulator
from framework.post_proc import envelope
from bisect import bisect
from tqdm import tqdm

linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

#
# chosen_angle_grid = [0, 10, 20, 30]
chosen_angle_grid = [30]
zcenter_grid = [-11e-3]
z_step = 2.5/4
# zcenter_grid = np.arange(-10, 10 + z_step, z_step) * 1e-3
# zcenter_grid = [10e-3]
residual_thickness_matrix = np.zeros((len(chosen_angle_grid), len(zcenter_grid)))

for _aa, chosen_angle in enumerate(tqdm(chosen_angle_grid)):
    for _zz, zcenter in enumerate(zcenter_grid):
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
        pipeline = Pipeline(radius, wall_width, c3, rho_steel, xcenter=0, zcenter=0)
        shifted_pipeline = Pipeline(radius, wall_width, c3, rho_steel, xcenter=0, zcenter=zcenter)

        # Ultrasound phased array transducer specs:
        transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
        transducer.zt += acoustic_lens.d


        # Raytracer engine to find time of flight between emitter and focus:
        raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)


        # Raytracer engine to find time of flight reflection:
        raytracer2 = ReflectionRayTracer(acoustic_lens, shifted_pipeline, transducer, transmission_loss=True, directivity=True)
        #

        raytracer3 = FocusRayTracer(acoustic_lens, shifted_pipeline, transducer, transmission_loss=True, directivity=True)

        #%%
        # Delay law related parameters:
        delta_alpha = np.deg2rad(.5)
        alpha_max = pi/4
        alpha_min = -pi/4
        alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

        focus_radius = inner_radius + (wall_width - 8.99e-3)
        focus_angle = np.copy(alpha_span)
        xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)



        #%%
        tofs_roi, _ = raytracer.solve(xf, zf)
        delay_law_focused = tofs_roi[transducer.num_elem // 2, :] - tofs_roi
        configs = {
            "surface_echoes": True,
            "gate_end": 80e-6,
            "gate_start": 30e-6,
            "fs": 64.5e6, # Hz
            "response_type": "s-scan",
            "emission_delaylaw": delay_law_focused,
            "reception_delaylaw": delay_law_focused
        }

        # Extract the reflector at desired position
        chosen_angle_list = [chosen_angle]
        idxs = [np.where(np.abs(np.degrees(focus_angle) - float(chosen_angle_list[i])) < 1e-6)[0][0] for i in range(len(chosen_angle_list))]
        x_reflector, z_reflector = xf[idxs], zf[idxs]

        # WIthout displacement:
        sim1 = Simulator(configs, [raytracer2], verbose=False)
        sim1.add_reflector(x_reflector, z_reflector + shifted_pipeline.zcenter, different_instances=True)
        sscan = sim1.get_response()

        sscan_env = envelope(sscan, axis=0)
        sscan_log = np.log10(sscan_env + 1e-6)

        #%%
        time_shift = 2 * (shifted_pipeline.zcenter/c2) * -1e6

        filled_marker_style = dict(marker='o', linestyle=':', markersize=20,
                                   color='none',
                                   markerfacecolor='none',
                                   markerfacecoloralt='none',
                                   markeredgecolor='red',
                                   label="Expected reflector position")


        i = 0

        ncols = sscan.shape[-1]
        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=ncols)

        plt.subplot(1, ncols, i + 1)


        im = plt.imshow(sscan_env,
                        extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim1.tspan[-1] * 1e6, sim1.tspan[0] * 1e6],
                        aspect='auto', interpolation='none', cmap='jet')

        # TOFs of interest:
        outer_surf_time = ((d - h0) / c1 + (h0 - pipeline.outer_radius) / c2) * 2 * 1e6 + time_shift
        inner_surf_time = outer_surf_time + 2 * pipeline.wall_width / c3 * 1e6
        scatterer_time = outer_surf_time + 2 * 1e6 * (pipeline.outer_radius - focus_radius) / c3

        # Plots:
        plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * outer_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        plt.plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        plt.plot(np.rad2deg(focus_angle[idxs[0]]), scatterer_time, **filled_marker_style)

        plt.ylim([65 + time_shift, 50 + time_shift])

        plt.grid(alpha=.25)
        plt.xlabel(r"$\alpha$-axis / (degrees)")
        plt.ylabel(r"time-axis / ($\mathrm{\mu s}$)")
        plt.xticks(np.arange(-45, 45 + 15, 15))

        if i + 1 == 1:
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

        max_per_column = np.argmax(sscan_env, axis=0)
        t_max_per_column = sim1.tspan[max_per_column]
        plt.plot(np.degrees(alpha_span), t_max_per_column * 1e6, color='magenta')
        plt.tight_layout()
        plt.savefig("../figures/sscan_surface.png", dpi=300)
        plt.show()

        #%%
        configs = {
            "surface_echoes": False,
            "gate_end": 80e-6,
            "gate_start": 30e-6,
            "fs": 64.5e6,  # Hz
            "response_type": "s-scan",
            "emission_delaylaw": delay_law_focused,
            "reception_delaylaw": delay_law_focused
        }
        # Without surface:
        sim2 = Simulator(configs, [raytracer3], verbose=False)
        sim2.add_reflector(x_reflector, z_reflector + shifted_pipeline.zcenter, different_instances=True)
        sscan = sim2.get_response()
        sscan_env = envelope(sscan, axis=0)

        t_flaw_idx, theta_flaw_idx, _ = np.where(sscan_env == sscan_env.max())
        theta_flaw, t_flaw = alpha_span[theta_flaw_idx], sim2.tspan[t_flaw_idx]


        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=ncols)
        im = plt.imshow(sscan_env,
                        extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim1.tspan[-1] * 1e6, sim1.tspan[0] * 1e6],
                        aspect='auto', interpolation='none', cmap='jet')
        plt.plot(np.degrees(theta_flaw), t_flaw * 1e6, 'xr')
        plt.ylim([65 + time_shift, 50 + time_shift])
        plt.grid(alpha=.25)
        plt.xlabel(r"$\alpha$-axis / (degrees)")
        plt.ylabel(r"time-axis / ($\mathrm{\mu s}$)")
        plt.xticks(np.arange(-45, 45 + 15, 15))
        plt.tight_layout()
        plt.savefig("../figures/sscan_flaw.png", dpi=300)

        #%% Residual thickness:

        delta_tof = t_flaw - t_max_per_column[bisect(alpha_span, theta_flaw)]
        residual_thickness = .5 * delta_tof * c3 * 1e3
        print(f"Residual thickness = {residual_thickness[0]:.4f} mm")
        residual_thickness_matrix[_aa, _zz] = residual_thickness


#%%

if len(zcenter_grid) != 0 or len(chosen_angle_grid) != 0:
    plt.figure(figsize=(7,3.4))

    for i, ang in enumerate(chosen_angle_grid):
        plt.plot(zcenter_grid * 1e3, residual_thickness_matrix[i, :], 'o--', label=f"SDH at ${ang:.1f}^\circ$.")

    zgrid = np.arange(zcenter_grid[0] - 15e-3, zcenter_grid[-1] + 15e-3, 1e-3)
    plt.plot(zgrid * 1e3, 8.99 * np.ones_like(zgrid), 'k--', label="Ideal")
    plt.xlabel("Tube center shift along z-axis / (mm)")
    plt.ylabel("Residual thickness / (mm)")
    plt.xlim([zcenter_grid[0] * 1e3 - 2.5, zcenter_grid[-1] * 1e3 + 2.5] )
    plt.ylim([7, 10])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figures/zshift_influence.png", dpi=300)
    plt.show()
