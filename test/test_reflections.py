import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens_imaging.transducer import Transducer
from numpy import pi, sin, cos
import matplotlib.ticker as ticker
from pipe_lens_imaging.reflection_raytracer import ReflectionRayTracer
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.simulator import Simulator
from framework.post_proc import envelope
from bisect import bisect
from tqdm import tqdm
from pipe_lens_imaging.ultrasound import compute_cl_aluminum, compute_cl_steel, compute_cl_water
from pipe_lens_imaging.utils import convert_parameters

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
chosen_angle_grid = [-28]
# ang_lim = 1 / 2 * 60 / (2 * np.pi * 51.55) * 360
# chosen_angle_grid = np.linspace(-ang_lim, ang_lim,
# chosen_angle_grid = [i * 10 * 360/(2 * np.pi * 60.6) for i in range(1, 4)]
# zcenter_grid = [0]
# z_step = 2.5/4
# zcenter_grid = np.arange(-10, 10 + z_step, z_step) * 1e-3
# zcenter_grid = np.arange(-10e-3, -5e-3, 1e-3)
zcenter_grid = [0]
residual_thickness_matrix = np.zeros((len(chosen_angle_grid), len(zcenter_grid)))
theta_flaw_matrix = np.zeros((len(chosen_angle_grid), len(zcenter_grid)))

for _aa, chosen_angle in enumerate(tqdm(chosen_angle_grid)):
    for _zz, zcenter in enumerate(zcenter_grid):

        # Parameters for computi
        c1 = 6332.934
        c2_project = 1483
        c2_law = 1483
        c2_inspection = 1483

        #
        # zshift_insp = -2.25e-3
        # zshift_insp = -4e-3
        zshift_insp = 0
        zshift_law = 0e-3
        x_transd_shift = 0

        c3 = 5900
        d = 170e-3  # in (m)
        alpha_max = pi / 4  # in (rad)
        alpha_0 = 0  # in (rad)
        h0 = 91.03396266259846e-3  # in (m)
        rho_aluminium = 2.710  # kg / m
        rho_water = 1.000
        rho_steel = 7.850

        acoustic_lens = AcousticLens(c1, c2_project, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water)

        # Pipeline-related parameters:
        radius_insp = 142e-3/2
        # radius_insp = 70e-3
        radius_law = 70e-3
        wall_width = 18.84e-3
        inner_radius = (radius_insp - wall_width)


        pipeline_law = Pipeline(radius_law, wall_width, c3, rho_steel, xcenter=0, zcenter=zshift_law)
        pipeline_insp = Pipeline(radius_insp, wall_width, c3, rho_steel, xcenter=0, zcenter=zshift_insp)

        # Ultrasound phased array transducer specs:
        transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
        transducer.zt += acoustic_lens.d

        transducer_insp = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
        transducer_insp.zt += acoustic_lens.d
        transducer_insp.xt -= x_transd_shift

        # Raytracer engine to find time of flight reflection:
        raytracer_insp_reflect = ReflectionRayTracer(acoustic_lens, pipeline_insp, transducer_insp, transmission_loss=True, directivity=True)
        # raytracer_insp_reflect.set_speeds(c1, c2_inspection, c3)

        #
        raytracer_insp_refract = FocusRayTracer(acoustic_lens, pipeline_insp, transducer_insp, transmission_loss=True, directivity=True)
        # raytracer_insp_refract.set_speeds(c1, c2_inspection, c3)



        #%%
        #
        raytracer_law = FocusRayTracer(acoustic_lens, pipeline_law, transducer)
        raytracer_law.set_speeds(c1, c2_law, c3)

        # Delay law related parameters:
        delta_alpha = np.deg2rad(.5)
        alpha_max = pi/4
        alpha_min = -pi/4
        alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

        focus_radius = inner_radius
        focus_angle = np.copy(alpha_span)
        xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)

        tofs_roi, _ = raytracer_law.solve(xf, zf + zshift_law)
        delay_law_focused = tofs_roi[transducer.num_elem // 2, :] - tofs_roi

        #%%

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
        idxs = np.searchsorted(focus_angle, np.radians(chosen_angle))
        # x_reflector, z_reflector = xf[idxs], zf[idxs]

        reflector_radius = inner_radius + 2.25e-3
        x_reflector, z_reflector = reflector_radius * sin(focus_angle[idxs]), reflector_radius * cos(focus_angle[idxs])

        x_reflector, z_reflector = np.array([x_reflector]), np.array([z_reflector])

        # With outer surface
        sim1 = Simulator(configs, [raytracer_insp_refract], verbose=False)
        sim1.add_reflector(x_reflector, z_reflector + zshift_insp, different_instances=True)
        sscan = sim1.get_response()

        sscan_env = envelope(sscan, axis=0)
        sscan_log = np.log10(sscan_env + 1e-6)

        #%%
        time_shift = 2 * (pipeline_insp.zcenter / c2_law) * -1e6

        filled_marker_style = dict(marker='o', linestyle=':', markersize=20,
                                   color='none',
                                   markerfacecolor='none',
                                   markerfacecoloralt='none',
                                   markeredgecolor='red',
                                   label="Expected reflector position")


        i = 0

        plt.figure()
        fig, ax = plt.subplots(figsize=(12, 10), nrows=2, ncols=1)
        plt.suptitle(f"zcenter_law = {zshift_law * 1e3:.2f} mm; zcenter_insp = {zshift_insp * 1e3:.2f}. c2_law = {c2_law:.2f}, c2_insp = {c2_inspection:.2f}, c2_proj = {c2_project:.2f}")
        im = ax[0].imshow(sscan_env,
                        extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim1.tspan[-1] * 1e6, sim1.tspan[0] * 1e6],
                        aspect='auto', interpolation='none', cmap='jet')

        # TOFs of interest:
        outer_surf_time = ((d - h0) / c1 + (h0 - pipeline_insp.outer_radius) / c2_inspection) * 2 * 1e6 + time_shift
        inner_surf_time = outer_surf_time + 2 * pipeline_insp.wall_width / c3 * 1e6
        scatterer_time = outer_surf_time + 2 * 1e6 * (pipeline_insp.outer_radius - reflector_radius) / c3

        # Nominal positions:
        ax[0].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * outer_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[0].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[0].plot(np.rad2deg(focus_angle[idxs]), scatterer_time, **filled_marker_style)

        ax[0].set_ylim([65 + time_shift, 50 + time_shift])

        plt.grid(alpha=.25)
        ax[0].set_xlabel(r"$\alpha$-axis / (degrees)")
        ax[0].set_ylabel(r"time-axis / ($\mathrm{\mu s}$)")
        ax[0].set_xticks(np.arange(-45, 45 + 15, 15))

        # Nominal positions:
        ax[0].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * outer_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[0].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[0].plot(np.rad2deg(focus_angle[idxs]), scatterer_time, **filled_marker_style)

        if i + 1 == 1:
            # plt.suptitle("Spatial Impulse Response simulation")
            # plt.colorbar()
            ax[0].legend()
            cax = ax[0].inset_axes([0.15, .125, 0.7, .025])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=.7)
            cb.ax.tick_params(axis='x', colors='white')  # set colorbar tick color
            cb.outline.set_edgecolor('white')  # set colorbar edgecolor
            cb.ax.xaxis.label.set_color('white')  # set colorbar label color
            for t in cb.ax.get_xticklabels():  # set colorbar tick labels
                t.set_color('white')

        max_per_column = np.argmax(sscan_env, axis=0)
        t_max_per_column = sim1.tspan[max_per_column]
        # ax[0].plot(np.degrees(alpha_span), t_max_per_column * 1e6, color='magenta')
        # plt.tight_layout()
        # plt.savefig("../figures/sscan_surface.png", dpi=300)
        # plt.show()

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

        # Without outer surface:
        sim2 = Simulator(configs, [raytracer_insp_reflect, raytracer_insp_refract], verbose=False)
        sim2.add_reflector(x_reflector, z_reflector + zshift_insp, different_instances=True)
        sscan = sim2.get_response()
        sscan_env = envelope(sscan, axis=0)

        t_flaw_idx, theta_flaw_idx, _ = np.where(sscan_env == sscan_env.max())
        theta_flaw, t_flaw = alpha_span[theta_flaw_idx], sim2.tspan[t_flaw_idx]


        im = ax[1].imshow(sscan_env,
                        extent=[np.rad2deg(alpha_min), np.rad2deg(alpha_max), sim1.tspan[-1] * 1e6, sim1.tspan[0] * 1e6],
                        aspect='auto', interpolation='None', cmap='jet')
        ax[1].plot(np.degrees(theta_flaw), t_flaw * 1e6, 'xr')
        ax[1].set_ylim([65 + time_shift, 50 + time_shift])
        plt.grid(alpha=.25)
        ax[1].set_xlabel(r"$\alpha$-axis / (degrees)")
        ax[1].set_ylabel(r"time-axis / ($\mathrm{\mu s}$)")
        ax[1].set_xticks(np.arange(-45, 45 + 15, 15))

        # Nominal positions:
        ax[1].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * outer_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[1].plot(np.degrees(alpha_span), np.ones_like(alpha_span) * inner_surf_time, '--', color="#A9A9A9", linewidth=1.5)
        ax[1].plot(np.rad2deg(focus_angle[idxs]), scatterer_time, **filled_marker_style)

        plt.tight_layout()
        # plt.savefig("../figures/sscan_flaw.png", dpi=300)

        #%% Residual thickness:

        t_outer_surface = t_max_per_column[bisect(alpha_span,  theta_flaw)]
        # t_outer_surface = 60.88e-6
        delta_tof = t_flaw - t_outer_surface
        residual_thickness = .5 * delta_tof * c3 * 1e3
        print(f"Residual thickness = {residual_thickness[0]:.4f} mm")
        residual_thickness_matrix[_aa, _zz] = residual_thickness
        theta_flaw_matrix[_aa, _zz] = theta_flaw


#%%
#
# if len(zcenter_grid) != 0 or len(chosen_angle_grid) != 0:
#     plt.figure(figsize=(7,3.4))
#
#     for i, ang in enumerate(chosen_angle_grid):
#         plt.plot(zcenter_grid * 1e3, residual_thickness_matrix[i, :], 'o--', label=f"SDH at ${ang:.1f}^\circ$.")
#
#     zgrid = np.arange(zcenter_grid[0] - 15e-3, zcenter_grid[-1] + 15e-3, 1e-3)
#     plt.plot(zgrid * 1e3, 8.99 * np.ones_like(zgrid), 'k--', label="Ideal")
#     plt.xlabel("Tube center shift along z-axis / (mm)")
#     plt.ylabel("Residual thickness / (mm)")
#     plt.xlim([zcenter_grid[0] * 1e3 - 2.5, zcenter_grid[-1] * 1e3 + 2.5] )
#     plt.ylim([7, 10])
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig("../figures/zshift_influence.png", dpi=300)
#     plt.show()


fig, ax = plt.subplots(figsize=(7, 3.5))
plt.plot(chosen_angle_grid, (reflector_radius * 1e3) * np.ones_like(residual_thickness_matrix), 'xk', label='Ideal')
plt.plot(np.degrees(theta_flaw_matrix), residual_thickness_matrix, 'or', markersize=3, label="After non-idealities")
plt.plot(-1 * np.asarray(chosen_angle_grid), (reflector_radius * 1e3)* np.ones_like(residual_thickness_matrix), 'xk')
plt.plot(-1 * np.degrees(theta_flaw_matrix), residual_thickness_matrix, 'or', markersize=3)
plt.ylim([15, 19])
plt.xticks(np.arange(-45, 45 + 15, 15))
plt.ylabel("Residual thickness / (mm)")
plt.xlabel("$\\alpha$-axis / (degrees)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()