import numpy as np
import matplotlib
from framework.post_proc import envelope
from framework import file_civa
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
from pipe_lens_imaging.pwi_tfm import cpwc_circle_kernel
from pipe_lens_imaging.pwi_tfm_utils import f_circ
import time


if __name__ == '__main__':
    #%% Data-input:

    ang_lim_list = [70]
    ang_step_list = [.5]


    data_insp = file_civa.read("../data/test.civa")

    progress = 0
    total = len(ang_lim_list) * len(ang_step_list)
    for ang_lim in ang_lim_list:
        for ang_step in ang_step_list:
            print(f"Progress = {progress}/{total}")
            progress += 1

            original_step = .5
            step = int(ang_step/original_step)


            pwi_data = data_insp.ascan_data[..., 0]
            _, Nang, _ = pwi_data.shape
            pwi_data = pwi_data[:, Nang//2 - int(ang_lim/original_step): 1 + Nang//2 + int(ang_lim/original_step), :]
            pwi_data = pwi_data[:, ::step, :]

            #%% User-input:

            # Transducer:
            xt = data_insp.probe_params.elem_center[:, 0] * 1e-3
            zt = data_insp.probe_params.elem_center[:, 2] * 1e-3

            fs = data_insp.inspection_params.sample_freq * 1e6

            gate_start = data_insp.inspection_params.gate_start * 1e-6

            # Interfaces:
            c_coupling = data_insp.inspection_params.coupling_cl
            c_specimen = data_insp.specimen_params.cl

            # Specimen:
            radius = 200e-3 / 2
            waterpath = 30e-3
            wall_thickness = 45e-3
            xcenter, zcenter = 0, waterpath + radius

            # ROI:
            roi_coord_system = "cartesian" # or "cartesian" or "polar"

            if roi_coord_system == "polar":
                delta_r = .05e-3
                delta_ang = .1
                r_roi = np.arange(radius - wall_thickness - 10e-3, radius + 10e-3, delta_r)
                ang_roi = np.radians(np.arange(-20, 90, delta_ang))

                aa, rr = np.meshgrid(-ang_roi, r_roi)
                xx, zz = xcenter - rr * np.sin(aa), zcenter - rr * np.cos(aa)

            elif roi_coord_system == "cartesian":
                zroi = np.linspace(20, 90, 700 * 2) * 1e-3
                xroi = np.linspace(-70, 70, 700 * 2) * 1e-3
                xx, zz = np.meshgrid(xroi, zroi)
            else:
                raise NotImplementedError

            print(f"Number of pixels = {xx.shape[0] * xx.shape[1]:.2E}")
            steering_angs = np.radians(data_insp.inspection_params.angles)
            steering_angs = steering_angs[Nang//2 - int(ang_lim/original_step): 1 + Nang//2 + int(ang_lim/original_step)]
            steering_angs = steering_angs[::step]


            #%%
            ti = time.time()
            img, delaylaw = cpwc_circle_kernel(pwi_data, xx, zz, xt, zt, xcenter, zcenter, radius, wall_thickness, steering_angs, c_coupling, c_specimen, fs, gate_start, spatialWeightingMask=True)
            tf = time.time()
            print(f"Elapsed-time = {tf - ti:.2f}")

            # img_norm = normalize(img)
            img_env = envelope(img, axis=0)
            img_env = np.abs(img)
            img_env /= img_env.max()
            img_db = 20 * np.log10(img_env + 1e-6)


            #%%


            xmin, xmax = np.min(xx), np.max(xx)
            zmin, zmax = np.min(zz), np.max(zz)

            if roi_coord_system == "cartesian":
                plt.figure(figsize=(14 * .7, 6 * .7))
                plt.imshow(img_db, cmap='inferno', vmax=0, vmin=-90, extent=[xmin * 1e3, xmax * 1e3, zmax * 1e3, zmin * 1e3])
                plt.colorbar()
                #
                ang_string = ""
                for t in [0, 15, 30, 45, 60]:
                    r = (radius - wall_thickness) + (wall_thickness - 25e-3)
                    theta = np.radians(90 - t)
                    xcirc, zcirc = xcenter + r * np.cos(theta), zcenter - r * np.sin(theta)
                    sdh = plt.Circle(xy=(xcirc * 1e3, zcirc * 1e3), radius=2, fill=False, edgecolor='w')
                    plt.gca().add_patch(sdh)
                    ang_string += fr"${t:.0f}^\circ$, "

                # Plot surfaces:

                plt.title(fr"CPWC through circular surface with firing at $\theta \in [{-ang_lim:.1f}^\circ, \dots, {ang_lim:.1f}^\circ]$ where $\Delta\theta = {ang_step:.1f}^\circ$" + "\n" + "SDH at: " + ang_string)
                xspan = np.linspace(xmin, xmax, 100)
                plt.plot(xspan * 1e3, f_circ(xspan, xcenter, zcenter, radius) * 1e3, 'w--')
                plt.plot(xspan * 1e3, f_circ(xspan, xcenter, zcenter, radius - wall_thickness) * 1e3, 'w--')

                plt.gca().fill_between([xt[0] * 1e3, xt[-1] * 1e3], [zmin * 1e3, zmin * 1e3], [zmax * 1e3, zmax * 1e3], color='k', alpha=.1, label='PA Active Aperture')

                plt.ylim([zmax * 1e3, zmin * 1e3])
                plt.xlim([xmin * 1e3, xmax * 1e3])
                plt.legend()
                plt.grid(alpha=.3)
                plt.xlabel(r"x-axis / $\mathrm{\mu s}$")
                plt.ylabel(r"z-axis / $\mathrm{\mu s}$")
                # plt.plot(xx * 1e3, zz * 1e3, 'xr', alpha=.1)
                plt.tight_layout()
                # plt.savefig(f"../figures/cpwc_analysis/cpwc_circ_{-ang_lim:.1f}_{ang_lim:.1f}_step{ang_step:.1f}.pdf")
            elif roi_coord_system == "polar":
                plt.figure(figsize=(9, 6))
                plt.imshow(img_db, 'inferno', vmax=0, vmin=-90, extent=[np.min(np.degrees(ang_roi)), np.max(np.degrees(ang_roi)), np.max(rr * 1e3), np.min(rr * 1e3)], aspect='auto', interpolation="None")
                plt.colorbar()

                ang_string = ""
                for t in [0, 15, 30, 45, 60]:
                    r = (radius - wall_thickness) + (wall_thickness - 25e-3)
                    theta = np.radians(t)
                    xcirc, zcirc = xcenter + r * np.cos(theta), zcenter - r * np.sin(theta)
                    sdh = plt.Circle(xy=(np.degrees(theta), r * 1e3), radius=2, fill=False, edgecolor='w')
                    plt.gca().add_patch(sdh)
                    plt.ylabel("r-axis / (mm)")
                    plt.xlabel(r"$\alpha$-axis / (degrees)")
                    ang_string += fr"${t:.0f}^\circ$, "

                xouter = np.degrees(np.arange(ang_roi[0], ang_roi[-1] + .1, .1))
                zouter = radius * np.ones_like(xouter) * 1e3
                zinner = zouter - wall_thickness * 1e3
                plt.plot(xouter, zouter, ':w')
                plt.plot(xouter, zinner, ':w')
                plt.xlim([np.degrees(ang_roi[0]), np.degrees(ang_roi[-1])])
                plt.ylim([r_roi[0] * 1e3, r_roi[-1] * 1e3])
                plt.title(fr"CPWC through circular surface with firing at $\theta \in [{-ang_lim:.1f}^\circ, \dots, {ang_lim:.1f}^\circ]$ where $\Delta\theta = {ang_step:.1f}^\circ$" + "\n" + "SDH at: " + ang_string)
                plt.tight_layout()

            else:
                raise NotImplementedError
            plt.show()
