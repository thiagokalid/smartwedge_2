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

    ang_lim_list = [40, 50, 60, 70]
    ang_step_list = [.5, 1., 2., 5., 10.]


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
            delta_r = .1e-3
            delta_ang = .5
            r_roi = np.arange(radius - wall_thickness, radius, delta_r)
            ang_roi = np.radians(np.arange(-30, 90, delta_ang))

            # rr, aa = np.meshgrid(r_roi, -ang_roi)
            # rr, aa = rr.flatten(order='F'), aa.flatten(order='F')
            # xxroi, zzroi = xcenter - rr * np.sin(aa), zcenter - rr * np.cos(aa)

            # plt.figure()
            # plt.plot(xxroi, zzroi, 'x', color='r')
            # plt.show()


            zroi = np.linspace(30, 80, 500) * 1e-3
            xroi = np.linspace(-60, 60, 600) * 1e-3

            steering_angs = np.radians(data_insp.inspection_params.angles)
            # steering_angs = np.radians(np.arange(-70, 70 + .5, .5))
            steering_angs = steering_angs[Nang//2 - int(ang_lim/original_step): 1 + Nang//2 + int(ang_lim/original_step)]
            steering_angs = steering_angs[::step]


            #%%
            ti = time.time()
            img, delaylaw = cpwc_circle_kernel(pwi_data, xroi, zroi, xt, zt, xcenter, zcenter, radius, wall_thickness, steering_angs, c_coupling, c_specimen, fs, gate_start)
            tf = time.time()
            print(f"Elapsed-time = {tf - ti:.2f}")

            # img_norm = normalize(img)
            img_env = envelope(img, axis=0)
            # img_env = np.abs(img)
            img_env /= img_env.max()
            img_db = 20 * np.log10(img_env + 1e-6)


            #%%
            
            xmin, xmax = np.min(xroi), np.max(xroi)
            zmin, zmax = np.min(zroi), np.max(zroi)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img_db, cmap='jet', vmax=0, vmin=-90, extent=[xmin * 1e3, xmax * 1e3, zmax * 1e3, zmin * 1e3])
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
            # plt.plot(xxroi * 1e3, zzroi * 1e3, 'xr', alpha=.1)
            plt.tight_layout()
            plt.savefig(f"../figures/cpwc_analysis/cpwc_circ_{-ang_lim:.1f}_{ang_lim:.1f}_step{ang_step:.1f}.pdf")
            # plt.show()

            # plt.figure()
            # plt.imshow(img_db, 'jet', vmax=0, vmin=-90, extent=[np.min(np.degrees(ang_roi)), np.max(np.degrees(ang_roi)), np.min(rr * 1e3), np.max(rr * 1e3)], aspect='auto')
            # for t in [0, 15, 30, 45, 60]:
            #     r = (radius - wall_thickness) + (wall_thickness - 25e-3)
            #     theta = np.radians(t)
            #     xcirc, zcirc = xcenter + r * np.cos(theta), zcenter - r * np.sin(theta)
            #     sdh = plt.Circle(xy=(np.degrees(theta), r * 1e3), radius=2, fill=False, edgecolor='w')
            #     plt.gca().add_patch(sdh)
            #     ang_string += fr"${t:.0f}^\circ$, "
            #
            # plt.show()
