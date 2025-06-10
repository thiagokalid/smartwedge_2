import numpy as np
import matplotlib
from framework.post_proc import envelope, normalize
from framework import file_m2k
import matplotlib.pyplot as plt
from pipe_lens_imaging.utils import api


matplotlib.use('TkAgg')
from pipe_lens_imaging.pwi_tfm import *
from matplotlib.ticker import FuncFormatter
from parameter_estimation.intsurf_estimation import profile_fadmm

from scipy.signal import find_peaks
from bisect import bisect


def first_peaks(img, height_perc=5):
    Ncols = img.shape[1]
    z = np.zeros(shape=(Ncols), dtype=np.int32)
    for c in range(Ncols):
        column = img[:, c]
        relative_max = column.max()

        result = find_peaks(column / relative_max, height=height_perc / 100)[0]
        if len(result) > 0:
            z[c] = result[0]
        else:
            z[c] = np.argmax(column)

    w = np.diag((np.max(img, axis=0)))
    return w, z

import time
import scipy
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

if __name__ == '__main__':
    yspan = np.arange(0, 170)

    chosen_y = np.arange(0, 170)


    generateVideo = True
    estimateSurf = True
    firstIteration = True
    for curr_y, vv in enumerate(tqdm(chosen_y)):
        jj = 169 - vv

        #%% Data-input:
        data_insp = file_m2k.read("../data/Varredura_PWI.m2k", freq_transd=5, bw_transd=.5, tp_transd='gaussian', sel_shots=int(jj), read_ascan=True)
        pwi_data = data_insp.ascan_data[..., 0]

        if firstIteration:
            Nt, Nangs, Nel = pwi_data.shape

            time_grid = data_insp.time_grid[:]
            #%% User-input:

            # Transducer:
            xt = data_insp.probe_params.elem_center[:, 0] * 1e-3
            zt = data_insp.probe_params.elem_center[:, 2] * 1e-3

            theta = np.radians(0)
            xt = np.cos(theta) * xt + np.sin(theta) * zt
            zt = -np.sin(theta) * xt + np.cos(theta) * zt

            fs = data_insp.inspection_params.sample_freq * 1e6

            gate_start = data_insp.inspection_params.gate_start * 1e-6

            # Interfaces:
            c_coupling = data_insp.inspection_params.coupling_cl
            c_specimen = data_insp.specimen_params.cl


            steering_angs = np.radians(data_insp.inspection_params.angles)

            radius = 140e-3 / 2
            waterpath = 32e-3
            wall_thickness = 17.23e-3 + 5e-3

            xcenter, zcenter = 0, waterpath + radius

            # ROI:
            delta_r = .05e-3
            delta_ang = .2
            r_roi = np.arange(radius - wall_thickness - 10e-3, radius + 10e-3, delta_r)
            ang_roi = np.radians(np.arange(-45, 45, delta_ang))
            aa, rr = np.meshgrid(-ang_roi, r_roi)
            xx, zz = xcenter - rr * np.sin(aa), zcenter - rr * np.cos(aa)

            Nrows, Ncols = zz.shape[0], zz.shape[1]
            baseline_shift = np.int64(gate_start * fs)
            thetas = steering_angs
            ang_span = np.degrees(ang_roi)
            r_span = r_roi * 1e3
            outer_radius = radius * 1e3
            inner_radius = (radius - 20e-3) * 1e3

            # Compute the delay-law:
            pwi_tof, xe, ze = compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
            t_ref = np.max(pwi_tof)
            delay_law = t_ref - pwi_tof

            mask = np.ones(shape=(Nangs, Nrows, Ncols), dtype=np.bool)

            tmp_mask = is_inside_pipe(xx, zz, xcenter, zcenter, radius, wall_thickness)
            mask *= tmp_mask[np.newaxis, ...]

            mask *= is_inside_pwr(thetas, xx, zz, xt, zt, xe, ze, xcenter, zcenter, radius, c_coupling,
                                  c_specimen)

            binary_mask = np.sum(mask, axis=0, dtype=np.bool) == True

            t0 = time.time()
            t_tfm = compute_t_tfm(xx, zz, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen, binary_mask)
            t_pwi = compute_t_pwi(
                thetas, xx, zz,
                t_ref, xcenter, zcenter,
                c_specimen,
                binary_mask)

            indexes = np.arange(Nangs * Ncols * Nrows, dtype=np.int64)[np.ravel(mask, order='C')]

            del binary_mask, mask

            outer_surface = np.memmap(filename="outer_surface.dat", shape=(len(yspan), Ncols), dtype=np.int32, mode="w+")
            inner_surface = np.memmap(filename="inner_surface.dat", shape=(len(yspan), Ncols), dtype=np.int32, mode="w+")

            firstIteration = False



        img = cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift, indexes)

        corners = [
            (-10e-3, 40e-3),
            (10e-3, 46.6e-3)
        ]

        img_env = envelope(img, axis=0)
        img_log = np.log10(img_env / img_env.max() + 1e-6)

        if estimateSurf:
            idx_beg, idx_end = bisect(r_span, 48.53) - 1, bisect(r_span, 71) - 1
            Nhalf = (idx_beg + idx_end) / 2

            sscan_env = img_env[::-1, :]


            Nhalf = sscan_env.shape[0] // 2

            w, z = first_peaks(sscan_env[Nhalf:idx_end, :], height_perc=10)
            inner_surf, resf, kf, pk, sk = profile_fadmm(w, z, lamb=5e13, x0=z, rho=1e11, eta=.999, itmax=25, tol=1e-3)
            inner_surface[vv, :] = np.int32(inner_surf) + Nhalf

            w, z = first_peaks(sscan_env[idx_beg:Nhalf, :], height_perc=10)
            outer_surf, resf, kf, pk, sk = profile_fadmm(w, z, lamb=5e13, x0=z, rho=1e11, eta=.999, itmax=25, tol=1e-3)
            outer_surface[vv, :] = np.int32(outer_surf) + idx_beg


        if generateVideo:
            plt.figure(figsize=(8, 6))
            plt.title(f"Normalized S-scan (log-scale) of a passive \n diretion sweep *without* acoustic lens. $y={yspan[vv]}$ mm.")
            plt.pcolormesh(ang_span, r_span, img_log, cmap='inferno', vmin=-5, vmax=0)
            plt.plot(ang_span, r_span[::-1][outer_surface[vv, :]], 'w-', linewidth=2)
            plt.plot(ang_span, r_span[::-1][inner_surface[vv, :]], 'w-', linewidth=2)

            plt.ylabel("Radial direction / (mm)")
            plt.xlabel(rf"$\alpha$-axis / (degrees)")
            plt.colorbar()
            plt.grid(alpha=.5)
            plt.ylim([48, 72])
            plt.xticks(np.arange(-45, 45 + 15, 15))
            plt.yticks(np.arange(50, 70 + 5, 5))
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"../figures/cpwc_frames_surf/file{vv:02d}.png")


