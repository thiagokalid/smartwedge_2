import numpy as np
import matplotlib
from framework.post_proc import envelope, normalize
from framework import file_m2k
import matplotlib.pyplot as plt
from pipe_lens_imaging.utils import api


matplotlib.use('TkAgg')
from pipe_lens_imaging.pwi_tfm import *
from matplotlib.ticker import FuncFormatter

from archive.cpwc.pw_circ import f_circ
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
    api_list, max_list = list(), list()

    ii = 0
    for jj in tqdm(range(15 - 10, 15 + 10)):
        #%% Data-input:
        data_insp = file_m2k.read("../data/res_dir_passiva_PWI.m2k", freq_transd=5, bw_transd=.5, tp_transd='gaussian', sel_shots=jj, read_ascan=True)
        pwi_data = data_insp.ascan_data[..., 0]
        Nt, Nangs, Nel = pwi_data.shape

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
        roi_coord_system = "cartesian"  # or "cartesian" or "polar"

        if roi_coord_system == "polar":
            delta_r = .05e-3
            delta_ang = .1
            r_roi = np.arange(radius - wall_thickness - 10e-3, radius + 10e-3, delta_r)
            ang_roi = np.radians(np.arange(-20, 90, delta_ang))

            aa, rr = np.meshgrid(-ang_roi, r_roi)
            xx, zz = xcenter - rr * np.sin(aa), zcenter - rr * np.cos(aa)

        elif roi_coord_system == "cartesian":
            xroi = np.linspace(-35, 35, 700) * 1e-3
            zroi = np.linspace(20, 70, 500) * 1e-3
            xx, zz = np.meshgrid(xroi, zroi)
        else:
            raise NotImplementedError

        print(f"Number of pixels = {xx.shape[0] * xx.shape[1]:.2e}")

        #%%
        ti = time.time()

        if ii == 0:
            t0 = time.time()
            Nrows, Ncols = zz.shape[0], zz.shape[1]
            Nangs = pwi_data.shape[1]
            baseline_shift = np.int64(gate_start * fs)
            thetas = steering_angs

            # Compute the delay-law:
            t0 = time.time()
            pwi_tof, xe, ze = compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
            t_ref = np.max(pwi_tof)

            mask = np.ones(shape=(Nangs, Nrows, Ncols))
            tmp_mask = is_inside_pipe(xx, zz, xcenter, zcenter, radius, wall_thickness)
            mask *= tmp_mask[np.newaxis, ...]
            mask *= is_inside_pwr(thetas, xx, zz, xt, zt, xe, ze, xcenter, zcenter, radius, c_coupling, c_specimen)
            binary_mask = np.sum(mask, axis=0) > 0
            t_tfm = compute_t_tfm(xx, zz, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen, binary_mask)
            t_pwi = compute_t_pwi(thetas, xx, zz, t_ref, xcenter, zcenter, c_specimen, binary_mask)
            print(f"First-time computations. Elapsed-time = {time.time() - t0:.2f} s")
            ii += 1


        img = cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift, mask)

        corners = [
            (-10e-3, 40e-3),
            (10e-3, 46.6e-3)
        ]
        tf = time.time()
        print(f"Elapsed-time = {tf - ti:.2f}")

        api_val, local_max, api_mask = api(img, xroi, zroi, corners)
        api_list.append(api_val)
        max_list.append(local_max)

        # img_env = np.abs(img)
        img_env = envelope(img, axis=0)
        img_db = 20 * np.log10(img_env / img_env.max() + 1e-6)

    max_vector = np.asarray(max_list)
    max_normalized = (max_vector - max_vector.min()) / (max_vector.max() - max_vector.min())
    ymax_idx = np.argmax(max_normalized)

    ystep = 1e-3
    yspan = np.arange(0, len(max_vector) * ystep, ystep)
    yspan -= yspan[ymax_idx]
    yspan *= 1e3

    ymax = yspan[np.argmax(max_normalized)]

    peaks_interp = lambda y: np.interp(y, yspan, max_normalized)
    cost_fun = lambda y: np.power(peaks_interp(y) - .5, 2)

    half_peak_loc_left = scipy.optimize.minimize(cost_fun, ymax, bounds=[(yspan[0], ymax)]).x[0]
    half_peak_loc_right = scipy.optimize.minimize(cost_fun, ymax, bounds=[(ymax, yspan[-1])]).x[0]
    fwhm = float(half_peak_loc_right - half_peak_loc_left)

    peaks_percentage = max_normalized * 100


    fig, ax = plt.subplots(figsize=(linewidth * .49, linewidth * .4))
    plt.plot(yspan, peaks_percentage, ":o", color="k")
    plt.xlabel("Position along passive direction / (mm)")
    plt.ylabel(r"Normalized amplitude / (\%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
    plt.ylim([-10, 110])
    plt.yticks(np.arange(0, 125, 25))
    plt.xlim([-8, 8])
    ytemp = np.arange(20, 60, 1)
    plt.xticks(np.linspace(-8, 8, 5))
    plt.grid(alpha=.5)
    plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
    plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

    ax.annotate("", xy=(half_peak_loc_left, 25), xytext=(half_peak_loc_right, 25),
                arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
                ha="center",  # Center text horizontally
                va="bottom"  # Position text below arrow
                )
    ax.annotate(rf"${fwhm:.2f}$ mm", xy=(0.22, 25), xytext=(0.22, 30),
                ha="center",  # Center text horizontally
                va="bottom"  # Position text below arrow
                )

    plt.tight_layout()
    plt.savefig("../figures/passive_dir_resolution_pa.pdf")
    plt.show()