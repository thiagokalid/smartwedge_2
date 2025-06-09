import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

from framework import file_m2k
from pipe_lens_imaging.utils import api, fwhm
from pipe_lens_imaging.pwi_tfm import (
    is_inside_pipe,
    is_inside_pwr,
    compute_delaylaw_pwi,
    compute_t_tfm,
    compute_t_pwi,
    cpwc_coherent_sum
)

# --- Configuration and Setup ---
matplotlib.use('TkAgg')

# LaTeX linewidth for consistent plotting with LaTeX documents
linewidth = 6.3091141732

# Matplotlib settings for LaTeX text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

# --- Main Execution Block ---
if __name__ == '__main__':
    selected_indexes = range(15 - 10, 15 + 10)
    api_values = np.zeros(len(selected_indexes))
    max_values = np.zeros(len(selected_indexes))
    first_iteration = True

    for i, selected_index in enumerate(tqdm(selected_indexes)):
        # --- Data Input ---
        data_insp = file_m2k.read(
            "../data/res_dir_passiva_PWI.m2k",
            freq_transd=5,
            bw_transd=.5,
            tp_transd='gaussian',
            sel_shots=selected_index,
            read_ascan=True
        )
        pwi_data = data_insp.ascan_data[..., 0]
        Nt, Nangs, Nel = pwi_data.shape

        # --- User Input & Initial Computations (Run once) ---
        if first_iteration:
            # Transducer parameters
            xt = data_insp.probe_params.elem_center[:, 0] * 1e-3
            zt = data_insp.probe_params.elem_center[:, 2] * 1e-3
            theta = np.radians(0)
            xt = np.cos(theta) * xt + np.sin(theta) * zt
            zt = -np.sin(theta) * xt + np.cos(theta) * zt

            fs = data_insp.inspection_params.sample_freq * 1e6
            gate_start = data_insp.inspection_params.gate_start * 1e-6

            # Interface parameters
            c_coupling = data_insp.inspection_params.coupling_cl
            c_specimen = data_insp.specimen_params.cl
            steering_angs = np.radians(data_insp.inspection_params.angles)

            radius = 140e-3 / 2
            waterpath = 32e-3
            wall_thickness = 17.23e-3 + 5e-3
            xcenter, zcenter = 0, waterpath + radius

            # ROI (Region of Interest)
            xroi = np.linspace(-35, 35, 700) * 1e-3
            zroi = np.linspace(20, 70, 500) * 1e-3
            xx, zz = np.meshgrid(xroi, zroi)

            # Define planewave imaging parameters
            Nrows, Ncols = zz.shape[0], zz.shape[1]
            Nangs = pwi_data.shape[1]
            baseline_shift = np.int64(gate_start * fs)
            thetas = steering_angs

            # Compute the delay-law
            pwi_tof, xe, ze = compute_delaylaw_pwi(
                thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen
            )
            t_ref = np.max(pwi_tof)
            delay_law = t_ref - pwi_tof

            # Create masks
            mask = np.ones(shape=(Nangs, Nrows, Ncols), dtype=np.bool_)
            tmp_mask = is_inside_pipe(xx, zz, xcenter, zcenter, radius, wall_thickness)
            mask *= tmp_mask[np.newaxis, ...]
            mask *= is_inside_pwr(
                thetas, xx, zz, xt, zt, xe, ze, xcenter, zcenter, radius, c_coupling, c_specimen
            )
            binary_mask = np.sum(mask, axis=0, dtype=np.bool_)

            # Compute time-of-flight (TOF) maps
            t_tfm = compute_t_tfm(
                xx, zz, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen, binary_mask
            )
            t_pwi = compute_t_pwi(
                thetas, xx, zz, t_ref, xcenter, zcenter, c_specimen, binary_mask
            )
            indexes = np.arange(Nangs * Ncols * Nrows, dtype=np.int64)[np.ravel(mask, order='C')]
            del mask, binary_mask # Free up memory
            first_iteration = False

        # --- Image Reconstruction ---
        img = cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift, indexes)

        # --- API Calculation ---
        corners = [
            (-10e-3, 40e-3),
            (10e-3, 46.6e-3)
        ]
        api_values[i], max_values[i], _ = api(img, xroi, zroi, corners)

    # --- Post-processing and Plotting ---
    # Compute FWHM
    yspan = np.array(selected_indexes)
    yspan -= yspan[np.argmax(max_values)]
    left_side_fwhm, right_side_fwhm = fwhm(signal=max_values, xspan=yspan)
    curr_fwhm = np.abs(right_side_fwhm - left_side_fwhm)

    # Normalize peaks for plotting
    peaks_percentage = max_values / max_values.max() * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(linewidth * .49, linewidth * .4))
    plt.plot(yspan, peaks_percentage, ":o", color="k")
    plt.xlabel("Position along passive direction / (mm)")
    plt.ylabel(r"Normalized amplitude / (\%)")

    # Y-axis formatting
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
    plt.ylim([-10, 110])
    plt.yticks(np.arange(0, 125, 25))

    # X-axis formatting
    plt.xlim([-8, 8])
    plt.xticks(np.linspace(-8, 8, 5))
    plt.grid(alpha=.5)

    # Plot FWHM lines
    ytemp = np.arange(20, 60, 1)
    plt.plot(left_side_fwhm * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
    plt.plot(right_side_fwhm * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

    # Add FWHM annotations
    ax.annotate("", xy=(left_side_fwhm, 25), xytext=(right_side_fwhm, 25),
                arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
                ha="center", va="bottom")
    ax.annotate(rf"${curr_fwhm:.2f}$ mm", xy=(0.22, 25), xytext=(0.22, 30),
                ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("../figures/passive_dir_resolution_pa.pdf")
    plt.show()