import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import gc
from bisect import bisect
import time
import scipy
from numpy import power

# Add parent directory to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipe_lens.imaging_utils import fwhm, convert_time2radius
from pipe_lens_imaging.utils import fwhm as fwhm_sscan  # Aliased to avoid conflict
from framework import file_m2k
from framework.post_proc import envelope

# --- Configuration and Setup ---
linewidth = 6.3091141732  # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

# Data root directory
ROOT_DIR = '../data/resolution/passive_dir/'

# Acoustic lens geometry parameters
ANGULAR_LOCATIONS = [
    0,  # degree
    10,  # degree
    20,  # degree
    30,  # degree
    38.5  # degree
]
NUM_VERSIONS = 3


# --- Data Processing ---

def process_single_element_data(root_dir, num_versions):
    """Processes single-element data to compute FWHM."""
    fwhm_mono = np.zeros(num_versions)
    print("Processing Single-element data:")
    for vv in tqdm(range(num_versions)):
        version = f"v{vv + 1}"
        data_path = os.path.join(root_dir, f"passive_dir_single_element_{version}.m2k")
        data = file_m2k.read(data_path, type_insp='contact', water_path=0,
                             freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

        time_grid = data.time_grid
        channels = data.ascan_data
        t_outer, t_inner = 19.8, 23

        num_steps = channels.shape[-1]
        peaks_mono = np.zeros(num_steps)
        for step in range(num_steps):
            idx_beg, idx_end = bisect(time_grid, t_outer), bisect(time_grid, t_inner)
            peaks_mono[step] = np.max(channels[idx_beg:idx_end, ..., step])

        # Finding FWHM
        xspan = np.arange(0, num_steps) - np.where(peaks_mono == peaks_mono.max())[0]
        minimum = np.min([peaks_mono[bisect(xspan, -4) - 1], peaks_mono[bisect(xspan, 4) - 1]])
        normalized_peaks = (peaks_mono - minimum) / (peaks_mono.max() - minimum)

        peaks_interp = lambda x: np.interp(x, xspan, normalized_peaks)
        cost_fun = lambda x: power(peaks_interp(x) - .5, 2)

        half_peak_loc_left = scipy.optimize.minimize(cost_fun, 0, bounds=[(xspan[0], 0)]).x
        half_peak_loc_right = scipy.optimize.minimize(cost_fun, 0, bounds=[(0, xspan[-1])]).x
        passive_flaw_width = half_peak_loc_right[0] - half_peak_loc_left[0]
        fwhm_mono[vv] = passive_flaw_width

        del data, channels
        gc.collect()
        time.sleep(1)  # Small delay for garbage collection to be effective
        gc.collect()

    print(f"FWHM single-element mean: {np.mean(fwhm_mono):.2f} and std: {np.std(fwhm_mono):.2f}")
    time.sleep(1)
    return fwhm_mono


def process_acoustic_lens_data(root_dir, angular_locations, num_versions):
    """Processes acoustic lens data to compute passive flaw widths."""
    passive_flaw_widths = np.zeros(shape=(len(angular_locations), num_versions), dtype=float)
    plot_values_for_0deg_v1 = None  # To store data for the specific plot

    print("Processing Phased Array data:")
    total_iterations = len(angular_locations) * num_versions
    with tqdm(total=total_iterations) as pbar:
        for index in range(total_iterations):
            ii = index // num_versions
            vv = index % num_versions
            version = f"v{vv + 1}"
            current_ang_location = angular_locations[ii]

            data_path = os.path.join(root_dir, f"passive_dir_{current_ang_location}degree_{version}.m2k")
            data = file_m2k.read(data_path, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

            if current_ang_location in [0, 10, 20]:
                data_ref_path = os.path.join(root_dir, "ref.m2k")
            else:
                data_ref_path = os.path.join(root_dir, "ref2.m2k")
            data_ref = file_m2k.read(data_ref_path, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

            channels_ref = np.mean(data_ref.ascan_data, axis=3)

            n_shots = data.ascan_data.shape[-1]
            log_cte = 1e-6
            time_grid = data.time_grid[:, 0]
            ang_span = np.linspace(-45, 45, 181)

            # Manually identified outer and inner surface times (in microseconds)
            if current_ang_location in [0, 10, 20]:
                t_outer, t_inner = 56.13, 63.22
            else:
                t_outer, t_inner = 53.8, 60.49

            r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

            peaks = np.zeros(n_shots)

            for i in range(n_shots):
                channels = data.ascan_data[..., i]
                sscan = np.sum(channels, axis=2)
                sscan_db = np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

                # Apply FWHM (from pipe_lens_imaging.utils.fwhm) for API area
                corners = [(64, -8 + current_ang_location), (55, +8 + current_ang_location)]
                # The fwhm function signature was (img, r_span, ang_span, corners) based on the original code
                # and returns (widths, heights, peaks, pixels_above_threshold, _)
                _, _, peaks[i], _, _ = fwhm_sscan(sscan, r_span, ang_span, corners)

            # Finding FWHM for the peak profile
            xspan = np.arange(0, n_shots) - np.where(peaks == peaks.max())[0]

            minimum = np.min([peaks[bisect(xspan, -5) - 1], peaks[bisect(xspan, 5) - 1]])
            normalized_peaks = (peaks - minimum) / (peaks.max() - minimum)
            peaks_percentage = normalized_peaks * 100

            # Using fwhm from pipe_lens.imaging_utils.fwhm
            half_peak_loc_left, half_peak_loc_right = fwhm(signal=normalized_peaks, xspan=xspan)
            passive_flaw_width = np.abs(half_peak_loc_right - half_peak_loc_left)
            passive_flaw_widths[ii, vv] = passive_flaw_width

            print(f"({float(half_peak_loc_left):.1f}, {float(half_peak_loc_right):.1f})")
            print(f"FWHM of {current_ang_location} degree ({version}) = {passive_flaw_width:.2f}")

            if (current_ang_location, version) == (0, "v1"):
                plot_values_for_0deg_v1 = [peaks_percentage, half_peak_loc_left, half_peak_loc_right,
                                           passive_flaw_width, xspan]

            del data, data_ref, sscan, sscan_db, channels, channels_ref
            gc.collect()
            time.sleep(0.1)  # Small delay to allow tqdm to update smoothly
            pbar.update(1)  # Update tqdm progress bar
            gc.collect()

    return passive_flaw_widths, plot_values_for_0deg_v1


# --- Main Execution ---
if __name__ == '__main__':
    fwhm_mono_results = process_single_element_data(ROOT_DIR, NUM_VERSIONS)
    passive_flaw_widths_results, plot_data_0deg_v1 = process_acoustic_lens_data(ROOT_DIR, ANGULAR_LOCATIONS,
                                                                                NUM_VERSIONS)

    # --- Plotting ---

    # Plot for 0 degree, v1 acoustic lens data
    if plot_data_0deg_v1:
        peaks_percentage_plot, half_peak_loc_left_plot, half_peak_loc_right_plot, passive_flaw_width_plot, xspan_plot = plot_data_0deg_v1

        fig, ax = plt.subplots(figsize=(linewidth * .49, linewidth * .4))
        plt.plot(xspan_plot, peaks_percentage_plot, ":o", color="k")
        plt.xlabel("Position along passive direction / (mm)")
        plt.ylabel(r"Normalized amplitude / (\%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
        plt.ylim([-10, 110])
        plt.yticks(np.arange(0, 125, 25))
        plt.xlim([-8, 8])
        ytemp = np.arange(20, 60, 1)
        plt.xticks(np.linspace(-4, 4, 5))
        plt.grid(alpha=.5)
        plt.plot(half_peak_loc_left_plot * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
        plt.plot(half_peak_loc_right_plot * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

        ax.annotate("", xy=(half_peak_loc_left_plot, 25), xytext=(half_peak_loc_right_plot, 25),
                    arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
                    ha="center", va="bottom")
        ax.annotate(rf"${passive_flaw_width_plot:.2f}$ mm", xy=(0.22, 25), xytext=(0.22, 30),
                    ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("../figures/passive_dir_resolution.pdf")
        plt.show()

    # ---
    # Plot comparing single-element and acoustic lens FWHM at different angles

    fig, ax = plt.subplots(figsize=(linewidth * .49, 2.6))
    plt.errorbar(ANGULAR_LOCATIONS, np.mean(passive_flaw_widths_results, axis=1),
                 np.std(passive_flaw_widths_results, axis=1), color='r', ls='None',
                 marker='o', capsize=5, capthick=1, ecolor='black', markersize=3, label="Acoustic lens")

    # Single-element result:
    x_mono = np.arange(-10, 50, 1)
    y_mono = np.ones_like(x_mono) * np.mean(fwhm_mono_results)
    std_mono = np.std(fwhm_mono_results)

    plt.plot(x_mono, y_mono, "-", color="#00CD6C", linewidth=2, label="Single-element")
    ax.fill_between(x_mono, y_mono - std_mono, y_mono + std_mono, alpha=0.2, color='#00CD6C')
    plt.xlabel(r"RBH position / (degrees)")
    plt.ylabel("FWHM along passive direction / (mm)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.1f}"))
    plt.ylim([1, 4])
    plt.yticks(np.arange(1, 4, .5))
    plt.xticks(np.linspace(0, 40, 5))
    plt.xlim([-5, 45])
    plt.grid(alpha=.5)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("../figures/passive_dir_resolution_different_angles.pdf")
    plt.show()