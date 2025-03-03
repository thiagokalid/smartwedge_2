from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/resolution/'
acoustic_lens_types = ["xl", "compact"]
acoustic_lens_type = acoustic_lens_types[1]
plot_fig = True # Chooses to either plot or save the output.

#%%

if acoustic_lens_type == "xl":
    n_shots = 25
elif acoustic_lens_type == "compact":
    n_shots = 40
else:
    raise ValueError("Invalid acoustic lens type")


data = file_m2k.read(root + f"active_dir_{acoustic_lens_type}_2degree.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian')
data_ref = file_m2k.read(root + f"{acoustic_lens_type}_ref.m2k", type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian')

#%%

log_cte = 1e-6
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)
vmax, vmin = 0, -120
sweep_angs = np.linspace(-6, 44, n_shots)

# It was manually identified where the outer and inner surface was (in microseconds):
if acoustic_lens_type == "xl":
    t_outer, t_inner = 55.27, 60.75
    rtop, rbottom = 62.0, 58.0
else:
    t_outer, t_inner = 53.45, 59.30
    rtop, rbottom = 62.0, 58.0

r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

# Define
m = 0
mmax = 4
plot_position = np.linspace(0, 36, mmax)


# Seta os vetores dos dos dados
widths, heights, maximums, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)

plt.figure(figsize=(15.5,9))
for i in tqdm(range(n_shots)):
    channels = data.ascan_data[:, :, :, i]
    channels_ref = np.mean(data_ref.ascan_data, axis=3)
    sscan = np.sum(channels - channels_ref, axis=2)
    sscan_db = 20 * np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

    # Aplica API para descobrir a área acima de -6 dB

    if n_shots == 40:
        current_angle = 1 * i
        corners = [(rtop, -8.5 + current_angle), (rbottom, 8.5 + current_angle)]
    else:
        current_angle = 2 * i
        corners = [(rtop, -8.5 + current_angle), (rbottom, 8.5 + current_angle)]

    widths[i], heights[i], maximums[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

    if current_angle in plot_position:
        m += 1
        plt.subplot(2, mmax, m)
        plt.title(rf"S-scan da posição ${current_angle}\degree$")
        plt.pcolormesh(ang_span, r_span, sscan_db, cmap='magma', vmin=vmin, vmax=vmax)
        plt.ylabel(r"Tempo em $\mu s$")
        plt.xlabel(r"Ângulo de varredura da tubulação")

        plt.subplot(2, mmax, m + mmax)
        plt.pcolormesh(ang_span, r_span, pixels_above_threshold, vmin=0, vmax=1)
        plt.ylabel(r"Tempo em $\mu s$")
        plt.xlabel(r"Ângulo de varredura da tubulação")

if acoustic_lens_type == "xl":
    plt.suptitle("Acoustic lens: XL")
else:
    plt.suptitle("Acoustic lens: compact")
plt.tight_layout()
plt.show()

#%% Plotting the results:

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Scatterer Width and Height based on FWHM")
plt.plot(sweep_angs, widths, 'o', color='g', label="Width")
plt.plot(sweep_angs, heights, 'o', color='r', label="Height")
plt.xticks(sweep_angs)
plt.ylabel("Flaw size in [mm]")
plt.xlabel(r"Tube sector angle ($\alpha$) in [degrees]")
plt.grid()
plt.xticks(np.arange(0, 45, 4))
plt.tight_layout()
plt.ylim([0.0, 9.75])
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Maximum Pixel Value in linear scale")
plt.plot(sweep_angs, maximums, 'o', color='b', label="XL")
plt.xticks(sweep_angs)
plt.ylabel("Pixel intensity")
plt.xlabel(r"Tube sector angle ($\alpha$) in [degrees]")
plt.grid()
plt.xticks(np.arange(-6, 45, 4))
plt.ylim([5e3, 2.138e5])

if acoustic_lens_type == "xl":
    plt.suptitle("Acoustic lens: XL")
else:
    plt.suptitle("Acoustic lens: compact")

plt.tight_layout()

if plot_fig:
    plt.show()
else:
    plt.savefig(f"../figures/active_dir_resolution_{acoustic_lens_type}.pdf")
