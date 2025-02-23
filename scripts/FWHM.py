from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm

#%%

root = '../data/fwhm/'
data = file_m2k.read(root + "xl_step2_degree.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian')
# data_ref = file_m2k.read(root + "ref.m2k", type_insp='contact', water_path=0, freq_transd=5,
#                              bw_transd=0.5, tp_transd='gaussian')

#%%

log_cte = 1e-6
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)
vmax = 0
vmin = -120
sweep_angs = np.arange(-6, 44, 2)

# S_scan para conferir o crop
s_scan = envelope(np.sum(data.ascan_data[..., 0], axis=2), axis=0)
sscan_db = 20 * np.log10(s_scan / s_scan.max() + log_cte)

plt.figure()
plt.imshow(sscan_db, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax,
           extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]])
plt.show()

# Define o r_span para API em mm²
r_span = convert_time2radius(time_grid, 55.27, 60.75, 5.9, 1.483, 1.483)
r1 = 50
r2 = 70

m = 0



first_time = True

n_shots = 25

#
plot_step = 6
mmax = int(np.ceil(n_shots / plot_step))


# Seta os vetores dos dos dados
widths, heights, maximums, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)

plt.figure()
for i in tqdm(range(n_shots)):
    channels = data.ascan_data[:, :, :, i]
    sscan = np.sum(channels, axis=2)
    sscan_db = 20 * np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

    # Aplica API para descobrir a área acima de -6 dB
    corners = [(62, -6.5 + 2 * i), (58.0, 6.5 + 2 * i)]
    widths[i], heights[i], maximums[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

    if i % plot_step == 0:
        m += 1
        plt.subplot(2, mmax, m)
        plt.title(f"S-scan da posição {i}")
        plt.pcolormesh(ang_span, r_span, sscan_db, cmap='magma', vmin=vmin, vmax=vmax)

        if first_time:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")

        plt.subplot(2, mmax, m + mmax)
        plt.pcolormesh(ang_span, r_span, pixels_above_threshold, vmin=0, vmax=1)


        if first_time:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")
        first_time = False
plt.show()

#%%


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
# plt.ylim([0.015, 0.293])
plt.suptitle(root)
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.title("Maximum Pixel Value in linear scale")
plt.plot(sweep_angs, maximums, 'o', color='b', label="XL")
plt.xticks(sweep_angs)
plt.ylabel("Pixel intensity")
plt.xlabel(r"Tube sector angle ($\alpha$) in [degrees]")
plt.grid()
plt.xticks(np.arange(-6, 45, 4))
plt.tight_layout()
plt.ylim([5e3, 2.138e5])
