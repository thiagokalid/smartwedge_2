import numpy as np
import matplotlib
from framework.post_proc import envelope, normalize
from framework import file_m2k
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
from pipe_lens_imaging.pwi_tfm import cpwc_circle_kernel
from archive.cpwc.pw_circ import f_circ
import time


if __name__ == '__main__':
    #%% Data-input:
    data_insp = file_m2k.read("../data/pwi_diy_teste.m2k", freq_transd=5, bw_transd=.5, tp_transd='gaussian', read_ascan=True)
    pwi_data = data_insp.ascan_data[..., 0]
    Nt, Nangs, Nel = pwi_data.shape

    plt.figure()
    plt.imshow(pwi_data[:, 0, :], extent=[-.5, 64.5, data_insp.time_grid[-1], data_insp.time_grid[0]], aspect='auto')
    plt.show()

    plt.plot(data_insp.time_grid * 1e-6 / 2 * 1483 * 1e3, pwi_data[:, 140, 32])
    plt.show()

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
    wall_thickness = 17.23e-3

    xcenter, zcenter = 0, waterpath + radius

    # ROI:
    xroi = np.linspace(-35, 35, 700) * 1e-3
    zroi = np.linspace(20, 70, 500) * 1e-3


    #%%
    ti = time.time()
    img, delaylaw = cpwc_circle_kernel(pwi_data, xroi, zroi, xt, zt, xcenter, zcenter, radius, steering_angs, c_coupling, c_specimen, fs, gate_start)
    tf = time.time()
    print(f"Elapsed-time = {tf - ti:.2f}")

    img_env = envelope(img, axis=0)
    img_db = 20 * np.log10(img_env / img_env.max() + 1e-6)


    #%%

    plt.figure(figsize=(8, 6))
    plt.imshow(img_db, cmap='jet', vmin=-50, vmax=0, extent=[xroi[0] * 1e3, xroi[-1] * 1e3, zroi[-1] * 1e3, zroi[0] * 1e3])
    plt.colorbar()

    # Plot surfaces:
    plt.title("PWI-TFM through cylindrical surface.")
    xspan = np.linspace(xroi[0], xroi[-1], 100)
    plt.plot(xspan * 1e3, f_circ(xspan, xcenter, zcenter, radius) * 1e3, 'w--')
    plt.plot(xspan * 1e3, f_circ(xspan, xcenter, zcenter, radius - wall_thickness) * 1e3, 'w--')
    plt.ylim([zroi[-1] * 1e3, zroi[0] * 1e3])

    plt.gca().fill_between([xt[0] * 1e3, xt[-1] * 1e3], [zroi[0] * 1e3, zroi[0] * 1e3],
                           [zroi[-1] * 1e3, zroi[-1] * 1e3], color='r', alpha=.1, label='PA Active Aperture')

    plt.ylim([zroi[-1] * 1e3, zroi[0] * 1e3])
    plt.xlim([xroi[0] * 1e3, xroi[-1] * 1e3])
    plt.legend()
    plt.grid(alpha=.3)
    plt.xlabel(r"x-axis / $\mathrm{\mu s}$")
    plt.ylabel(r"z-axis / $\mathrm{\mu s}$")

    for x, y in zip([20, 1.9, -16], [51.4, 47.5, 52.8]):
        sdh = plt.Circle(xy=(x, y), radius=2, fill=False, edgecolor='w')
        plt.gca().add_patch(sdh)

    plt.tight_layout()
    plt.show()


