import numpy as np
import matplotlib
from framework.post_proc import envelope
from framework import file_civa
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pipe_lens_imaging.pwi_tfm_contact import cpwc_contact_kernel
import time


if __name__ == '__main__':
    #%% Data-input:
    data_insp = file_civa.read("../data/contact.civa")
    pwi_data = data_insp.ascan_data[..., 0]

    #%% User-input:

    # Transducer:
    xt = data_insp.probe_params.elem_center[:, 0] * 1e-3
    zt = data_insp.probe_params.elem_center[:, 2] * 1e-3

    fs = data_insp.inspection_params.sample_freq * 1e6

    gate_start = data_insp.inspection_params.gate_start * 1e6

    # Interfaces:

    c = data_insp.specimen_params.cl

    # ROI:
    xroi = np.linspace(-30, 30, 600) * 1e-3
    zroi = np.linspace(0, 50, 500) * 1e-3

    steering_angs = np.radians(data_insp.inspection_params.angles)

    #%%
    ti = time.time()
    img1 = cpwc_contact_kernel(pwi_data, xroi, zroi, xt, zt, steering_angs, c, fs, gate_start)
    tf = time.time()
    print(f"Elapsed-time = {tf - ti:.2f}")

    ti = time.time()
    img2 = cpwc_contact_kernel(pwi_data, xroi, zroi, xt, zt, steering_angs, c, fs, gate_start)
    tf = time.time()
    print(f"Elapsed-time = {tf - ti:.2f}")

    img1_db = 20 * np.log10(envelope(img1, axis=0) + 1e-6)

    plt.figure()
    plt.imshow(img1_db, cmap='jet', vmin=-60, vmax=0)
    plt.show()

    img2_db = 20 * np.log10(envelope(img2, axis=0) + 1e-6)

    plt.figure()
    plt.imshow(img2_db, cmap='jet', vmin=-60, vmax=0)
    plt.show()
