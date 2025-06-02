import numpy as np
import matplotlib
from framework.post_proc import envelope, normalize
from framework import file_m2k
import matplotlib.pyplot as plt
from file_law import file_law
from pipe_lens_imaging.pwi_tfm import compute_delaylaw_pwi
matplotlib.use('TkAgg')


if __name__ == '__main__':
    #%% Data-input:
    data_insp = file_m2k.read("../data/pwi_diy_teste.m2k", freq_transd=5, bw_transd=.5, tp_transd='gaussian', read_ascan=True)
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

    # Steering angles for delay-law:
    steering_angs = np.radians(data_insp.inspection_params.angles)

    radius = 140e-3 / 2
    waterpath = 32e-3
    wall_thickness = 17.23e-3 + 5e-3

    xcenter, zcenter = 0, waterpath + radius


    # Compute delay-law:
    pwi_tof, _, _ = compute_delaylaw_pwi(steering_angs, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
    t_ref = np.max(pwi_tof)
    delay_law = t_ref - pwi_tof


    # Plot law for a given i-th angle:
    i = 140
    plt.figure(figsize=(8, 4))
    plt.title(fr"Delay-law for PW at ${np.degrees(steering_angs[i]):.2f}^\circ$ ")
    plt.bar(np.arange(1, 64 + 1), delay_law[i, :] * 1e6, width=1, color='r', edgecolor='k')
    plt.xlim([-4, 64 + 4])
    plt.xticks(np.arange(0, 64 + 8, 8))
    # plt.ylim([-1, np.max(delay_law[:, :]) * 1e6 + 1])
    plt.ylabel(r"Delay / $\mathrm{(\mu s)}$")
    plt.xlabel("Element")
    plt.show()


    # Save delay-law:
    root = "../delay_law/"
    filename = "testes.law"
    path = root + filename
    file_law.write(path, emission=delay_law, reception=delay_law)

    # Display parameters.
    print(f"""
    Lei focal calculada para ondas planas.
    -----------------------------------------------------------------------------------
    Ângulos de varredura (dentro do tubo): de {np.degrees(steering_angs[0]):.2f}° a {np.degrees(steering_angs[-1]):.2f}° com passo de {np.degrees(steering_angs[0] - steering_angs[1]):.2f}°.
    Tubo com raio de {radius * 1e3:.2f} mm, coluna d'água de {waterpath * 1e3 :.2f} mm.
    Velocidade de propagação no meio acoplante de {c_coupling:.2f} m/s e no tubo de {c_specimen:.2f} m/s.
    Transdutor com {len(xt):.0f} elementos e {(xt[1]-xt[0]) * 1e3:.2f} mm de pitch.
    -----------------------------------------------------------------------------------
    O arquivo foi salvo em '{path}'.
    """)