import numpy as np
import matplotlib.pyplot as plt


def plot_setup(acoustic_lens, pipeline, transducer, show=True, legend=True):
    transducer_x = np.arange(transducer.num_elem) * transducer.pitch
    transducer_x = transducer_x - np.mean(transducer_x)
    transducer_y = np.ones_like(transducer_x) * acoustic_lens.d

    x_alpha, z_alpha = acoustic_lens.xlens, acoustic_lens.zlens

    x_pipe = pipeline.xout
    z_pipe = pipeline.zout

    plt.figure()
    plt.plot(transducer_x, transducer_y, 'o', markersize=1, label="Transducer", color="green")
    plt.plot(x_alpha, z_alpha, label="Refracting surface", color="red")
    plt.plot(x_pipe, z_pipe, label="Pipe", color="blue")
    plt.scatter(0, 0, label="Origin (0, 0)", color="orange")
    plt.scatter(0, acoustic_lens.d, label="Transducer's center", color="black")
    if legend:
        plt.legend()
    plt.axis("equal")
    if show:
        plt.show()


def plot_normal(angle, x, z, scale=0.007, color='purple'):
    normal_dx = np.cos(angle)
    normal_dz = np.sin(angle)

    normal_end_x_pos = x + normal_dx * scale
    normal_end_z_pos = z + normal_dz * scale
    normal_end_x_neg = x - normal_dx * scale
    normal_end_z_neg = z - normal_dz * scale
    plt.plot([normal_end_x_neg, normal_end_x_pos],
             [normal_end_z_neg, normal_end_z_pos],
             color, linewidth=1.0, linestyle='-')


def plot_line(angle, x, z, scale=0.007, color='purple', x_pos=True, z_pos=True, x_neg=True, z_neg=True):
    normal_dx = np.cos(angle)
    normal_dz = np.sin(angle)

    normal_end_x_pos = x + normal_dx * scale if x_pos else x
    normal_end_z_pos = z + normal_dz * scale if z_pos else z
    normal_end_x_neg = x - normal_dx * scale if x_neg else x
    normal_end_z_neg = z - normal_dz * scale if z_neg else z

    plt.plot([normal_end_x_neg, normal_end_x_pos],
             [normal_end_z_neg, normal_end_z_pos],
             color, linewidth=1.0, linestyle='-')

