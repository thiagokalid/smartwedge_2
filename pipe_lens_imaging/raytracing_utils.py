import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, sqrt

__all__ = [
    "roots_bhaskara",
    "rhp",
    "uhp",
    "snell",
    "refraction",
    "reflection",
    "plot_setup",
    "plot_normal",
    "plot_line"
]


def roots_bhaskara(a, b, c):
    '''Computes the roots of the polynomial ax^2 + bx + c = 0'''

    sqdelta = sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqdelta) / (2 * a)
    x2 = (-b - sqdelta) / (2 * a)
    return x1, x2


def rhp(x):
    '''Projects an angle to the Right Half Plane [-pi/2; pi/2]'''
    x = np.mod(x, pi)
    x = x - (x > pi / 2) * pi
    x = x + (x < -pi / 2) * pi
    return x


def uhp(x):
    '''Projects an angle to the Upper Half Plane [0; pi]'''
    x = rhp(x)
    x = x + (x < 0) * pi
    return x


def snell(v1, v2, gamma1, dydx):
    """Computes the the new angle  after the refraction
  of the first angle with the Snell's law (top-down)"""
    gamma1 = uhp(gamma1)
    slope = rhp(np.arctan(dydx))
    normal = slope + pi / 2
    theta1 = gamma1 - normal
    arg = sin(theta1) * v2 / v1
    bad_index = np.abs(arg) > 1
    # Forcing the argument to be always within the [-1, 1] interval:
    arg[bad_index] = np.tanh(arg[bad_index])
    theta2 = np.arcsin(arg)
    gamma2 = slope - pi / 2 + theta2
    return gamma2, theta1, theta2

def refraction(incidence_phi, dzdx, v1, v2):
    """
    dzdx : tuple or ndarray
    """
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray) or isinstance(dzdx, float):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = np.arcsin((v2 / v1) * np.sin(theta_1))
    refractive_phi = phi_slope - (np.pi / 2) + theta_2

    return refractive_phi, phi_normal, theta_1, theta_2


def reflection(incidence_phi, dzdx):
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray) or isinstance(dzdx, float):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = -theta_1
    reflective_phi = phi_slope - (np.pi / 2) + theta_2
    return reflective_phi, phi_normal, theta_1, theta_2




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