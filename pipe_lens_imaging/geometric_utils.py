import numba
import numpy as np
from numpy import sin, cos, pi, arange, sqrt

__all__ = [
    "f_circ",
    "rotate",
    "dxdy_tube",
    "circle_cartesian",
    "pol2cart",
    "line_equation_polar",
    "findIntersectionBetweenAcousticLensAndRay",
]


@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    return zc - sqrt(r**2 - (x - xc)**2)


def rotate(x, y, angle, shift_x=0, shift_y=0):
    newx = x * cos(angle) - y * sin(angle) + shift_x
    newy = x * sin(angle) + y * cos(angle) + shift_y
    return newx, newy


def dxdy_tube(x, r_circ):
    return -x / sqrt(r_circ ** 2 - x ** 2)


def circle_cartesian(r, xcenter=0.0, zcenter=0.0, angstep=1e-2):
    alpha = arange(-pi, pi + angstep, angstep)
    x, z = pol2cart(r, alpha)
    return x + xcenter, z + zcenter


def pol2cart(rho, phi):
    z = rho * cos(phi)
    x = rho * sin(phi)
    return x, z


def line_equation_polar(alpha, a, b, eps=1e-12):
    denom = np.cos(alpha) - a * np.sin(alpha)
    safe_denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
    return b / safe_denom


def findIntersectionBetweenAcousticLensAndRay(a_ray, b_ray, acoustic_lens, tol=1e-3):
    alpha_step = np.radians(0.1)
    ang_span = np.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step)

    alpha_root = np.zeros_like(a_ray)
    for ray in range(len(a_ray)):
        _a_ray, _b_ray = a_ray[ray], b_ray[ray]
        x1, y1 = pol2cart(line_equation_polar(ang_span, _a_ray, _b_ray), ang_span)
        x2, y2 = acoustic_lens.xy_from_alpha(ang_span)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        alpha_0 = ang_span[dist.argmin()]

        ang_span_finer = np.arange(alpha_0 - alpha_step, alpha_0 + alpha_step, alpha_step / 100)
        x1, y1 = pol2cart(line_equation_polar(ang_span_finer, _a_ray, _b_ray), ang_span_finer)
        x2, y2 = acoustic_lens.xy_from_alpha(ang_span_finer)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        alpha_root[ray] = ang_span_finer[dist.argmin()] if dist.min() <= tol else np.nan

    return alpha_root
