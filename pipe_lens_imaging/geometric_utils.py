import numba

import numpy as np

from numpy import sin, cos, pi, arange, sqrt

@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    return zc - sqrt(r**2 - (x - xc)**2)

def rotate(x, y, angle, shift_x=0, shift_y=0):
    newx = x * cos(angle) - y * sin(angle) + shift_x
    newy = x * sin(angle) + y * cos(angle) + shift_y
    return (newx, newy)

def dxdy_tube(x, r_circ):
    '''Computes the slope of the cilinder (circle) for a given x'''
    return -x / sqrt(r_circ ** 2 - x ** 2)

def circle_cartesian(r, xcenter=0.0, zcenter=0.0, angstep=1e-2,):
    alpha = arange(-pi, pi + angstep, angstep)
    x, z = pol2cart(r, alpha)
    return x + xcenter, z + zcenter

def pol2cart(rho, phi):
    z = rho * cos(phi)
    x = rho * sin(phi)
    return x, z

def line_equation_polar(alpha, a, b, eps=1e-12):
    denom = np.cos(alpha) - a * np.sin(alpha)
    # Ensure denominator is not too close to zero
    safe_denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
    return b / safe_denom

def findIntesectionBetweenCurveAndLine(a_line, b_line, x_q, xlens, zlens):
    num_alphas = len(x_q)

    intersection_x = np.empty_like(x_q)
    intersection_z = np.empty_like(x_q)

    xx = [None] * num_alphas
    zz = [None] * num_alphas

    for ray in range(num_alphas):
        xx[ray] = np.linspace(x_q[ray] - 0.15, x_q[ray] + 0.15, num_alphas)

        # Utiliza várias coordenadas x (linspace) para encontrar vários valores da reta "de reflexão"
        zz[ray] = a_line[ray] * xx[ray] + b_line[ray]

        _x_intersection, _y_intersect = find_line_curve_intersection(xx[ray], zz[ray], xlens, zlens)

        intersection_x[ray] = _x_intersection
        intersection_z[ray] = _y_intersect

    return intersection_x, intersection_z


def findIntersectionBetweenAcousticLensAndRay(a_ray, b_ray, acoustic_lens, tol=1e-3):
    # Finding an initial guess:
    alpha_step = np.radians(.1)
    ang_span = np.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step)

    # Iterate over each ray:
    alpha_root = np.zeros_like(a_ray)
    for ray in range(len(a_ray)):
        _a_ray, _b_ray = a_ray[ray], b_ray[ray]
        x1, y1 = pol2cart(line_equation_polar(ang_span, _a_ray, _b_ray), ang_span)
        x2, y2 = acoustic_lens.xy_from_alpha(ang_span)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        alpha_0 = ang_span[dist.argmin()]

        ang_span_finer = np.arange(alpha_0 - alpha_step, alpha_0 + alpha_step, alpha_step/100)
        x1, y1 = pol2cart(line_equation_polar(ang_span_finer, _a_ray, _b_ray), ang_span_finer)
        x2, y2 = acoustic_lens.xy_from_alpha(ang_span_finer)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        alpha_root[ray] = ang_span_finer[dist.argmin()] if dist.min() <= tol else np.nan


    return alpha_root

