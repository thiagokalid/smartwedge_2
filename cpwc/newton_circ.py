import numba

from numpy import sqrt


# xa, za : element center in meter
# xf, zf : focus location in meter
# c1, c2 : speed at coupling medium and pipe
# xc, zc : tube center location
# r: radius of the tube

@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    return zc - sqrt(r**2 - (x - xc)**2)

def tof(x, xa, za, xf, zf, c1, c2, xc, zc, r):
    z  = zc - sqrt(r**2 - (x - xc)**2)
    tof = (sqrt((xa - x) ** 2 + (za - z) ** 2) / c1 + sqrt((x - xf) ** 2 + (z - zf) ** 2) / c2)
    return tof * 1e6

@numba.njit(fastmath=True)
def dtof_dx(x, xa, za, xf, zf, c1, c2, xc, zc, r):
    dtau_dx = (x - xf - (-x + xc) * (zc - zf - sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(r ** 2 - (x - xc) ** 2)) / (
                c2 * sqrt((x - xf) ** 2 + (zc - zf - sqrt(r ** 2 - (x - xc) ** 2)) ** 2)) + (
                x - xa + (-x + xc) * (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(r ** 2 - (x - xc) ** 2)) / (
                c1 * sqrt((-x + xa) ** 2 + (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) ** 2))
    return dtau_dx * 1e6

@numba.njit(fastmath=True)
def d2tof_dx2(x, xa, za, xf, zf, c1, c2, xc, zc, r):
    dt2dx2 = (1 + (x - xc) ** 2 / (r ** 2 - (x - xc) ** 2) - (-zc + zf + sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(
        r ** 2 - (x - xc) ** 2) - (x - xc) ** 2 * (-zc + zf + sqrt(r ** 2 - (x - xc) ** 2)) / (
                 r ** 2 - (x - xc) ** 2) ** (3 / 2)) / (
                c2 * sqrt((x - xf) ** 2 + (-zc + zf + sqrt(r ** 2 - (x - xc) ** 2)) ** 2)) - (
                -x + xf + (x - xc) * (-zc + zf + sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(r ** 2 - (x - xc) ** 2)) ** 2 / (
                c2 * ((x - xf) ** 2 + (-zc + zf + sqrt(r ** 2 - (x - xc) ** 2)) ** 2) ** (3 / 2)) + (
                1 + (x - xc) ** 2 / (r ** 2 - (x - xc) ** 2) - (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(
            r ** 2 - (x - xc) ** 2) - (x - xc) ** 2 * (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) / (
                            r ** 2 - (x - xc) ** 2) ** (3 / 2)) / (
                c1 * sqrt((x - xa) ** 2 + (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) ** 2)) - (
                -x + xa + (x - xc) * (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) / sqrt(r ** 2 - (x - xc) ** 2)) ** 2 / (
                c1 * ((x - xa) ** 2 + (za - zc + sqrt(r ** 2 - (x - xc) ** 2)) ** 2) ** (3 / 2))
    return dt2dx2 * 1e6

def newton(f, dfdx, x0 = 0, maxiter: int = 10, tol: float = 1e-6):
    for i in range(maxiter):
        x = x0 - f(x0)/dfdx(x0)
        if abs(x - x0) < tol:
            break
        else:
            x0 = x
    return x

@numba.njit(fastmath=True)
def newton_circ(xa, za, xf, zf, c1, c2, xc, zc, radius, x0 = 0, maxiter: int = 10, tol: float = 1e-6):
    for i in range(maxiter):
        f = dtof_dx(x0, xa, za, xf, zf, c1, c2, xc, zc, radius)
        dfdx = d2tof_dx2(x0, xa, za, xf, zf, c1, c2, xc, zc, radius)

        x = x0 - f/dfdx
        if abs(x - x0) < tol:
            break
        else:
            x0 = x
    return x

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    xa, za = 25e-3, 0
    xf, zf = 0, 40e-3

    r = 70e-3
    waterpath = 30e-3
    xc, zc = 0, (r + waterpath)

    c1, c2 = 1483, 5900

    f = lambda x: dtof_dx(x, xa, za, xf, zf, c1, c2, xc, zc, r)
    dfdx = lambda x: d2tof_dx2(x, xa, za, xf, zf, c1, c2, xc, zc, r)

    x0 = newton(f, dfdx, 0, maxiter=10, tol=1e-6)

    xspan = np.linspace(-r, r, 1000)
    plt.plot(xspan, f_circ(xspan, xc, zc, r), linewidth=3, color='k')
    plt.plot(xa, za, 'ks')
    plt.plot(xf, zf, 'ro')
    plt.plot([xa, x0, xf], [za, f_circ(x0, xc, zc, r), zf], 'lime', linewidth=1)


