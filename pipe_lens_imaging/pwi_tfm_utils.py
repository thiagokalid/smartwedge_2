from numpy import sin, cos, sqrt
import numba

@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    return zc - sqrt(r**2 - (x - xc)**2)

# Functions related to finding entry point during TFM step:
@numba.njit(fastmath=True)
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

# Functions related to finding entry point during plane-wave step:
@numba.njit(fastmath=True)
def cost_fun(x, theta, xa, za, xc, zc, r, c1, c2):
    z = f_circ(x, xc, zc, r)
    dzdx = -(-x + xc)/sqrt(r**2 - (x - xc)**2)
    den = sqrt((x - xa) ** 2 + (z - za) ** 2)
    return dzdx * (c2 / c1 * (z - za) / den - cos(theta)) + (c2 / c1 * (x - xa) / den - sin(theta))

@numba.njit(fastmath=True)
def cost_fun_prime(x, theta, xa, za, xc, zc, r, c1, c2):
    dfdx = -(-x + xc)*(c2*(-x + xa + (-x + xc)*(-za + zc - sqrt(r**2 - (x - xc)**2))/sqrt(r**2 - (x - xc)**2))*(-za + zc - sqrt(r**2 - (x - xc)**2))/(c1*((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2)**(3/2)) - c2*(-x + xc)/(c1*sqrt(r**2 - (x - xc)**2)*sqrt((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2)))/sqrt(r**2 - (x - xc)**2) + (-cos(theta) + c2*(-za + zc - sqrt(r**2 - (x - xc)**2))/(c1*sqrt((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2)))/sqrt(r**2 - (x - xc)**2) - (-x + xc)*(x - xc)*(-cos(theta) + c2*(-za + zc - sqrt(r**2 - (x - xc)**2))/(c1*sqrt((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2)))/(r**2 - (x - xc)**2)**(3/2) + c2*(x - xa)*(-x + xa + (-x + xc)*(-za + zc - sqrt(r**2 - (x - xc)**2))/sqrt(r**2 - (x - xc)**2))/(c1*((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2)**(3/2)) + c2/(c1*sqrt((x - xa)**2 + (-za + zc - sqrt(r**2 - (x - xc)**2))**2))
    return dfdx

@numba.njit(fastmath=True)
def newton_pwi(theta, xa, za, c1, c2, xc, zc, radius, x0 = 0, maxiter: int = 10, tol: float = 1e-6):
    for i in range(maxiter):
        f = cost_fun(x0, theta, xa, za, xc, zc, radius, c1, c2)
        dfdx = cost_fun_prime(x0, theta, xa, za, xc, zc, radius, c1, c2)

        x = x0 - f/dfdx
        if abs(x - x0) < tol:
            break
        else:
            x0 = x
    if i == maxiter:
        print("Maxiter reached")
    return x
