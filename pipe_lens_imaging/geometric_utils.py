import numba

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