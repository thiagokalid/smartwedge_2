from numpy import sin, cos, pi, arange, sqrt

def rotate(x, y, angle, shift_x=0, shift_y=0):
    newx = x * cos(angle) - y * sin(angle) + shift_x
    newy = x * sin(angle) + y * cos(angle) + shift_y
    return (newx, newy)

def dxdy_tube(x, r_circ):
    '''Computes the slope of the cilinder (circle) for a given x'''
    return -x / sqrt(r_circ ** 2 - x ** 2)

def circle_cartesian(r, angstep=1e-2):
    alpha = arange(-pi, pi + angstep, angstep)
    return pol2cart(r, alpha)

def pol2cart(rho, phi):
    y = rho * cos(phi)
    x = rho * sin(phi)
    return x, y


