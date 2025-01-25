from numpy import sin, cos, pi, arange, sqrt

class Pipeline:
    def __init__(self, outer_radius: float, wall_width: float, c: float):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_width
        self.wall_width = wall_width
        self.c = c

        self.xint, self.zint = circle_cartesian(self.inner_radius)
        self.xout, self.zout = circle_cartesian(self.outer_radius)

    def xy_from_alpha(self, alpha: float) -> [float, float]:
        raise NotImplementedError

    def dydx_from_alpha(self, alpha: float) -> float:
        raise NotImplementedError

    def dydx(self, x):
        return -x / sqrt(self.outer_radius ** 2 - x ** 2)

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
