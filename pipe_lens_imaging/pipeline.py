from numpy import sqrt

from pipe_lens_imaging.geometric_utils import circle_cartesian

class Pipeline:
    def __init__(self, outer_radius: float, wall_thickness: float, c: float, rho: float):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_width = wall_thickness
        self.c = c
        self.rho = rho

        self.xint, self.zint = circle_cartesian(self.inner_radius)
        self.xout, self.zout = circle_cartesian(self.outer_radius)

    def xy_from_alpha(self, alpha: float) -> [float, float]:
        raise NotImplementedError

    def dydx_from_alpha(self, alpha: float) -> float:
        raise NotImplementedError

    def dydx(self, x):
        return -x / sqrt(self.outer_radius ** 2 - x ** 2)
