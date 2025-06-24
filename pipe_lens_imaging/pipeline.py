from numpy import sqrt

from pipe_lens_imaging.geometric_utils import circle_cartesian

class Pipeline:
    def __init__(self, outer_radius: float, wall_thickness: float, c: float, rho: float, x_center: float = 0., z_center: float = 0.):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_width = wall_thickness
        self.c = c
        self.rho = rho
        self.x_center = x_center
        self.z_center = z_center

        self.xint, self.zint = circle_cartesian(self.inner_radius)
        self.xout, self.zout = circle_cartesian(self.outer_radius)

    def xy_from_alpha(self, alpha: float) -> [float, float]:
        raise NotImplementedError

    def dydx_from_alpha(self, alpha: float) -> float:
        raise NotImplementedError

    def dydx(self, x):
        return -(x - self.x_center) / sqrt(self.outer_radius ** 2 - (x - self.x_center) ** 2)
