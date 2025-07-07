from numpy import sqrt

from pipe_lens_imaging.geometric_utils import circle_cartesian

class Pipeline:
    def __init__(self, outer_radius: float, wall_thickness: float, c: float, rho: float, xcenter: float = 0., zcenter: float = 0.):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_width = wall_thickness
        self.c = c
        self.rho = rho
        self.xcenter = xcenter
        self.zcenter = zcenter

        self.xint, self.zint = circle_cartesian(self.inner_radius, xcenter, zcenter)
        self.xout, self.zout = circle_cartesian(self.outer_radius, xcenter, zcenter)

    def xy_from_alpha(self, alpha: float) -> [float, float]:
        raise NotImplementedError

    def dydx_from_alpha(self, alpha: float) -> float:
        raise NotImplementedError

    def dydx(self, x, mode='full'):
        dy = -(x - self.xcenter)
        dx = sqrt(self.outer_radius ** 2 - (x - self.xcenter) ** 2)
        if mode == "full":
            return dy/dx
        elif mode == 'partial':
            return dy, dx