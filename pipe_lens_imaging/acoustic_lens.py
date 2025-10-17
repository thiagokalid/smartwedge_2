import numpy as  np
from numpy import cos, sin, sqrt, pi, arcsin, linspace

__all__ = ['AcousticLens']

from pipe_lens_imaging.geometric_utils import line_equation_polar

def roots_bhaskara(a, b, c):
    sqrt_delta = np.sqrt(np.square(b) - 4 * a * c)
    den = 2 * a
    x1 = (-b + sqrt_delta) / den
    x2 = (-b - sqrt_delta) / den

    return x1, x2


class AcousticLens:
    def __init__(self, c1: float, c2: float, d: float, alpha_max: float, alpha_0: float, h0: float, rho1: float, rho2: float):
        """
        :param c1: Speed of sound (acoustic lens) in (m/s)
        :param c2: Speed of sound (coupling medium) in (m/s)
        :param d: Height of transducer in (m)
        :param alpha_max: Maximum sectorial angle in (rad)
        :param alpha_0: Reference angle (boundary condition) in (rad)
        :param h0: Length h chosen at the reference angle in (m)
        """

        self.d = d
        self.c1 = c1
        self.c2 = c2
        self.alpha_0 = alpha_0
        self.h0 = h0
        self.l0 = self.d - self.h0
        self.rho1 = rho1
        self.rho2 = rho2

        x0, z0 = h0 * cos(pi/2 - alpha_0), h0 * sin(pi/2 - alpha_0)
        xt, zt = 0, d

        self.T = sqrt((x0 - xt)**2 + (z0 - zt)**2)/self.c1 + self.h0/self.c2

        self.alpha_max = alpha_max
        self.a = (c1/c2)**2 - 1
        self.b = lambda alpha : 2 * d * cos(alpha) - 2 * self.T * c1 ** 2 / c2
        self.c = (c1 * self.T) ** 2 - d ** 2

        self.xlens, self.zlens = self.xy_from_alpha(linspace(-self.alpha_max, self.alpha_max, 1000))

    def h(self, alpha):
        return (-self.b(alpha) - sqrt(self.b(alpha) ** 2 - 4 * self.a * self.c)) / (2 * self.a)

    def dhda(self, alpha):
        """
        This function computes the acoustic lens derivative in polar coordinates.

        :param alpha: pipe inspection angle in rad.
        :return: derivative value of z(alpha) in polar coordinates.
        """
        return -1 / (2 * self.a) * (
                    -2 * self.d * sin(alpha) + 1 / 2 * 1 / sqrt(self.b(alpha) ** 2 - 4 * self.a * self.c) * (
                        -4 * self.b(alpha) * self.d * sin(alpha)))

    def xy_from_alpha(self, alpha):
        """Computes the (x,y) coordinates of the lens for a given pipe angle"""
        z = self.h(alpha)
        y = z * cos(alpha)
        x = z * sin(alpha)
        return x, y

    def dydx_from_alpha(self, alpha, mode='full'):
        """Computes the slope (dy/dx) of the lens for a given alpha"""

        h_ = self.h(alpha)

        # Equations (A.19a) and (A.19b) in Appendix A.2.2.
        dy = self.dhda(alpha) * np.cos(alpha) - h_ * np.sin(alpha)
        dx = self.dhda(alpha) * np.sin(alpha) + h_ * np.cos(alpha)

        if mode == 'full':
            return  dy/dx
        elif mode == 'partial':
            return dy, dx
        else:
            raise ValueError("mode must be 'full' or 'parts'")

    def pipe2steering_angle(self, alpha):
        # alpha : pipe angle
        # beta: steering angle
        x, y = self.h(alpha) * sin(alpha), self.h(alpha) * cos(alpha)
        r = sqrt(x**2 + (self.d - y)**2)
        beta = arcsin(self.h(alpha) * sin(alpha) / r)
        return beta




