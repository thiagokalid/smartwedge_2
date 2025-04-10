from numpy import cos, sin, sqrt, pi, arcsin

__all__ = ['AcousticLens']

class AcousticLens:
    def __init__(self, c1: float, c2: float, d: float, alpha_max: float, alpha_0: float, h0: float):
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

        x0, z0 = h0 * cos(pi/2 - alpha_0), h0 * sin(pi/2 - alpha_0)
        xt, zt = 0, d

        self.T = sqrt((x0 - xt)**2 + (z0 - zt)**2)/self.c1 + self.h0/self.c2

        self.alpha_max = alpha_max
        self.a = (c1/c2)**2 - 1
        self.b = lambda alpha : 2 * d * cos(alpha) - 2 * self.T * c1 ** 2 / c2
        self.c = (c1 * self.T) ** 2 - d ** 2

    def h(self, alpha):
        """
        This function computes the acoustic lens in polar coordinate.

        :param alpha: pipe sector angle in rad.
        :return:
        """

        return (-self.b(alpha) - sqrt(self.b(alpha)**2 - 4 * self.a * self.c)) / (2 * self.a)

    def dhda(self, alpha):
        """
        This function computes the acoustic lens derivative in polar coordinates.

        :param alpha: pipe inspection angle in rad.
        :return: derivative value of z(alpha) in polar coordinates.
        """
        return -1/(2 * self.a) * (-2 * self.d * sin(alpha) + 1/2 * 1/sqrt(self.b(alpha)**2 - 4 * self.a * self.c) * (-4 * self.b(alpha) * self.d * sin(alpha)))

    def xy_from_alpha(self, alpha):
        """Computes the (x,y) coordinates of the lens for a given pipe angle"""
        z = self.h(alpha)
        y = z * cos(alpha)
        x = z * sin(alpha)
        return x, y

    def dydx_from_alpha(self, alpha):
        """Computes the slope (dy/dx) of the lens for a given alpha"""
        alpha_ = pi / 2 - alpha
        z = self.h(alpha)
        dydx = (-self.dhda(alpha) * sin(alpha_) + z * cos(alpha_)) / (-self.dhda(alpha) * cos(alpha_) - z * sin(alpha_))
        return dydx

    def pipe2steering_angle(self, alpha):
        # alpha : pipe angle
        # beta: steering angle
        x, y = self.h(alpha) * sin(alpha), self.h(alpha) * cos(alpha)
        r = sqrt(x**2 + (self.d - y)**2)
        beta = arcsin(self.h(alpha) * sin(alpha) / r)
        return beta






