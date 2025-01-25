from numpy import cos, sin, sqrt, pi

__all__ = ['AcousticLens']

class AcousticLens:
    def __init__(self, d: float, c1: float, c2: float, tau: float):
        """

        :param d: distance between the transducer and pipeline center in m.
        :param c1: longitudinal wave speed on wedge in m/s.
        :param c2:  longitudinal wave speed on coupling medium in m/s.
        :param tau: isocronic time of flight in seconds.
        """

        self.d = d
        self.c1 = c1
        self.c2 = c2
        self.tau = tau

        self.max_alpha = pi/4
        self.a = (c1/c2)**2 - 1
        self.b = lambda alpha : 2*d*cos(alpha) - 2 * tau * c1**2/c2
        self.c = (c1*tau)**2 - d**2

    def z(self, alpha):
        """
        This function computes the acoustic lens in polar coordinate.

        :param alpha: pipeline inspection angle in rad.
        :return:
        """

        return (-self.b(alpha) - sqrt(self.b(alpha)**2 - 4 * self.a * self.c)) / (2 * self.a)

    def dzda(self, alpha):
        """
        This function computes the acoustic lens derivative in polar coordinates.

        :param alpha: pipeline inspection angle in rad.
        :return: derivative value of z(alpha) in polar coordinates.
        """
        return -1/(2 * self.a) * (-2 * self.d * sin(alpha) + 1/2 * 1/sqrt(self.b(alpha)**2 - 4 * self.a * self.c) * (-4 * self.b(alpha) * self.d * sin(alpha)))

    def xy_from_alpha(self, alpha):
        """Computes the (x,y) coordinates of the lens for a given pipeline angle"""
        z = self.z(alpha)
        y = z * cos(alpha)
        x = z * sin(alpha)
        return x, y

    def dydx_from_alpha(self, alpha):
        """Computes the slope (dy/dx) of the lens for a given alpha"""
        alpha_ = pi / 2 - alpha
        z = self.z(alpha)
        dydx = (-self.dzda(alpha) * sin(alpha_) + z * cos(alpha_)) / (-self.dzda(alpha) * cos(alpha_) - z * sin(alpha_))
        return dydx

    def pipeline2steering_angle(self, alpha):
        # alpha : pipeline angle
        # beta: steering angle
        x, y = self.z(alpha) * cos(alpha), self.z(alpha) * sin(alpha)
        r = sqrt(x**2 + (self.d - y)**2)
        beta = self.z(alpha) * alpha / r
        return beta






