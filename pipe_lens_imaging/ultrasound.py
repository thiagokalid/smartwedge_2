from numpy import cos, sin, abs, arcsin, power, pi
import numpy as np

def liquid2solid_t_coeff(theta_p1, theta_p2, cp1, cp2, cs2, rho1, rho2):
    theta_p1 = abs(theta_p1)
    theta_p2 = abs(theta_p2)

    # Equation 6.118:
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Equation 6.120 from Fundamentals of Ultrasonic (Schmerr 2016)
    # Delta 1 is analogous with the acoustic impedance Z1:
    delta1 = cos(theta_p2)
    # Delta 2 is analogous with the acoustic impedance Z2:
    delta2 = \
        (rho2 * cp2 * cos(theta_p1)) / (rho1 * cp1) * (
                power(cos(2 * theta_s2), 2) + (cs2 ** 2 * sin(2 * theta_s2) * sin(2 * theta_p2)) / (
                cp2 ** 2)
        )

    delta = delta1 + delta2

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type P (pressure):
    Tpp = -(2 * rho2 * cp2 * cos(theta_p1) * cos(2 * theta_s2)) / (rho1 * cp1 * delta)  # Equation 6.122a

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type S (shear):
    Tsp = (4 * rho2 * cs2 * cos(theta_p1) * cos(theta_p2) * sin(theta_s2)) / (
            rho1 * cp1 * delta)  # Equation 6.122a

    return Tpp, Tsp

def solid2solid_t_coeff(theta_p1, theta_p2, cp1, cp2, cs1, cs2, rho1, rho2):
    theta_p1 = abs(theta_p1)
    theta_p2 = abs(theta_p2)

    # Equation 6.118:
    theta_s1 = arcsin(cs1 * sin(theta_p1) / cp1)
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Equation 6.120 from Fundamentals of Ultrasonic (Schmerr 2016)
    # Delta 1 is analogous with the acoustic impedance Z1:
    delta1 = (cp1 * cos(theta_p2))/(cp2 * cos(theta_p1)) * (
        cos(2 * theta_s1)**2 + (cs1**2 * sin(2*theta_s1) * sin(2*theta_p1))/(cp1**2)
    )
    # Delta 2 is analogous with the acoustic impedance Z2:
    delta2 = (rho2/rho1 * (
        cos(2*theta_s2)**2 + (cs2**2 * sin(2 * theta_s2) * sin(2*theta_p2))/(cp2**2)
    ))

    delta = delta1 + delta2

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type P (pressure):
    Tpp = -(2 * rho2 * cp2 * cos(theta_p1) * cos(2 * theta_s2)) / (rho1 * cp1 * delta)  # Equation 6.122a

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type S (shear):
    Tsp = (4 * rho2 * cs2 * cos(theta_p1) * cos(theta_p2) * sin(theta_s2)) / (
            rho1 * cp1 * delta)  # Equation 6.122a

    return Tpp, Tsp

def sinc(x):
    return sin(x) / x

def far_field_directivity(k, a, theta_rad):
    return abs(sinc(k * a * sin(theta_rad) / 2))  # Valor muito pequeno, é isso mesmo

def __far_field_directivity_solid(theta, cl, cs, k):
    F0 = lambda zeta: np.abs((2*zeta**2 - (cl/cs)**2)**2 - 4*zeta**2 * (zeta**2 - 1)**(1/2) * (zeta**2 - (cl/cs)**2)**(1/2))
    Dl = ((cl/cs)**2 - 2 * sin(theta)**2) * cos(theta) / (F0(np.complex64(sin(theta))))
    # Ds = (cl/cs)**(5/2) * (((cl/cs)**2 * sin(theta)**2 - 1)**(1/2) * sin(2*theta)) / (F0(np.complex64(k * sin(theta))))
    return Dl

def far_field_directivity_solid(theta, cl, cs, k, a):
    Dl = __far_field_directivity_solid(theta, cl, cs, k)
    Df = far_field_directivity(k, a, theta)

    return Df * Dl



# if False:
#     import matplotlib
#     matplotlib.use("TkAgg")
#     import matplotlib.pyplot as plt
#     x = np.complex64(np.radians(np.arange(0, 90)))
#
#     plt.plot(np.degrees(x), far_field_directivity_solid(x, 6300, 6300/2, k, a))
#     plt.show()
#
#
#     D = lambda theta: ((2) ** 2 - 2 * sin(theta) ** 2) * cos(theta) / (F0(np.complex64(sin(theta))))
#     plt.plot(x * 180/pi, D(x))
#     plt.show()