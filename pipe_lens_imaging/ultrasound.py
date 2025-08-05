import numpy as np
from numpy import cos, sin, abs, arcsin, power

__all__ = [
    "fluid2solid_t_coeff",
    "solid2fluid_t_coeff",
    "solid2solid_t_coeff",
    "fluid2solid_r_coeff",
    "sinc",
    "far_field_directivity",
    "far_field_directivity_solid"
]

def fluid2solid_t_coeff(theta_p1, theta_p2, cp1_fluid, cp2_solid, cs2_solid, rho1_fluid, rho2_solid):
    theta_p1 = abs(theta_p1)
    theta_p2 = abs(theta_p2)

    # Equation 6.118:
    theta_s2 = arcsin(cs2_solid * sin(theta_p2) / cp2_solid)

    # Equation 6.120 from Fundamentals of Ultrasonic (Schmerr 2016)
    # Delta 1 is analogous with the acoustic impedance Z1:
    delta1 = cos(theta_p2)
    # Delta 2 is analogous with the acoustic impedance Z2:
    delta2 = \
        (rho2_solid * cp2_solid * cos(theta_p1)) / (rho1_fluid * cp1_fluid) * (
                power(cos(2 * theta_s2), 2) + (cs2_solid ** 2 * sin(2 * theta_s2) * sin(2 * theta_p2)) / (
                cp2_solid ** 2)
        )

    delta = delta1 + delta2

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type P (pressure):
    Tpp = -(2 * rho2_solid * cp2_solid * cos(theta_p1) * cos(2 * theta_s2)) / (rho1_fluid * cp1_fluid * delta)  # Equation 6.122a

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type S (shear):
    Tsp = (4 * rho2_solid * cs2_solid * cos(theta_p1) * cos(theta_p2) * sin(theta_s2)) / (
            rho1_fluid * cp1_fluid * delta)  # Equation 6.122a

    return Tpp, Tsp

def solid2fluid_t_coeff(theta_p1, theta_p2, cp1_solid, cs1_solid, cp2_fluid, rho1_solid, rho2_fluid):
    # Compute the transmission coefficient as if the propagation direction was reversed:
    Tpp_reversed, Tsp_reversed = fluid2solid_t_coeff(theta_p2, theta_p1, cp2_fluid, cp1_solid, cs1_solid, rho2_fluid, rho1_solid)

    # Compute the shear-wave incidence angle:
    theta_s1 = arcsin(cs1_solid * sin(theta_p1) / cp1_solid)

    # Compute the true coefficients through Stoke's relation. See 6.150a (Schmerr, 2016).
    Tpp = Tpp_reversed * (rho2_fluid * cp2_fluid * np.cos(theta_p1)) / (rho1_solid * cp1_solid * np.cos(theta_p2))
    Tps = Tsp_reversed * (rho2_fluid * cp2_fluid * np.cos(theta_s1)) / (rho1_solid * cp1_solid * np.cos(theta_p2))

    return Tpp, Tps

def solid2solid_r_coeff(theta_p1, theta_p2, cp1, cp2, cs1, cs2, rho1, rho2):
    """
    Calculates the transmission and reflection coefficients for a solid-solid interface (P-wave incidence).
    Returns Rpp (P-to-P reflection)
    """
    theta_p1 = abs(theta_p1)
    theta_p2 = abs(theta_p2)

    # Snell's Law to find shear angles
    theta_s1 = arcsin(cs1 * sin(theta_p1) / cp1)
    theta_s2 = arcsin(cs2 * sin(theta_p1) / cp1)

    # Intermediate terms from Eqs. (6.162) and (6.163)
    d1 = (cs1 / cp1) * sin(2 * theta_s1) * sin(theta_p1) + cos(2 * theta_s1) * cos(theta_s1)
    d2 = (cs1 / cp1) * sin(2 * theta_p1) * sin(theta_s1) + cos(2 * theta_s1) * cos(theta_p1)

    l1 = (cs1 / cp2) * sin(2 * theta_s1) * sin(theta_p2) + (rho2 / rho1) * cos(2 * theta_s2) * cos(theta_s1)
    m1 = -(cs1 / cs2) * sin(2 * theta_s1) * cos(theta_s2) - (rho2 / rho1) * sin(2 * theta_s2) * cos(theta_s1)
    l2 = (cs1 / cp2) * cos(2 * theta_s1) * sin(theta_p2) - (rho2 / rho1) * cos(2 * theta_s2) * sin(theta_s1)
    m2 = (cs1 / cs2) * cos(2 * theta_s1) * cos(theta_s2) + (rho2 / rho1) * sin(2 * theta_s2) * sin(theta_s1)
    l3 = -(cp1 / cp2) * cos(2 * theta_s1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (rho1 * cp1 * cp2) * sin(
        2 * theta_p2) * sin(theta_p1)
    m3 = -(cp1 / cs2) * cos(2 * theta_s1) * sin(theta_s2) + (rho2 * cs2 ** 2) / (rho1 * cp1 * cs2) * cos(
        2 * theta_s2) * sin(theta_p1)
    l4 = -(cs1 ** 2) / (cp1 * cp2) * sin(2 * theta_p1) * cos(theta_p2) - (rho2 * cs2 ** 2) / (rho1 * cp1 * cp2) * sin(
        2 * theta_p2) * cos(theta_p1)
    m4 = (cs1 ** 2) / (cp1 * cs2) * sin(2 * theta_p1) * sin(theta_s2) - (rho2 * cs2 ** 2) / (rho1 * cp1 * cs2) * cos(
        2 * theta_s2) * cos(theta_p1)

    # Denominator from Eq. (6.166)
    delta = (l2 / d1 + l4 / d2) * (m1 / d1 - m3 / d2) - (l1 / d1 - l3 / d2) * (m2 / d1 - m4 / d2)


    # Potential amplitude ratios from Eq. (6.164)
    Ar_Ai = ((l2 / d1 - l4 / d2) * (m1 / d1 - m3 / d2) - (l1 / d1 - l3 / d2) * (m2 / d1 + m4 / d2)) / delta

    # Convert potential amplitude ratio to reflection and transmission coefficient (Eq. 4.28):
    Rpp = Ar_Ai * rho1 / rho1

    return Rpp

def solid2solid_t_coeff(theta_p1, theta_p2, cp1, cp2, cs1, cs2, rho1, rho2):
    """
    Calculates the transmission and reflection coefficients for a solid-solid interface (P-wave incidence).
    Returns Tpp (P-to-P transmission)
    """
    theta_p1 = abs(theta_p1)
    theta_p2 = abs(theta_p2)

    # Snell's Law to find shear angles
    theta_s1 = arcsin(cs1 * sin(theta_p1) / cp1)
    theta_s2 = arcsin(cs2 * sin(theta_p1) / cp1)

    # Intermediate terms from Eqs. (6.162) and (6.163)
    d1 = (cs1 / cp1) * sin(2 * theta_s1) * sin(theta_p1) + cos(2 * theta_s1) * cos(theta_s1)
    d2 = (cs1 / cp1) * sin(2 * theta_p1) * sin(theta_s1) + cos(2 * theta_s1) * cos(theta_p1)

    l1 = (cs1 / cp2) * sin(2 * theta_s1) * sin(theta_p2) + (rho2 / rho1) * cos(2 * theta_s2) * cos(theta_s1)
    m1 = -(cs1 / cs2) * sin(2 * theta_s1) * cos(theta_s2) - (rho2 / rho1) * sin(2 * theta_s2) * cos(theta_s1)
    l2 = (cs1 / cp2) * cos(2 * theta_s1) * sin(theta_p2) - (rho2 / rho1) * cos(2 * theta_s2) * sin(theta_s1)
    m2 = (cs1 / cs2) * cos(2 * theta_s1) * cos(theta_s2) + (rho2 / rho1) * sin(2 * theta_s2) * sin(theta_s1)
    l3 = -(cp1 / cp2) * cos(2 * theta_s1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (rho1 * cp1 * cp2) * sin(
        2 * theta_p2) * sin(theta_p1)
    m3 = -(cp1 / cs2) * cos(2 * theta_s1) * sin(theta_s2) + (rho2 * cs2 ** 2) / (rho1 * cp1 * cs2) * cos(
        2 * theta_s2) * sin(theta_p1)
    l4 = -(cs1 ** 2) / (cp1 * cp2) * sin(2 * theta_p1) * cos(theta_p2) - (rho2 * cs2 ** 2) / (rho1 * cp1 * cp2) * sin(
        2 * theta_p2) * cos(theta_p1)
    m4 = (cs1 ** 2) / (cp1 * cs2) * sin(2 * theta_p1) * sin(theta_s2) - (rho2 * cs2 ** 2) / (rho1 * cp1 * cs2) * cos(
        2 * theta_s2) * cos(theta_p1)

    # Denominator from Eq. (6.166)
    delta = (l2 / d1 + l4 / d2) * (m1 / d1 - m3 / d2) - (l1 / d1 - l3 / d2) * (m2 / d1 - m4 / d2)

    #
    delta1 = cs1/cp1 * np.sin(2 * theta_s1) * np.sin(theta_p1) + np.cos(2 * theta_s1) * np.cos(theta_s1)
    delta2 = cs1/cp1 * np.sin(2 * theta_p1) * np.sin(theta_s1) + np.cos(2 * theta_s1) * np.cos(theta_p1)

    # Potential amplitude ratios from Eq. (6.164)
    At_Ai = -2 * ((m2 / delta1) + (m4 / delta2)) / delta

    # Convert to transmission coefficient:
    Tpp = At_Ai * rho2 / rho1 # Eq. 4.28

    return Tpp # in theory Tps, Tsp and Tss is missing.

def fluid2solid_r_coeff(theta_p1, theta_p2, cp1, cp2, cs2, rho1, rho2):
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

    Rpp = (delta2 - delta1) / (delta2 + delta1)

    return Rpp

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
