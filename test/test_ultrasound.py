# test_calculator.py
from numpy import sin, arcsin
import numpy as np
import unittest
from pipe_lens_imaging.ultrasound import *

CP_WATER, RHO_WATER = 1483, 1000 # m/s and kg/m³
CP_STEEL, CS_STEEL, RHO_STEEL = 5900, 2950, 7750 # m/s and kg/m³
CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM = 6300, 3150, 2700 # m/s and kg/m³

# Acoustic impedance:
Zwater = CP_WATER * RHO_WATER
Zsteel = CP_STEEL * RHO_STEEL
Zaluminium = CP_ALUMINIUM * RHO_ALUMINIUM

class TestCoefficients(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.incidence_angle = 0

    def test_solid2solid_A(self):
        # Transmission from medium 1 to medium 2
        # First medium:
        cp1, cs1, rho1 = CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM
        Z1 = cp1 * rho1

        # Second medium:
        cp2, cs2, rho2 = CP_STEEL, CS_STEEL, RHO_STEEL
        Z2 = cp2 * rho2

        # Angles:
        theta_p1 = self.incidence_angle
        theta_p2 = arcsin(cp1 / cp2 * sin(theta_p1))

        # Acoustic impedance approximation (valid for incidence angle = 0):
        Tpp_Z = np.float32(2 * Z2 / (Z1 + Z2))
        Rpp_Z = np.float32((Z2 - Z1) / (Z1 + Z2))

        # General formula:
        Tpp, Rpp = solid2solid_tr_coeff(theta_p1, theta_p2, cp1, cp2, cs1, cs2, rho1, rho2)

        np.testing.assert_almost_equal(Tpp, Tpp_Z)
        np.testing.assert_almost_equal(Rpp, Rpp_Z)

    def test_solid2solid_B(self):
        # Transmission from medium 1 to medium 2
        # First medium:
        cp1, cs1, rho1 = CP_STEEL, CS_STEEL, RHO_STEEL
        Z1 = cp1 * rho1

        # Second medium:
        cp2, cs2, rho2 = CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM
        Z2 = cp2 * rho2

        # Angles:
        theta_p1 = self.incidence_angle
        theta_p2 = arcsin(cp1 / cp2 * sin(theta_p1))

        # Acoustic impedance approximation (valid for incidence angle = 0):
        Tpp_Z = np.float32(2 * Z2 / (Z1 + Z2))
        Rpp_Z = np.float32((Z2 - Z1) / (Z1 + Z2))

        # General formula:
        Tpp, Rpp = solid2solid_tr_coeff(theta_p1, theta_p2, cp1, cp2, cs1, cs2, rho1, rho2)

        np.testing.assert_almost_equal(Tpp, Tpp_Z)
        np.testing.assert_almost_equal(Rpp, Rpp_Z)

    def test_fluid2solid(self):
        # Transmission from medium 1 to medium 2

        # First medium:
        cp1, rho1 = CP_WATER, RHO_WATER
        Z1 = cp1 * rho1

        # Second medium:
        cp2, cs2, rho2 = CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM
        Z2 = cp2 * rho2

        # Incidence and refraction angles:
        theta_p1 = self.incidence_angle
        theta_p2 = arcsin(cp1 / cp2 * sin(theta_p1))

        # Acoustic impedance approximation (valid for incidence angle = 0):
        Tpp_Z = np.float32(2 * Z2 / (Z1 + Z2))
        Rpp_Z = np.float32((Z2 - Z1) / (Z1 + Z2))

        # General formula:
        Tpp, Tps = fluid2solid_t_coeff(theta_p1, theta_p2, cp1, cp2, cs2, rho1, rho2)
        Rpp = fluid2solid_r_coeff(theta_p1, theta_p2, cp1, cp2, cs2, rho1, rho2)

        np.testing.assert_almost_equal(Tpp, Tpp_Z)
        np.testing.assert_almost_equal(Tps, 0) # no mode conversion for normal incidence
        np.testing.assert_almost_equal(Rpp, Rpp_Z)

    def test_solid2fluid(self):
        # Transmission from medium 1 to medium 2

        # Second medium:
        cp2, rho2 = CP_WATER, RHO_WATER
        Z2 = cp2 * rho2

        # First medium:
        cp1, cs1, rho1 = CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM
        Z1 = cp1 * rho1

        # Incidence and refraction angles:
        theta_p1 = self.incidence_angle
        theta_p2 = arcsin(cp1 / cp2 * sin(theta_p1))

        # Acoustic impedance approximation (valid for incidence angle = 0):
        Tpp_Z = np.float32(2 * Z2 / (Z1 + Z2))
        Rpp_Z = np.float32((Z2 - Z1) / (Z1 + Z2))

        # General formula:
        Tpp = solid2fluid_t_coeff(theta_p1, theta_p2, cp1, cp2, cs1, rho1, rho2)
        # Rpp = solid2fluid_r_coeff(theta_p1, theta_p2, cp1, cp2, cs1, rho1, rho2)

        np.testing.assert_almost_equal(Tpp, Tpp_Z)
        # np.testing.assert_almost_equal(Rpp, Rpp_Z)

if __name__ == '__main__':
    unittest.main()