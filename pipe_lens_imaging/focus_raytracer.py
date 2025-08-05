import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from pipe_lens_imaging.raytracer_utils import roots_bhaskara, snell, uhp
from pipe_lens_imaging.ultrasound import far_field_directivity_solid, fluid2solid_t_coeff, solid2fluid_t_coeff
from pipe_lens_imaging.raytracer_solver import RayTracerSolver

__all__ = ['FocusRayTracer']

FLOAT = np.float32

class FocusRayTracer(RayTracerSolver):
    def get_tofs(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = len(solution[0]['xlens'])

        c1, c2, c3 = self.get_speeds()

        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([solution[0]['xf'], solution[0]['zf']]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        coords_outer = np.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "transmission_loss": np.ones((n_elem, n_elem, n_focii), dtype=FLOAT),
            "directivity": np.ones((n_elem, n_elem, n_focii), dtype=FLOAT)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['xlens'][i], solution[j]['zlens'][i]
            coords_outer[j, 0, i], coords_outer[j, 1, i] = solution[j]['xpipe'][i], solution[j]['zpipe'][i]

            if self.transmission_loss:
                Tpp_12, _ = solid2fluid_t_coeff(
                    solution[j]['interface_12'][0][i], solution[j]['interface_12'][1][i],
                    c1, c2, c1/2,
                    self.acoustic_lens.rho1, self.acoustic_lens.rho2
                )
                Tpp_23, _ = fluid2solid_t_coeff(
                    solution[j]['interface_23'][1][i], solution[j]['interface_23'][0][i],
                    c2, c3, c3/2,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )
                amplitudes["transmission_loss"][j, :, i] *= Tpp_12 * Tpp_23

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        coord_elements_mat = np.tile(coord_elements[:, :, np.newaxis], (1, 1, n_focii))
        coord_reflectors_mat = np.tile(coords_reflectors[:, :, np.newaxis], (1, 1, n_elem))

        d1 = norm(coords_lens - coord_elements_mat, axis=1)
        d2 = norm(coords_lens - coords_outer, axis=1)
        d3 = norm(coords_outer - coord_reflectors_mat.T, axis=1)

        tofs = d1 / c1 + d2 / c2 + d3 / c3
        return tofs, amplitudes

    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        xlens, ylens = self.acoustic_lens.xy_from_alpha(acurve)
        gamma1 = np.arctan2((ylens - zc), (xlens - xc))
        gamma1 = gamma1 + (gamma1 < 0) * np.pi

        gamma2, inc12, ref12 = snell(c1, c2, gamma1, self.acoustic_lens.dydx_from_alpha(acurve))
        a_line = np.tan(uhp(gamma2))
        b_line = ylens - a_line * xlens

        a = a_line**2 + 1
        b = 2 * a_line * b_line - 2 * (self.pipeline.xcenter + a_line * self.pipeline.zcenter)
        c = b_line ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_line - self.pipeline.outer_radius ** 2

        xcirc1, xcirc2 = roots_bhaskara(a, b, c)
        ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
        upper = ycirc1 > ycirc2
        xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
        ycirc = ycirc1 * upper + ycirc2 * (1 - upper)

        gamma3, inc23, ref23 = snell(c2, c3, gamma2, self.pipeline.dydx(xcirc))
        a3 = np.tan(gamma3)
        b3 = ycirc - a3 * xcirc

        xbottom = -b3 / a3
        a4 = -1 / a3
        b4 = yf - a4 * xf

        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf)**2 + (yin - yf)**2

        return {
            'xlens': xlens, 'zlens': ylens,
            'xpipe': xcirc, 'zpipe': ycirc,
            'dist': dist, 'xf': xf, 'zf': yf,
            'interface_12': [inc12, ref12],
            'interface_23': [inc23, ref23]
        }
