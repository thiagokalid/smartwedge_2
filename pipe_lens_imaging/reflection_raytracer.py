import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from .raytracer_solver import RayTracerSolver
from pipe_lens.raytracing_utils import refraction, reflection, roots_bhaskara
from pipe_lens_imaging.ultrasound import liquid2solid_t_coeff, far_field_directivity_solid, solid2liquid_t_coeff, \
    liquid2solid_r_coeff
import matplotlib.pyplot as plt
from .raytracer_utils import plot_setup, plot_line, plot_normal
from .geometric_utils import findIntesectionBetweenCurveAndLine, findIntersectionBetweenAcousticLensAndRay

__all__ = ['ReflectionRayTracer']

FLOAT = np.float32


class ReflectionRayTracer(RayTracerSolver):
    def get_tofs(self, solution):
        c1, c2, c3 = self.get_speeds()

        n_elem = self.transducer.num_elem
        amplitudes = {
            "transmission_loss": np.ones((n_elem, n_elem, 1), dtype=FLOAT),
            "reflection_loss": np.ones((n_elem, n_elem, 1), dtype=FLOAT),
            "directivity": np.ones((n_elem, n_elem, 1), dtype=FLOAT)
        }

        tofs = np.zeros(shape=(n_elem, n_elem, 1), dtype=FLOAT)

        for tx in range(self.transducer.num_elem):
            # Computing Time-Of-Flights:
            # Distance from emitter to lens (emission):
            d_12 = np.sqrt((solution[tx]["xlens"] - self.transducer.xt[tx])**2 +
                         (solution[tx]["zlens"] - self.transducer.zt[tx])**2)

            # Distance from lens to pipe outer surface (refraction):
            d_23 = np.sqrt((solution[tx]["xlens"] - solution[tx]["xpipe"])**2 +
                         (solution[tx]["zlens"] - solution[tx]["zpipe"])**2)

            # Distance from pipe outer surface to lens (reflection):
            d_32 = np.sqrt((solution[tx]["xpipe"] - solution[tx]["xlens2"])**2 +
                         (solution[tx]["zpipe"] - solution[tx]["zlens2"])**2)

            # Distance from lens to receiver (refraction):
            d_21 = np.sqrt((solution[tx]["xlens2"] - solution[tx]["target_x"])**2 +
                         (solution[tx]["zlens2"] - solution[tx]["target_z"])**2)

            tof = d_12 / c1 + d_23 / c2 + d_32 / c2 + d_21 / c1
            dists = solution[tx]["dist"]
            tof[np.abs(dists) >= self.transducer.element_width] = -1
            tofs[tx, :, 0] = tof

            # Extract the amplitudes:
            for rx in range(self.transducer.num_elem):
                if self.transmission_loss:
                    Tpp_12, _ = solid2liquid_t_coeff(
                            solution[tx]['interface_12'][0][rx], solution[tx]['interface_12'][1][rx],
                            c1, c2, c1 / 2,
                            self.acoustic_lens.rho1, self.acoustic_lens.rho2
                        )


                    Tpp_r = liquid2solid_r_coeff(
                        solution[tx]['interface_23'][0][rx], solution[tx]['interface_23'][1][rx],
                        c2, c3, c3/2,
                        self.acoustic_lens.rho2, self.pipeline.rho
                    )

                    Tpp_21, _ = liquid2solid_t_coeff(
                        solution[tx]['interface_21'][0][rx], solution[tx]['interface_21'][1][rx],
                        c2, c1, c1/2,
                        self.acoustic_lens.rho1, self.acoustic_lens.rho2
                    )

                    amplitudes['transmission_loss'][tx, rx, 0] = Tpp_12 * Tpp_r * Tpp_21

                elif self.directivity:
                    deltax = self.transducer.xt[rx] - solution[tx]["zlens2"][rx]
                    deltaz = self.transducer.zt[rx] - solution[tx]["xlens2"][rx]

                    theta = np.arctan2(deltaz, deltax)
                    k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                    amplitudes["directivity"][tx, rx] *= far_field_directivity_solid(
                        theta, c1, c1 / 2, k, self.transducer.element_width
                    )


        return tofs, amplitudes

    def _distalpha(self, xa: float, za: float, xf: ndarray, zf: ndarray, alpha: np.ndarray, plot=False):
        c1, c2, c3 = self.get_speeds()

        x_p, z_p = self.acoustic_lens.xy_from_alpha(alpha)

        # Equation (B.2) in Appendix B.
        phi_ap = np.arctan2(za - z_p, xa - x_p)

        # Refraction (c1 -> c2)
        d_zh, d_xh = self.acoustic_lens.dydx_from_alpha(alpha, mode='partial')
        phi_pq, phi_h, inc_12, ref_12 = refraction(phi_ap, (d_zh, d_xh), c1, c2)

        # Line equation
        a_pq = np.tan(phi_pq)
        b_pq = z_p - a_pq * x_p

        # Equation (B.11b) in Appendix B.
        # Equation (B.11b) in Appendix B.
        A = np.square(a_pq) + 1
        B = 2 * a_pq * b_pq - 2 * (self.pipeline.xcenter + a_pq * self.pipeline.zcenter)
        C = np.square(
            b_pq) + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_pq - self.pipeline.outer_radius ** 2

        x_q1, x_q2 = roots_bhaskara(A, B, C)
        z_q1 = a_pq * x_q1 + b_pq
        z_q2 = a_pq * x_q2 + b_pq
        mask_upper = z_q1 > z_q2
        x_q = np.where(mask_upper, x_q1, x_q2)
        z_q = np.where(mask_upper, z_q1, z_q2)

        # Reflection in the pipe
        slope_zc_x = self.pipeline.dydx(x_q, mode='full')

        # If using the function below, there is no need for uhp()
        phi_l, phi_c, inc_23, _ = reflection(phi_pq, slope_zc_x)
        _, _, inc_23, ref_23 = refraction(phi_pq, slope_zc_x, c1, c2)

        # Line equation
        a_l = np.tan(phi_l)
        b_l = z_q - a_l * x_q

        # alpha_intersection = self.acoustic_lens.findIntersectionWithLine(a_l, b_l, alpha_0=alpha)
        # intersection_x, intersection_z = self.acoustic_lens.xy_from_alpha(alpha_intersection)

        # intersection_x, intersection_z = findIntesectionBetweenCurveAndLine(a_l, b_l, x_q, self.acoustic_lens.xlens, self.acoustic_lens.zlens)
        alpha_mins = findIntersectionBetweenAcousticLensAndRay(a_l, b_l, self.acoustic_lens)
        intersection_x, intersection_z = self.acoustic_lens.xy_from_alpha(alpha_mins)

        # Refraction (c2 -> c1)
        alpha_intersection = np.arctan2(intersection_x, intersection_z)
        d_z_intersection, d_x_intersection = self.acoustic_lens.dydx_from_alpha(alpha_intersection, mode='partial')
        phi_last, phi_intersection_incidence, inc_21, ref_21 = refraction(phi_l, (d_z_intersection, d_x_intersection), c2, c1)

        # Line equation
        a_intersection = np.tan(phi_last)
        b_intersection = intersection_z - a_intersection * intersection_x

        x_in = (zf - b_intersection) / a_intersection
        z_in = self.acoustic_lens.d * np.ones_like(x_in)

        dist = np.sqrt((x_in - xf)**2 + (z_in - zf)**2)

        if plot:
            plot_setup(self.acoustic_lens, self.pipeline, self.transducer, show=False, legend=False)
            plt.title(f"Element at {xa} m shooting")
            plt.xlim([-0.1, 0.1])
            for idx, ray in enumerate(range(0, len(alpha), 10)):
                if idx == 0:
                    plt.plot([xa, x_p[ray]], [za, z_p[ray]], "C0", label="Incident ray")
                    plt.plot([x_p[ray], x_q[ray]], [z_p[ray], z_q[ray]], "C1", label="Refracted ray (c1->c2)")
                    plt.plot([x_q[ray], intersection_x[ray]], [z_q[ray], intersection_z[ray]], "C2",
                             label="Reflected ray")
                    plt.plot([intersection_x[ray], x_in[ray]], [intersection_z[ray], z_in[ray]], "C3",
                             label="Refracted ray (c2->c1)")
                else:
                    plt.plot([xa, x_p[ray]], [za, z_p[ray]], "C0")
                    plt.plot([x_p[ray], x_q[ray]], [z_p[ray], z_q[ray]], "C1")
                    plt.plot([x_q[ray], intersection_x[ray]], [z_q[ray], intersection_z[ray]], "C2")
                    plt.plot([intersection_x[ray], x_in[ray]], [intersection_z[ray], z_in[ray]], "C3")

                # plt.plot(x_f, z_f, 'o', markersize=0.1)
                # plot_line(phi_last[ray], intersection_x[ray], intersection_z[ray], scale=0.2, x_pos=False, z_pos=False)
                # plot_normal(phi_h[ray], x_p[ray], z_p[ray])
                # plot_normal(phi_c[ray], x_q[ray], z_q[ray])
                # if ray < len(intersection_x):
                #     plot_normal(phi_intersection_incidence[ray], intersection_x[ray], intersection_z[ray])
            plt.legend()
            plt.show()

        return {
            "dist": dist,
            "xlens": x_p,
            "zlens": z_p,
            "xpipe": x_q,
            "zpipe": z_q,
            "xlens2": intersection_x,
            "zlens2": intersection_z,
            "target_x": x_in,
            "target_z": z_in,
            "interface_12": [inc_12, ref_12],
            "interface_23": [inc_23, ref_23],
            "interface_21": [inc_21, ref_21]
        }