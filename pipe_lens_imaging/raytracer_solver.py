from typing import Any

import numpy as np

from numpy import ndarray, pi
from numpy.linalg import norm
from abc import ABC, abstractmethod

from pipe_lens.raytracing_utils import uhp, roots_bhaskara, snell
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens_imaging.transducer import Transducer
from bisect import bisect
from scipy.optimize import minimize_scalar

from pipe_lens_imaging.ultrasound import *

__all__ = ["RayTracerSolver"]

FLOAT = np.float32


class RayTracerSolver(ABC):
    def __init__(self, acoustic_lens: AcousticLens, pipeline: Pipeline, transducer: Transducer, transmission_loss: bool= False, reflection_loss: bool = False, directivity: bool= False):
        self.transducer = transducer
        self.pipeline = pipeline
        self.acoustic_lens = acoustic_lens

        self.transmission_loss = transmission_loss
        self.reflection_loss = reflection_loss
        self.directivity = directivity

        self.c1 = self.c2 = self.c3 = None

    def _solve(self, xf, zf, maxiter: int = 6, solver: str='newton'):
        if isinstance(xf, (int, float)) and isinstance(zf, (int, float)):
            xf, zf = np.array([xf]), np.array([zf])

        # Match valid solvers:
        if solver == 'newton':
            solution = self._newton_batch(xf, zf, maxiter)
        elif solver == 'grid-search':
            solution = self._grid_search_batch(xf, zf, maxiter)
        elif solver == 'scipy-bounded':
            solution = self._scipy_bounded_batch(xf, zf, maxiter)
        else:
            raise ValueError(f"{solver} is an invalid solver.")

        return solution

    def set_speeds(self, c1, c2, c3):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def get_speeds(self):
        c1 = self.acoustic_lens.c1 if self.c1 is None else self.c1  # Wedge material
        c2 = self.acoustic_lens.c2 if self.c2 is None else self.c2  # Coupling medium
        c3 = self.pipeline.c if self.c3 is None else self.c3  # Pipeline material
        return c1, c2, c3

    def solve(self, xf, zf, maxiter: int = 6, solver='newton'):
        # Find focii TOF:
        solution = self._solve(xf, zf, maxiter, solver)

        tofs, amplitudes = self.get_tofs(solution)

        return tofs, amplitudes

    def _newton_batch(self, xf: ndarray, yf: ndarray, iter: int, verbose=False) -> list:
        '''Calls the function newton() one time for each transducer element.
      The set of angles found for a given element are used as initial guess for
      the next one. Starts from the center of the transducer.'''

        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem
        results[N_elem // 2] = self._newton(xc[N_elem // 2], yc[N_elem // 2], xf, yf)
        results[N_elem // 2 - 1] = self._newton(xc[N_elem // 2 - 1], yc[N_elem // 2 - 1], xf, yf)
        for i in range(1, N_elem // 2):
            i_m = N_elem // 2 - i - 1
            i_p = N_elem // 2 + i

            alpha_init = np.arctan(results[i_p - 1]['xlens'] / results[i_p - 1]['zlens'])

            # Compute the optimal path for a given transducer (xc,yc) focus (xf, yf) pair:
            results[i_p] = self._newton(xc[i_p], yc[i_p], xf, yf, alpha_init=alpha_init, iter=iter)
            bad_indices = results[i_p]['dist'] > 1e-8

            if verbose:
                if np.count_nonzero(bad_indices) > 0:
                    print('Bad indices found at ' + str(i_p) + ': ' + str(np.count_nonzero(bad_indices)))

            alpha_init = np.arctan(results[i_m + 1]['xlens'] / results[i_m + 1]['zlens'])
            results[i_m] = self._newton(xc[i_m], yc[i_m], xf, yf, alpha_init=alpha_init, iter=iter)
            bad_indices = results[i_m]['dist'] > 1e-8

            if verbose:
                if np.count_nonzero(bad_indices) > 0:
                    print('Bad indices found at ' + str(i_m) + ': ' + str(np.count_nonzero(bad_indices)))
        return results

    def _grid_search_batch(self, xf: ndarray, yf: ndarray, alpha_step=1e-3, dist_tol=.1e3, delta_alpha=30e-3, verbose=False) -> list:
        '''Calls the function newton() one time for each transducer element.
      The set of angles found for a given element are used as initial guess for
      the next one. Starts from the center of the transducer.'''

        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem

        for i in range(N_elem):
            results[i] = self._grid_search(xc[i], yc[i], xf, yf, alpha_step, dist_tol, delta_alpha)
            print('i = ', i)
        return results

    def _grid_search(self, xc: float, yc: float, xf: ndarray, yf: ndarray, alpha_step: float, tol: float, delta_alpha: float) -> dict:
        alpha_grid_coarse = np.arange(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max + alpha_step, alpha_step)
        alpha_grid_fine = np.arange(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max + alpha_step/10, alpha_step/10)
        alphaa = np.zeros_like(xf)

        for i, (x_target, y_target) in enumerate(zip(xf, yf)):
            # Compute distances for the coarse grid
            dic_coarse_distances = self._dist_kernel(
                xc, yc,
                x_target * np.ones_like(alpha_grid_coarse),
                y_target * np.ones_like(alpha_grid_coarse),
                alpha_grid_coarse
            )
            # Find alpha minimizing distance on coarse grid
            alpha_coarse_min = alpha_grid_coarse[np.nanargmin(dic_coarse_distances['dist'])]

            # Define fine grid search bounds around coarse minimum
            fine_start_idx = bisect(alpha_grid_fine, alpha_coarse_min - delta_alpha)
            fine_end_idx = bisect(alpha_grid_fine, alpha_coarse_min + delta_alpha)
            alpha_fine_subset = alpha_grid_fine[fine_start_idx:fine_end_idx]

            # Compute distances on the fine grid subset
            fine_distances = self._dist_kernel(
                xc, yc,
                x_target * np.ones_like(alpha_fine_subset),
                y_target * np.ones_like(alpha_fine_subset),
                alpha_fine_subset
            )

            # Find alpha minimizing distance on fine grid subset
            alphaa[i] = alpha_fine_subset[np.nanargmin(fine_distances['dist'])]

        # Final evaluation with all optimal alphas
        final_results = self._dist_kernel(xc, yc, xf, yf, alphaa)
        final_results['firing_angle'] = alphaa
        # Set distances above tolerance to NaN
        final_results['dist'][final_results['dist'] >= tol] = np.nan

        return final_results

    def _newton(self, xc: float, yc: float, xf: ndarray, yf: ndarray, alpha_init=None, iter: int = 10) -> dict:
        '''Uses the Newton-Raphson method to compute the direction in which
      the transducer at (xc, yc) should fire in order to hit the "pixel"
      at (xf, yf). A dictionary is returned with the following information:

      (xcurve, ycurve): position where the ray hits the lens
      (xcirc, ycirc): position where the ray hits the circle (cylinder)
      (xin, yin): point on the ray closest to (xf, yf)
      dist: squared distance (xin, yin) to (xf, yf) (should be close to zero)
      maxdist: maximum squared distance (assuming an array of pixels was passed)
      mindist: minimum squared distance (assuming an array of pixels was passed)'''

        if alpha_init is None:
            alphaa = np.arctan(xf / yf) # might work
        else:
            alphaa = alpha_init
        maxdist = list()
        mindist = list()
        cost_fun_hist = list()
        for i in range(iter):
            dic, d1, d2 = self._dist_and_derivatives(xc, yc, xf, yf, alphaa, eps=1e-4)
            step = d1 / d2
            alphaa -= step
            alphaa = np.clip(alphaa, -self.acoustic_lens.alpha_max * .9, self.acoustic_lens.alpha_max * .9)
            maxdist.append(dic['dist'].max())
            mindist.append(dic['dist'].min())
            cost_fun_hist.append(dic['dist'][-1])
        dic['maxdist'] = maxdist
        dic['mindist'] = mindist
        dic['firing_angle'] = alphaa
        return dic

    def _scipy_bounded_batch(self, xf: ndarray, yf: ndarray, iter: int=10, verbose=False):
        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem
        results[N_elem // 2] = self._scipy_bounded(xc[N_elem // 2], yc[N_elem // 2], xf, yf)
        results[N_elem // 2 - 1] = self._scipy_bounded(xc[N_elem // 2 - 1], yc[N_elem // 2 - 1], xf, yf)
        for i in range(1, N_elem // 2):
            i_m = N_elem // 2 - i - 1
            i_p = N_elem // 2 + i

            alpha_init = np.arctan(results[i_p - 1]['xlens'] / results[i_p - 1]['zlens'])

            # Compute the optimal path for a given transducer (xc,yc) focus (xf, yf) pair:
            results[i_p] = self._scipy_bounded(xc[i_p], yc[i_p], xf, yf, alpha_init=alpha_init, iter=iter)

            alpha_init = np.arctan(results[i_m + 1]['xlens'] / results[i_m + 1]['zlens'])
            results[i_m] = self._scipy_bounded(xc[i_m], yc[i_m], xf, yf, alpha_init=alpha_init, iter=iter)

        return results

    def _scipy_bounded(self, xc: float, yc: float, xf: ndarray, yf: ndarray, alpha_init=None, iter: int = 10,
                       tol: float = .5e-3) -> dict:
        N = len(xf)
        alphaa = np.zeros(N)

        for jj in range(N):
            cost_fun_i = lambda x, idx=jj: self._dist_kernel(
                xc, yc, xf[idx:idx + 1], yf[idx:idx + 1], alpha=np.array([x])
            )['dist'][0]

            res = minimize_scalar(
                cost_fun_i,
                bounds=(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max),
                method='bounded',
                tol=tol,
                options={"maxiter": iter}
            )

            alphaa[jj] = res.x if res.fun <= tol else np.nan

        # Evaluate final distances with best alphas
        dic = self._dist_kernel(xc, yc, xf, yf, alpha=alphaa)

        # Compute max and min distances
        dist_array = dic['dist']
        # dic['maxdist'] = np.nanmax(dist_array)
        # dic['mindist'] = np.nanmin(dist_array)
        dic['firing_angle'] = alphaa

        return dic

    def _dist_and_derivatives(self, xc: float, yc: float, xf: ndarray, yf: ndarray, acurve: ndarray, eps: float = 1e-5):
        '''Computes the squared distance using distalpha as well as the first and
      second derivatives of the squared distance with relation to alpha.'''
        dm = self._dist_kernel(xc, yc, xf, yf, acurve - eps)['dist']
        dic = self._dist_kernel(xc, yc, xf, yf, acurve)
        d0 = dic['dist']
        dp = self._dist_kernel(xc, yc, xf, yf, acurve + eps)['dist']
        der1 = (dp - dm) * .5 / eps
        der2 = (dm - 2 * d0 + dp) / eps ** 2
        return dic, der1, der2

    ##### Case-specific:
    @abstractmethod
    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: ndarray):
        pass

    @abstractmethod
    def get_tofs(self, solutions):
        pass