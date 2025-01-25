from types import FunctionType
from numpy import ndarray

import numpy as np

from pipe_lens.transducer import Transducer

__all__ = ["RayTracer"]

class RayTracer:
    def __init__(self, f1: FunctionType, df1_dx: FunctionType, f2:FunctionType, df2_dx: FunctionType, transducer: Transducer, tol: float=1e-1):
        self.tol = tol
        self.transducer = transducer

        # Functions which defines the first refraction profile:
        self.f1 = f1
        self.df1_dx = df1_dx

        # Second profile:
        self.f2 = f2
        self.df2_dx = df2_dx

    def solve(self, foci: ndarray) -> [ndarray, ndarray]:
        # foci is a 'Nfocus x 2' ndarray with coordinates to each focusing point in cartesian coordinates
        Nfocus = foci.shape[0]
        Nel = self.transducer.n_elem

        # Each entry represent the ray which obeys Snell's law between i-th emitter to j-th focus
        tof = np.zeros(shape=(Nel, Nfocus))
        steering_ang = np.zeros(shape=(Nel, Nel))

        for n in range(Nel * Nfocus):
            i = n // Nfocus  # i-th emitter (row index)
            j = n % Nfocus  # j-th focus (column index)

            emitter = self.transducer.get_coords(i)
            focus = foci[j]
            tof[i, j], steering_ang[i, j] = self.__solve(emitter, focus)

        return tof, steering_ang

    def __solve(self, emitter: ndarray, focus: ndarray, maxiter: int=30) -> [float, float]:
        raise NotImplementedError