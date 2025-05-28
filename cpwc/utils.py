import numpy as np

from scipy.spatial.distance import cdist

def compute_tof(xroi, zroi, xt, zt, c_specimen, c_coupling = None):
    Nx, Nz = xroi.shape[0], zroi.shape[0]

    if c_coupling is not None:


    else:
        xxroi, zzroi = np.meshgrid(xroi, zroi)
        xxroi, zzroi = xxroi.reshape(Nx * Nz, order='F'), zzroi.reshape(Nx * Nz, order='F')
        roi = np.asarray([xxroi, zzroi]).T
        transducer = np.asarray([xt, zt]).T
        dist = cdist(transducer, roi).T
        tof = dist / c_specimen
    return tof

def newton_circle_reception(xf: float, zf: float, xa: float, za: float, xc, zc, r, c1, c2, maxiter: int = 10, tol: float = 1e-6):
    # (xf, zf) focus location
    # (xa, za) element location
    # (xc, zc) center of circle with 'r' radius.
    # All values in SI unit, i.e. meter, second, etc.

    i = 0
    x0 = 0

    

    while (i < maxiter) and abs(f(x0)) > tol:
