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

