import numpy as np
import numba
from mpmath.functions.bessel import c_memo

from .newton_circ import *



# @numba.njit(parallel=True)
def cpwc_circle_kernel(pwi_data, xroi, zroi, xt, zt, xcenter, zcenter, radius, thetas, c_coupling, c_specimen, fs, gate_start=0):
    Nt, Nangs, Nel = pwi_data.shape
    Nrows = zroi.shape[0]
    Ncols = xroi.shape[0]

    baseline_shift = np.int64(gate_start * fs)

    # Thread-local buffers to avoid race conditions
    # thread_imgs = np.zeros((numba.get_num_threads(), Nrows, Ncols), dtype=np.float32)
    img = np.zeros((Nrows, Ncols), dtype=np.float32)
    

    for idx in numba.prange(Nangs * Ncols * Nrows * Nel):
        k = idx // (Ncols * Nrows * Nel)
        rem = idx % (Ncols * Nrows * Nel)

        i = rem // (Nrows * Nel)
        rem = rem % (Nrows * Nel)

        j = rem // Nel
        n = rem % Nel

        thetak = thetas[k]
        xr_i = xroi[i]
        zr_j = zroi[j]
        xt_n = xt[n]

        di = (zr_j * np.cos(thetak) + xr_i * np.sin(thetak))
        ti = di / c_specimen

        xe = newton_circ(
            xt_n, 0,
            xr_i, zr_j,
            c_coupling, c_specimen,
            xcenter, zcenter, radius,
            x0 = 0,
            maxiter = 10,
            tol = 1e-6
        )
        ze = f_circ(xe, xcenter, zcenter, radius)
        tv = np.sqrt((xe - xt_n)**2 + (ze - 0)**2) / c_coupling + np.sqrt((xe - xr_i)**2 + (ze - zr_j)**2) / c_specimen
        shift = int(np.rint((ti + tv) * fs) - baseline_shift)

        if 0 <= shift < Nt:
            # thread_id = numba.np.ufunc.parallel._get_thread_id()
            img[j, i] += pwi_data[shift, k, n]

    # Reduce thread-local buffers into final image
    # for t in range(numba.get_num_threads()):
    #     img += thread_imgs[t]

    return img


@numba.njit(parallel=True)
def cpwc_roi_dist(xroi, zroi, xt, theta, c_specimen, fs, tgs, t_tfm):
    Nz = zroi.shape[0]
    Nx = xroi.shape[0]
    Nel = xt.shape[0]

    baseline_shift = np.int64(tgs * fs)

    j = np.zeros((Nx * Nz, Nel), dtype=np.int64)
    for i in numba.prange(Nx):
        for jj in range(Nz):
            i_i = i * Nz + jj
            di = (zroi[jj] * np.cos(theta) + xroi[i] * np.sin(theta))
            # dv = np.sqrt((xroi[i] - xt) ** 2 + (zroi[jj] - 0)**2)

            # Atraso na Ida e na Volta (lei focal de emissão e recepção):
            ti = di / c_specimen
            # tv = dv / c_specimen
            tv = t_tfm[i_i, :]

            shift = np.rint((ti + tv) * fs)
            j[i_i, :] = shift - baseline_shift

    return j

def tof_cwpc_contact(xroi, zroi, xt, theta, c_specimen):
    Nz = zroi.shape[0]
    Nx = xroi.shape[0]
    Nel = xt.shape[0]

    xxroi, zzroi = np.meshgrid(xroi, zroi)
    di = xxroi * np.sin(theta) + zzroi * np.cos(theta)
    ti = np.reshape(di / c_specimen, newshape=(Nx * Nz, 1), order='F')
    t_pwi = np.tile(ti, reps=(1, Nel))

    return np.float64(t_pwi)

@numba.njit(parallel=True)
def cpwc_sum(data, img, j):
    i = np.arange(j.shape[1])
    for jj in numba.prange(j.shape[0]):
        idx = j[jj, :]
        for ii in range(i.shape[0]):
            img[jj] += data[idx[ii], ii]

    return img


