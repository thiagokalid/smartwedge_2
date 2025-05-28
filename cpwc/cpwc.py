import numpy as np
import numba

from .utils import compute_tof


def cpwc_contact_kernel(pwi_data, xroi, zroi, xt, zt, thetas, c_specimen, fs, gate_start=0):
    Nt, Nangs, Nel = pwi_data.shape

    tgs = gate_start

    # Dimensões ROI
    Nrows = zroi.shape[0]
    Ncols = xroi.shape[0]

    # Imagem
    flatten_img = np.zeros((Ncols * Nrows, 1), dtype=pwi_data.dtype)

    t_tfm = np.float64(compute_tof(xroi, zroi, xt, zt, c_specimen))

    baseline_shift = np.int64(tgs * fs)

    for k, thetak in enumerate(thetas):
        data = np.vstack((pwi_data[:, k, :], np.zeros((1, Nel)))).astype(pwi_data.dtype)

        # j = cpwc_roi_dist(xroi, zroi, xt, thetak, c_specimen, fs, tgs, t_tfm)

        t_pwi = tof_cwpc_contact(xroi, zroi, xt, thetak, c_specimen)
        j = np.rint((t_pwi + t_tfm) * fs) - baseline_shift

        # Checa se os delays são factíveis, isto é, estão no intervalo de número de amostras:
        j = np.int64(j.reshape(Nrows * Ncols, Nel, order='F'))
        j[(j < 0) | (j >= Nt)] = -1

        # Soma as amostras de Ascan coerentemente
        aux = np.zeros(Nrows * Ncols, dtype=pwi_data.dtype)
        flatten_img[:, 0] += cpwc_sum(data, aux, j)

    img = flatten_img.reshape((Nrows, Ncols), order='F')

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


