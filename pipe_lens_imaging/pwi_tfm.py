import numpy as np
import numba

from .pwi_tfm_utils import *
import time


# @numba.njit(parallel=True)
def cpwc_circle_kernel(pwi_data, xroi, zroi, xt, zt, xcenter, zcenter, radius, thetas, c_coupling, c_specimen, fs, gate_start=0):
    Nrows = zroi.shape[0]
    Ncols = xroi.shape[0]

    baseline_shift = np.int64(gate_start * fs)

    pwi_tof = compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
    t_ref = np.max(pwi_tof)
    delay_law = t_ref - pwi_tof

    t0 = time.time()
    t_tfm = compute_t_tfm(xroi, zroi, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
    print(f"t_tfm. Time-elapsed= {time.time() - t0:.2f}")

    t0 = time.time()
    t_pwi = compute_t_pwi(
        thetas, xroi, zroi, 
        t_ref, xcenter, zcenter,
        c_specimen)
    print(f"t_pwi. Time-elapsed= {time.time() - t0:.2f}")

    t0 = time.time()
    img = cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift)
    print(f"coherent-sum. Time-elapsed= {time.time() - t0:.2f}")

    return img, delay_law

@numba.njit(parallel=True)
def cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift):
    Nt, Nangs, Nel = pwi_data.shape
    Nrows, Ncols, _ = t_tfm.shape

    img = np.zeros((Nrows, Ncols), dtype=np.float32)

    # Thread-local buffers to avoid race conditions
    thread_imgs = np.zeros((numba.get_num_threads(), Nrows, Ncols), dtype=np.float32)

    for idx in numba.prange(Nangs * Ncols * Nrows * Nel):
        k = idx // (Ncols * Nrows * Nel)
        rem = idx % (Ncols * Nrows * Nel)

        i = rem // (Nrows * Nel)
        rem = rem % (Nrows * Nel)

        j = rem // Nel
        n = rem % Nel

        # k-th angle:
        ti = t_pwi[k, j, i]

        tv = t_tfm[j, i, n]
        shift = int(np.rint((ti + tv) * fs) - baseline_shift)

        if 0 <= shift < Nt:
            thread_id = numba.np.ufunc.parallel._get_thread_id()
            thread_imgs[thread_id, j, i] += pwi_data[shift, k, n]

    # Reduce thread-local buffers into final image
    for t in range(numba.get_num_threads()):
        img += thread_imgs[t]

    return img

@numba.njit(parallel=True)
def compute_t_tfm(xroi, zroi, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen):
    Nx, Nz, Nel = xroi.shape[0], zroi.shape[0], xt.shape[0]

    t_tfm = np.zeros((Nz, Nx, Nel), dtype=np.float32)

    for idx in numba.prange(Nx * Nz * Nel):
        n = idx % Nel
        i = (idx // Nel) % Nx
        j = (idx // (Nel * Nx)) % Nz

        xr_i = xroi[i]
        zr_j = zroi[j]
        xt_n = xt[n]
        zt_n = zt[n]

        xe = newton_circ(
            xt_n, zt_n,
            xr_i, zr_j,
            c_coupling, c_specimen,
            xcenter, zcenter, radius,
            x0=0,
            maxiter=10,
            tol=1e-6
        )
        ze = f_circ(xe, xcenter, zcenter, radius)

        t_coupling = np.sqrt((xe - xt_n) ** 2 + (ze - 0) ** 2) / c_coupling
        t_specimen = np.sqrt((xe - xr_i) ** 2 + (ze - zr_j) ** 2) / c_specimen

        t_tfm[j, i, n] = t_coupling + t_specimen

    return t_tfm

@numba.njit(parallel=True)
def compute_t_pwi(thetas, xroi, zroi, tref, xref, zref, c_specimen):
    Nangs, Nx, Nz = thetas.shape[0], xroi.shape[0], zroi.shape[0]

    t_pwi = np.zeros((Nangs, Nz, Nx), dtype=np.float32)

    for idx in numba.prange(Nangs * Nx * Nz):
        v = idx % Nangs
        i = (idx // Nangs) % Nx
        j = (idx // (Nangs * Nx)) % Nz

        xi = xroi[i]
        zj = zroi[j]
        ux = np.sin(thetas[v])
        uz = np.cos(thetas[v])

        t_pwi[v, j, i] = tref - (ux * (xref - xi) + uz * (zref - zj)) / c_specimen

    return t_pwi

@numba.njit(parallel=True)
def compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen):
    Nangs, Nel = thetas.shape[0], xt.shape[0]

    xref, zref = xcenter, zcenter
    tv = np.zeros((Nangs, Nel), dtype=np.float32)

    for idx in numba.prange(Nangs * Nel):
        v = idx % Nangs       # angle index
        n = idx // Nangs      # element index

        # Coordinates of element n
        xt_n = xt[n]
        zt_n = zt[n]

        # Compute refraction point on circular interface
        xe = newton_pwi(
            thetas[v],
            xt_n, zt_n,
            c_coupling, c_specimen,
            xcenter, zcenter, radius,
            x0=0,
            maxiter=10,
            tol=1e-6
        )
        ze = f_circ(xe, xcenter, zcenter, radius)

        # Unit vector for the incident wave
        ux = np.sin(thetas[v])
        uz = np.cos(thetas[v])

        # Coupling and specimen travel times
        t_coupling = np.sqrt((xe - xt_n)**2 + (ze - zt_n)**2) / c_coupling
        t_specimen = np.abs(ux * (xref - xe) + uz * (zref - ze)) / c_specimen

        tv[v, n] = t_coupling + t_specimen

    return tv