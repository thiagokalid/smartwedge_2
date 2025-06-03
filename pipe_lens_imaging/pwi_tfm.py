import numpy as np
import numba

from .pwi_tfm_utils import *
import time

def cpwc_circle_kernel(pwi_data, xroi, zroi, xt, zt, xcenter, zcenter, radius, thickness, thetas, c_coupling, c_specimen, fs, gate_start=0, insideMaterialMask=True, spatialWeightingMask=True, verbose=True):
    Nrows, Ncols = zroi.shape[0], zroi.shape[1]
    Nangs = pwi_data.shape[1]
    baseline_shift = np.int64(gate_start * fs)

    # Compute the delay-law:
    t0 = time.time()
    pwi_tof, xe, ze = compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen)
    t_ref = np.max(pwi_tof)
    delay_law = t_ref - pwi_tof
    if verbose:
        print(f"delay-law. Time-elapsed = {time.time() - t0:.2f} s")

    t0 = time.time()

    mask = np.ones(shape=(Nangs, Nrows, Ncols))

    if insideMaterialMask:
        tmp_mask = is_inside_pipe(xroi, zroi, xcenter, zcenter, radius, thickness)
        mask *= tmp_mask[np.newaxis, ...]
    if verbose:
        print(f"pipe_filter. Time-elapsed = {time.time() - t0:.2f} s")

    t0 = time.time()
    if spatialWeightingMask:
        mask *= is_inside_pwr(thetas, xroi, zroi, xt, zt, xe, ze, xcenter, zcenter, radius, c_coupling, c_specimen)
    if verbose:
        print(f"pwi_filter. Time-elapsed = {time.time() - t0:.2f} s")

    binary_mask = np.sum(mask, axis=0) > 0

    t0 = time.time()
    t_tfm = compute_t_tfm(xroi, zroi, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen, binary_mask)
    if verbose:
        print(f"t_tfm. Time-elapsed = {time.time() - t0:.2f} s")

    t0 = time.time()
    t_pwi = compute_t_pwi(
        thetas, xroi, zroi, 
        t_ref, xcenter, zcenter,
        c_specimen,
        binary_mask)
    if verbose:
        print(f"t_pwi. Time-elapsed= {time.time() - t0:.2f} s")

    t0 = time.time()
    img = cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift, mask)
    if verbose:
        print(f"coherent-sum. Time-elapsed= {time.time() - t0:.2f} s")

    return img, delay_law

@numba.njit(fastmath=True, parallel=True)
def cpwc_coherent_sum(pwi_data, t_pwi, t_tfm, fs, baseline_shift, mask):
    Nt, Nangs, Nel = pwi_data.shape
    Nrows, Ncols, _ = t_tfm.shape

    img = np.zeros((Nrows, Ncols), dtype=np.float32)
    thread_imgs = np.zeros((numba.get_num_threads(), Nrows, Ncols), dtype=np.float32)

    for idx in numba.prange(Nangs * Nel):
        k = idx // Nel
        n = idx % Nel

        thread_id = numba.np.ufunc.parallel._get_thread_id()

        for j in range(Nrows):
            for i in range(Ncols):
                if mask[k, j, i] == False:
                    continue

                ti = t_pwi[k, j, i]
                tv = t_tfm[j, i, n]
                shift = int(round((ti + tv) * fs)) - baseline_shift

                if 0 <= shift < Nt:
                    thread_imgs[thread_id, j, i] += pwi_data[shift, k, n]

    for t in range(numba.get_num_threads()):
        img += thread_imgs[t]

    return img

@numba.njit(parallel=True)
def compute_t_tfm(xroi, zroi, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen, mask):
    Nx, Nz, Nel = xroi.shape[1], zroi.shape[0], xt.shape[0]

    t_tfm = np.zeros((Nz, Nx, Nel), dtype=np.float32)

    for idx in numba.prange(Nx * Nz * Nel):
        n = idx % Nel
        i = (idx // Nel) % Nx
        j = (idx // (Nel * Nx)) % Nz

        if mask[j, i] == False:
            continue

        xr_i = xroi[j, i]
        zr_j = zroi[j, i]
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
def compute_t_pwi(thetas, xroi, zroi, tref, xref, zref, c_specimen, mask):
    Nangs, Nx, Nz = thetas.shape[0], xroi.shape[1], zroi.shape[0]

    t_pwi = np.zeros((Nangs, Nz, Nx), dtype=np.float32)

    for idx in numba.prange(Nangs * Nx * Nz):
        v = idx % Nangs
        i = (idx // Nangs) % Nx
        j = (idx // (Nangs * Nx)) % Nz

        if mask[j, i] == False:
            continue

        xi = xroi[j, i]
        zj = zroi[j, i]
        ux = np.sin(thetas[v])
        uz = np.cos(thetas[v])

        t_pwi[v, j, i] = tref - (ux * (xref - xi) + uz * (zref - zj)) / c_specimen

    return t_pwi

@numba.njit(parallel=True)
def compute_delaylaw_pwi(thetas, xt, zt, xcenter, zcenter, radius, c_coupling, c_specimen):
    Nangs, Nel = thetas.shape[0], xt.shape[0]

    # if False:
    xref, zref = xcenter, zcenter

    xe, ze = np.zeros(shape=(Nangs, Nel), dtype=np.float32), np.zeros(shape=(Nangs, Nel), dtype=np.float32)

    tv = np.zeros((Nangs, Nel), dtype=np.float32)

    for idx in numba.prange(Nangs * Nel):
        v = idx % Nangs       # angle index
        n = idx // Nangs      # element index

        # Coordinates of element n
        xt_n = xt[n]
        zt_n = zt[n]

        # Compute refraction point on circular interface
        xe[v, n] = newton_pwi(
            thetas[v],
            xt_n, zt_n,
            c_coupling, c_specimen,
            xcenter, zcenter, radius,
            x0=0,
            maxiter=15,
            tol=1e-9
        )
        ze[v, n] = f_circ(xe[v, n], xcenter, zcenter, radius)

        # Unit vector for the incident wave
        ux = np.sin(thetas[v])
        uz = np.cos(thetas[v])

        # Coupling and specimen travel times
        t_coupling = np.sqrt((xe[v, n] - xt_n)**2 + (ze[v, n] - zt_n)**2) / c_coupling
        t_specimen = np.abs(ux * (xref - xe[v, n]) + uz * (zref - ze[v, n])) / c_specimen

        tv[v, n] = t_coupling + t_specimen

    return tv, xe, ze

def is_inside_pipe(xroi, zroi, xcenter, zcenter, radius, thickness):
    mask1 = np.sqrt((xcenter - xroi) ** 2 + (zcenter - zroi)**2) <= radius
    mask2 = np.sqrt((xcenter - xroi) ** 2 + (zcenter - zroi)**2) >= radius - thickness
    return mask1 * mask2

def is_inside_pwr(thetas, xroi, zroi, xt, zt, xe, ze, xcenter, zcenter, radius, c_coupling, c_specimen):
    Nangs = thetas.shape[0]
    Nx, Nz = xroi.shape[0], xroi.shape[1]
    mask = np.zeros((Nangs, Nx, Nz), dtype=bool)  # start with all False

    for v in range(Nangs):
        for ii in [0, -1]:
            # ii = -1# 0 = left boundary ray, -1 = right boundary ray
            xt_min, zt_min = xt[ii], zt[ii]
            xe_min, ze_min = xe[v, ii], ze[v, ii]

            # Compute distance between transmitter and entry point
            den = np.sqrt((xe_min - xt_min)**2 + (ze_min - zt_min)**2)

            # Surface slope of circular arc
            dfdx = (xe_min - xcenter) / np.sqrt(radius**2 - (xe_min - xcenter)**2)

            # Compute incidence angle (sine)
            sine_incidence = -dfdx * (ze_min - zt_min) / den - (xe_min - xt_min) / den

            # Snell's Law: refracted angle (sine)
            sin_arg = c_specimen / c_coupling * sine_incidence
            sin_arg = np.clip(sin_arg, -1, 1)
            ang_refract = np.arcsin(sin_arg)


            # Surface tangent and normal unit vectors
            t_hat = np.array([1, dfdx]) / np.sqrt(1 + dfdx**2)
            n_hat = np.array([-dfdx, 1]) / np.sqrt(1 + dfdx ** 2)

            # Direction of refracted ray
            ray_dir = np.cos(ang_refract) * n_hat + np.sin(ang_refract) * (-t_hat)

            # if ii == 0:
            #     ray_dir_left = ray_dir
            # else:
            #     ray_dir_right = ray_dir

            dx, dz = ray_dir[0], ray_dir[1]
            line_slope = dz / dx

            # Refracted ray line through (xe_min, ze_min)
            # a = line_slope
            # b = ze_min - line_slope * xe_min

            x1, z1 = xe_min, ze_min
            x2, z2 = xe_min + 1, ze_min + line_slope

            # Direction vector of the line
            dx, dz = x2 - x1, z2 - z1

            # Vector from line start to test point
            vx, vz = xroi - x1, zroi - z1

            # Cross product
            cross = dx * vz - dz * vx

            if thetas[v] >= 0:
                if ii == 0:
                    mask_left = cross <= 0
                else: # ii = -1
                    mask_right = cross >= 0

            else:
                if ii == 0:
                    mask_left = cross >= 0
                else: # ii = -1
                    mask_right = cross <= 0
            #
            # if v == 141 - 60:
            #     pass
            # if v == 141 + 60:
            #     pass
            #
            # if v == -1:
            #     print(np.degrees(np.arcsin(sine_incidence)))
            #     print(np.degrees(ang_refract))
            #
            #     import matplotlib
            #     import matplotlib.pyplot as plt
            #     matplotlib.use('TkAgg')
            #     xspan = np.arange(-radius, radius + .1e-3, .1e-3)
            #     xmin, xmax = np.min(xroi) * 1e3, np.max(xroi) * 1e3
            #     zmin, zmax = np.min(zroi) * 1e3, np.max(zroi) * 1e3
            #
            #
            #     plt.plot([xmin, xmin, xmax, xmax], [zmin, zmax, zmax, zmin], 'k', markersize=1)
            #     plt.plot(xspan * 1e3, f_circ(xspan, xcenter, zcenter ,radius) * 1e3)
            #     plt.plot(xt * 1e3, zt * 1e3, 'sk')
            #     plt.plot([xt[0] * 1e3, xe[v, 0] * 1e3], [zt[0] * 1e3, ze[v, 0] * 1e3], 'lime', linewidth=1)
            #     plt.plot([xt[-1] * 1e3, xe[v, -1] * 1e3], [zt[-1] * 1e3, ze[v, -1] * 1e3], 'lime', linewidth=1)
            #     plt.plot(xt[0] * 1e3, zt[0] * 1e3, 'sr')
            #     plt.plot(xt[-1] * 1e3, zt[-1] * 1e3, 'sr')
            #
            #     plt.plot(xcenter * 1e3, zcenter * 1e3, 'sr')
            #
            #     origin = [xe[v, 0] * 1e3], [ze[v, 0] * 1e3]
            #     plt.quiver(*origin, *(ray_dir_left * 1e1), angles='xy', scale_units='xy', scale=.5, color='r')
            #
            #     origin = [xe[v, -1] * 1e3], [ze[v, -1] * 1e3]
            #     plt.quiver(*origin, *(ray_dir_right * 1e1), angles='xy', scale_units='xy', scale=.5, color='r')
            #
            #     plt.axis("equal")
            #     plt.ylim([0, zcenter * 1e3])
            #     plt.show()
            #
            #     plt.figure()
            #     ext = [np.min(xroi) * 1e3, np.max(xroi) * 1e3, np.min(zroi) * 1e3, np.max(zroi) * 1e3]
            #     plt.imshow(mask_right * mask_left, extent=ext, aspect='auto')
            #     plt.show()

        # Final mask for this angle: inside left and right rays
        mask[v] = mask_left * mask_right

    return mask