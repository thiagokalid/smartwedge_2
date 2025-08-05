from numpy import sqrt
from numba import njit, prange
import numpy as np

from numpy import ndarray

FLOAT = np.float32


def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

@njit(parallel=True)
def fmc_sim_kernel(tspan: ndarray, tofs_tx: ndarray, tofs_rx: np.ndarray, tx_losses: ndarray, rx_losses : ndarray, n_elem: int, fc_Hz: float, bw: float) -> ndarray:
    ascan_data = np.zeros(shape=(len(tspan), n_elem, n_elem), dtype=FLOAT)

    for combined_idx in prange(n_elem * n_elem):
        idx_e = combined_idx // n_elem
        idx_r = combined_idx % n_elem
        tof_e = tofs_tx[idx_e, idx_r]
        tof_r = tofs_rx[idx_e, idx_r]

        if tof_e == -1 or tof_r == -1:
            continue
        else:
            # Ideal gausspulse shifted (spatial impulse simulator):
            ascan_data[:, idx_e, idx_r] = numba_gausspulse(tspan - (tof_r + tof_e), fc_Hz, bw)

        # Losses during transmission, i.e., transmission coefficient.
        ascan_data[:, idx_e, idx_r] *= tx_losses[idx_e, idx_r]

        # Losses during reception. They might include transmission coefficient, reflection coefficients and directivity.
        ascan_data[:, idx_e, idx_r] *= rx_losses[idx_e, idx_r]

    return ascan_data

@njit(fastmath=True)
def numba_gausspulse(t, fc_Hz, bw, bwr=-6):
    ref = pow(10.0, bwr / 20.0)
    a = -(np.pi * fc_Hz * bw) ** 2 / (4.0 * np.log(ref))
    return np.real(np.exp(-a * np.power(t, 2)) * np.exp(1j * 2 * np.pi * fc_Hz * t))


def fmc2sscan(fmc_sims: ndarray, shifts_e, shifts_r, n_elem: int):
    # From a given FMC apply the delays and compute the Summed-Scan (S-Scan):
    num_sims = fmc_sims.shape[-1]
    num_elems = fmc_sims.shape[1]
    num_samples = fmc_sims.shape[0]
    num_laws = shifts_r.shape[1]

    sscan = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    # signal_recepted_by_focus = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    for scan_idx in range(num_laws): #
        # Delay And Sum in emission:
        shift_e = shifts_e[:, scan_idx]
        rolled_fmc = np.zeros_like(fmc_sims)
        for i in range(n_elem):
            rolled_fmc[:, i, :, :] = np.roll(fmc_sims[:, i, :, :], int(shift_e[i]), axis=0)
        das_emission = np.sum(rolled_fmc, axis=1)
        # signal_recepted_by_focus[:, scan_idx, :] = np.sum(das_emission, axis=1)

        # Delay And Sum in reception:
        shift_r = shifts_r[:, scan_idx]
        das = np.zeros_like(das_emission)
        for i in range(num_elems):
            das[:, i, :] = np.roll(das_emission[:, i, :], int(shift_r[i]), axis=0)
        ascan = np.sum(das, axis=1)
        sscan[:, scan_idx, :] = ascan

    # return sscan, signal_recepted_by_focus
    return sscan