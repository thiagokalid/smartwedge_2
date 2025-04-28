import numpy as np

from numpy import ndarray

from pipe_lens.raytracer import RayTracer

from pipe_lens_imaging.simulator_utils import dist

from tqdm import tqdm

from numba import njit, prange

__all__ = ["Simulator"]

FLOAT = np.float32

class Simulator:
    def __init__(self, sim_params: dict, raytracer: RayTracer, directivity: bool=False, transmission_loss: bool=False):
        self.raytracer = raytracer
        self.directivity = directivity
        self.transmission_loss = transmission_loss

        # Reflectors position:
        self.xf, self.zf = None, None
        self.tofs = None  # tof between each element and each reflector.

        # Gate-related parameters:
        self.fs = sim_params["fs"]  # Sampling frequency in Hz
        self.gate_start = sim_params["gate_start"]  # Gate start in seconds
        self.gate_end = sim_params["gate_end"]  # Gate end in seconds
        self.tspan = np.arange(self.gate_start, self.gate_end + 1 / self.fs, 1 / self.fs)  # Time-grid.

        # Inspection type:
        self.response_type = sim_params["response_type"]  # e.g. fmc, s-scan
        match self.response_type:
            case "s-scan":
                self.delaylaw_t, self.delaylaw_r = sim_params["emission_delaylaw"], sim_params["reception_delaylaw"]

                # Delay law in number of samples to shift during DAS:
                self.shifts_e = np.round(self.delaylaw_t * self.fs)
                self.shifts_r = np.round(self.delaylaw_r * self.fs)

            case "fmc":
                pass

            case _:
                raise NotImplementedError

        # Configuration list of multiple simulations instances:
        self.sim_list = []
        self.fmcs = None
        self.sscans = None

    def add_reflector(self, xf, zf, different_instances: bool=False):
        self.xf, self.zf = xf, zf
        if different_instances:
            for x, z in zip(xf, zf):
                sim = {
                    "reflectors": [x, z],
                    "response": None,
                }
                self.sim_list.append(sim)
        else:
            sim = {
                "reflectors": [xf, zf],
                "response": None,
            }
            self.sim_list.append(sim)

    def __compute_amplitudes(self, xf, zf):
        solution = self.raytracer.solve(xf, zf)
        n_reflections = len(xf)
        n_elem = self.transducer.num_elem
        n_focii = len(xf)

        tofs = np.zeros(shape=(self.transducer.num_elem, n_focii), dtype=FLOAT)

        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([xf, zf]).T
        coords_lens = np.zeros(shape=(n_elem, 2, n_focii))
        coords_outer = np.zeros(shape=(n_elem, 2, n_focii))

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, :, i] = solution[j]['xlens'][i], solution[j]['zlens'][i]
            coords_outer[j, :, i] = solution[j]['xpipe'][i], solution[j]['zpipe'][i]

        coord_elements_mat = np.tile(coord_elements[:, :, np.newaxis], (1, 1, n_focii))
        coord_reflectors_mat = np.tile(coords_reflectors[:, :, np.newaxis], (1, 1, n_elem))

        # Compute distances between points where refractions is expected.
        d1 = norm(coords_lens - coord_elements_mat, axis=1)  # distance between elements and lens
        d2 = norm(coords_lens - coords_outer, axis=1)  # distance between lens and pipe outer surface
        d3 = norm(coords_outer - coord_reflectors_mat.T, axis=1)  # distance between pipe outer surface and focus

        c1 = self.acoustic_lens.c1  # lens material
        c2 = self.acoustic_lens.c2  # coupling medium
        c3 = self.pipeline.c  # pipe material

        return d1 / c1 + d2 / c2 + d3 / c3

    def __simulate(self):
        Nel = self.raytracer.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)

        self.tofs = self.raytracer.get_tofs(self.xf, self.zf, maxiter=6)
        amplitudes = self.__compute_amplitudes(self.raytracer)
        self.fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=FLOAT)

        for i in prange(Nsim):
            self.fmcs[..., i] = fmc_sim_kernel(self.tspan, self.tofs[:, i], Nel, self.raytracer.transducer.fc, self.raytracer.transducer.bwr)

    def __get_sscan(self):
        self.fmcs = self.__get_fmc()
        self.sscans = fmc2sscan(
            self.fmcs,
            self.shifts_e,
            self.shifts_r,
            self.raytracer.transducer.num_elem
        )
        return self.sscans

    def __get_fmc(self):
        if self.fmcs is None:
            self.__simulate()
        return self.fmcs

    def get_response(self):
        if len(self.sim_list) == 0:
            raise ValueError("No reflector set. You must add at least one reflector to simulate its response.")

        match self.response_type:
            case "s-scan":
                return self.__get_sscan()

            case "fmc":
                return self.__get_fmc()

            case _:
                raise NotImplementedError

@njit(parallel=True)
def fmc_sim_kernel(tspan: ndarray, tofs: ndarray, n_elem: int, fc_Hz: float, bw: float) -> ndarray:
    ascan_data = np.zeros(shape=(len(tspan), n_elem, n_elem), dtype=FLOAT)

    for combined_idx in prange(n_elem * n_elem):
        idx_e = combined_idx // n_elem
        idx_r = combined_idx % n_elem
        tof_e = tofs[idx_e]
        tof_r = tofs[idx_r]

        # Ideal gausspulse shifted (spatial impulse simulator):
        ascan_data[:, idx_e, idx_r] = numba_gausspulse(tspan - (tof_r + tof_e), fc_Hz, bw)

        # Considering amplitude loss due to transmission coefficient (value is equal to 1 if not considered):
        ascan_data[:, idx_e, idx_r] *= 1 * 1

        # Considering amplitude loss due to directivity (value is equal to 1 if not considered):
        ascan_data[:, idx_e, idx_r] *= 1 * 1

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
    num_laws = shifts_r.shape[0]

    sscan = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    # signal_recepted_by_focus = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    for scan_idx in tqdm(range(num_laws)): #
        # Delay And Sum in emission:
        shift_e = shifts_e[scan_idx, :]
        rolled_fmc = np.zeros_like(fmc_sims)
        for i in range(n_elem):
            rolled_fmc[:, i, :, :] = np.roll(fmc_sims[:, i, :, :], int(shift_e[i]), axis=0)
        das_emission = np.sum(rolled_fmc, axis=1)
        # signal_recepted_by_focus[:, scan_idx, :] = np.sum(das_emission, axis=1)

        # Delay And Sum in reception:
        shift_r = shifts_r[scan_idx, :]
        das = np.zeros_like(das_emission)
        for i in range(num_elems):
            das[:, i, :] = np.roll(das_emission[:, i, :], int(shift_r[i]), axis=0)
        ascan = np.sum(das, axis=1)
        sscan[:, scan_idx, :] = ascan

    # return sscan, signal_recepted_by_focus
    return sscan


# Raytracer has simulation.