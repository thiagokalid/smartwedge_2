import numpy as np

from numba import prange

from pipe_lens_imaging.raytracer import RayTracer
from pipe_lens_imaging.simulator_utils import fmc_sim_kernel, fmc2sscan

__all__ = ["Simulator"]

FLOAT = np.float32

class Simulator:
    def __init__(self, sim_params: dict, raytracer: RayTracer):
        self.raytracer = raytracer

        # Reflectors position:
        self.xf, self.zf = None, None

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

    def __simulate(self):
        Nel = self.raytracer.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)

        tofs, amplitudes = self.raytracer.solve(self.xf, self.zf)
        self.fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=FLOAT)

        for i in prange(Nsim):
            self.fmcs[..., i] = fmc_sim_kernel(self.tspan, tofs[:, i], amplitudes[:, i], Nel, self.raytracer.transducer.fc, self.raytracer.transducer.bw)

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