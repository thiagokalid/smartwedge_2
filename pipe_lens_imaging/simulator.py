import numpy as np

from numba import prange

from pipe_lens_imaging.raytracer_solver import RayTracerSolver
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.reflection_raytracer import ReflectionRayTracer
from pipe_lens_imaging.simulator_utils import fmc_sim_kernel, fmc2sscan

__all__ = ["Simulator"]

FLOAT = np.float32

class Simulator:
    def __init__(self, sim_params: dict, raytracer_list: list, verbose: bool = True):
        self.raytracer_list = raytracer_list
        self.transducer = self.raytracer_list[0].transducer
        # Reflectors position:
        self.xf, self.zf = None, None
        self.verbose = verbose

        # Gate-related parameters:
        self.fs = sim_params["fs"]  # Sampling frequency in Hz
        self.gate_start = sim_params["gate_start"]  # Gate start in seconds
        self.gate_end = sim_params["gate_end"]  # Gate end in seconds
        self.surface_echoes = sim_params["surface_echoes"]
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
        Nel = self.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)
        self.fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=FLOAT)

        for i, raytracer in enumerate(self.raytracer_list):
            if self.verbose:
                print(f"Raytracer running: {i + 1}/{len(self.raytracer_list)}")
            if isinstance(raytracer, FocusRayTracer):
                self.fmcs += self._simulate_focus_raytracer(raytracer)
            elif isinstance(raytracer, ReflectionRayTracer) and self.surface_echoes:
                self.fmcs += self._simulate_reflection_raytracer(raytracer)  # or handle it appropriately
            else:
                raise NotImplementedError(f"Unknown raytracer type: {type(raytracer)}")

    def _simulate_focus_raytracer(self, focus_raytracer):
        Nel = focus_raytracer.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)

        tofs, amplitudes = focus_raytracer.solve(self.xf, self.zf)
        fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=FLOAT)

        for i in prange(Nsim):
            tx_coeff_i = amplitudes['transmission_loss'][..., i]
            rx_coeff_i = amplitudes['directivity'][..., i] * amplitudes['transmission_loss'][..., i]

            tofs_i = tofs[..., i]
            tofs_i = np.tile(tofs_i[:, np.newaxis], reps=(1, Nel))

            fmcs[..., i] = fmc_sim_kernel(
                self.tspan,
                tofs_i, tofs_i.T,
                tx_coeff_i, rx_coeff_i,
                Nel, focus_raytracer.transducer.fc, focus_raytracer.transducer.bw
            )
        return fmcs

    def _simulate_reflection_raytracer(self, reflection_raytracer):
        Nel = reflection_raytracer.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)

        tofs, amplitudes = reflection_raytracer.solve(self.transducer.xt, self.transducer.zt, solver='scipy-bounded')
        fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=FLOAT)

        for i in prange(Nsim):
            tx_coeff_i = np.nan_to_num(amplitudes['transmission_loss'][..., i], nan=1.)
            rx_coeff_i = np.nan_to_num(amplitudes['directivity'][..., i], nan=1.)

            tofs_tx = np.nan_to_num(tofs[..., i], nan=-1.)
            tofs_rx = 0 * np.nan_to_num(tofs[..., i], nan=-1.)


            fmcs[..., i] = fmc_sim_kernel(
                self.tspan,
                tofs_tx, tofs_rx.T,
                tx_coeff_i, rx_coeff_i,
                Nel, reflection_raytracer.transducer.fc, reflection_raytracer.transducer.bw
            )

        return fmcs

    def __get_sscan(self):
        self.fmcs = self.__get_fmc()
        self.sscans = fmc2sscan(
            self.fmcs,
            self.shifts_e,
            self.shifts_r,
            self.transducer.num_elem
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