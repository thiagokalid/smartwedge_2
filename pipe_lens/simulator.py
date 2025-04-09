from numpy import ndarray

from pipe_lens.raytracer import RayTracer

__all__ = ["Simulator"]

class Simulator:
    def __init__(self, raytracer: RayTracer, directivity: bool=False, transmission_loss: bool=False):
        self.raytracer = raytracer
        self.directivity = directivity
        self.transmission_loss = transmission_loss
        self.simulation_list = []

    def create_simulation(self, xf, zf):
        # Creates a new simulation batch
        raise NotImplementedError

    def simulate_batch(self):
        # Returns FMC list
        raise NotImplementedError

    def get_sscans(self, delay_law: ndarray):
        # Calls simulate_batch and convert FMC to s-scans based on given delay law
        raise NotImplementedError

    def simulate(self, delay_law=None):
        # Computes theoretical FMC response
        raise NotImplementedError
