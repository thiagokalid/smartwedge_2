import numpy as np

from numpy import cos, sin, deg2rad

from pipe_lens.transducer import Transducer
from pipe_lens.geometric_utils import Pipeline
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.raytracer import RayTracer

class DelayLaw:
    def __init__(self, wave_type: str, transducer: Transducer, reception_type="same"):
        if wave_type in ["focused wave", "plane wave"]:
            self.wave_type = wave_type
        else:
            raise NotImplementedError("Unsupported wave_type: {}".format(wave_type))

        self.transducer = transducer
        self.emission_law = None
        self.reception_law = None

        self.reception_type = reception_type


class FocusedWave(DelayLaw):
    def __init__(self, acoustic_lens: AcousticLens, pipeline: Pipeline, transducer: Transducer):
        super().__init__(wave_type="focused wave", transducer=transducer)
        self.pipeline = pipeline
        self.acoustic_lens = acoustic_lens

        # Create the ray-tracer solver:
        self.raytracer = RayTracer(self.acoustic_lens, self.pipeline, self.transducer)

        # Attributes later to be defined.
        self.xf, self.zf = None, None
        self.tof_matrix = None

    def compute(self, focusing_radius: float= 65e-3, alpha_min: float= -45., alpha_max: float= 45., delta_alpha:float= .5):
        pipeline_angs = deg2rad(np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha))

        # Foci coordinates:
        self.xf, self.zf = focusing_radius * cos(pipeline_angs), focusing_radius * sin(pipeline_angs)

        solutions = self.raytracer.solve(self.xf, self.zf)
        self.tof_matrix = np.asarray([solution["tof"] for solution in solutions], dtype=float) # Dimension: Nchannels x Nfocus
        self.emission_law = self.tof_matrix.max() - self.tof_matrix

        if self.reception_type == "same":
            self.reception_law = self.emission_law
        else:
            raise NotImplementedError("Unsupported reception law.")


# class FocusedWave(DelayLaw):


        # if self.type == "focused wave":
        #     # Ensure required arguments exist
        #     required_keys = ["focusing_radius", "alpha_max", "alpha_min", "delta_alpha"]
        #     for key in required_keys:
        #         if key not in kwargs:
        #             raise ValueError(f"Missing required argument '{key}' for focused wave.")
        #
        #     # Assign variables
        #     self.focus = kwargs["focusing_radius"]
        #     self.focusing_radius = kwargs["alpha_max"]
        #     self.alpha_min = kwargs["alpha_max"]
        #     self.delta_alpha = kwargs["delta_alpha"]
        #
        # elif self.type == "plane wave":
        #     required_keys = ["alpha_min", "alpha_max", "delta_alpha"]

    # def compute(self):