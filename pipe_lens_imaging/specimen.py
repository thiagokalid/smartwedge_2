import numpy as np
import matplotlib.pyplot as plt

from bisect import bisect
from pipe_lens_imaging.geometric_utils import rotate
from pipe_lens_imaging.specimen_utils import *
from pipe_lens_imaging.utils import get_class_by_attribute

__all__ = ["PipeWithGrooves", "get_specimen", "specimens"]


class GenericSpecimen:
    def __init__(self, origin, rotation_around_origin_deg):
        self.xdict = {}
        self.ydict = {}
        self.zdict = {}
        self.rdict = {}
        self.thetadict = {}
        self.delta_x = self.delta_y = self.delta_z = .05e-3  # in mm
        self.theta_discr = .005 * np.pi / 180  # in deg
        self.rdiscr = .05e-3  # in mm
        self.origin = origin
        self.center = self.origin
        self.rotation_around_origin_deg = rotation_around_origin_deg

    def fit(self):
        # Interface method.
        pass

    def adjust(self):
        if self.rotation_around_origin_deg != 0:
            # Rotates according to user configuration:
            self.rotate(self.rotation_around_origin_deg)

        if self.origin[0] != 0 and self.origin[1] != 0:
            # Shifts according to user configuration:
            self.move(self.origin)

        # Finds polar equivalents:
        self.rdict, self.thetadict = cart2polar_coord_dict(self.xdict, self.zdict)

        return None

    def rotate(self, rotate_ang_deg):
        self.xdict, self.zdict = rotate_coord_dict(self.xdict, self.zdict, rotate_ang_deg)

        # Finds polar equivalents:
        self.rdict, self.thetadict = cart2polar_coord_dict(self.xdict, self.zdict)

    def move(self, new_center=(0, 0)):
        self.xdict, self.zdict = shift_coord_dict(self.xdict, self.zdict, new_center, self.center)
        self.center = new_center

        # Finds polar equivalents:
        self.rdict, self.thetadict = cart2polar_coord_dict(self.xdict, self.zdict)

    def get_coords(self, coord_system="polar"):
        # If it's an invalid coordinate system:
        if coord_system not in ["cartesian", "polar"]:
            raise TypeError("Invalid coordinate system.")

        if coord_system == "cartesian":
            return self.xdict, self.zdict
        elif coord_system == "polar":
            return self.rdict, self.thetadict

    def draw(self, coord_system="cartesian", axis=plt.gca(), show=False, configs:dict = {''}, scale=1):
        self.plot_surface(coord_system, axis, configs, scale=scale)
        # axis.set_aspect('equal')
        if show:
            plt.show()

    def plot_surface(self, coord_system, axis, configs, scale=1):
        # If it's an invalid coordinate system:
        if coord_system not in ["cartesian", "polar"]:
            raise TypeError("Invalid coordinate system.")

        if coord_system == "cartesian":
            x1, x2 = self.xdict, self.zdict
        else:  # polar
            x1, x2 = self.rdict, self.thetadict

        iter = 0
        for key in x1.keys():
            for a1, a2 in zip(x1[key], x2[key]):
                if iter == 0:
                    curr_label = configs['label']
                else:
                    curr_label = "_"
                if coord_system == "polar":
                    ang_deg = np.rad2deg(a2)
                    r = a1
                    axis.plot(ang_deg, r * scale, **configs)
                else:
                    x, z = a1, a2
                    axis.plot(x * scale, z * scale, **configs)
                iter += 1

        return None

    def __create_flat_bottom_hole(self, flaw_description):
        # Interface;
        pass

    def __create_rounded_bottom_hole(self, flaw_description):
        # Interface;
        pass


class PipeWithGrooves(GenericSpecimen):
    def __init__(self, outer_radius:float, wall_thickness: float, cl:float, rho: float,
                 flaws_description=None, origin=(0, 0), rotation_around_origin_deg=0
                 , angular_span_deg=(-180, 180)):
        super().__init__(origin, rotation_around_origin_deg)

        # Important features:
        keys = ["inner_surface", "outer_surface"]
        self.xdict = {k: [] for k in keys}
        self.ydict = {k: [] for k in keys}
        self.zdict = {k: [] for k in keys}
        self.transition_angs = {k: [] for k in keys}
        self.flaws_transition_angs = {k: [] for k in keys}
        self.flaws_description = flaws_description
        self.outer_radius = outer_radius  # in mm
        self.inner_radius = outer_radius - wall_thickness  # in mm
        self.wall_thickness = wall_thickness
        self.cl = cl
        self.rho = rho

        # The angle clockwise direction is positive, and the 0º is coincident with the z-axis.
        # Tubes are defined as [-180º, 180º].
        # If it's defined, for instance, as [-45º, 45º], it will be a cylindrical section.
        self.angular_span = tuple(np.deg2rad(angular_span_deg))

        # Creates the specimen features:
        self.fit()

        # Adjust according to user-defined parameters:
        self.adjust()

    def get_inner_surface(self, ang_deg):
        if self.center != (0, 0):
            raise ValueError("The tube must be centered at (0, 0)")
        else:
            return find_multiple_radii_from_ang(ang_deg, self.rdict, self.thetadict, surf_type="inner_surface")

    def get_outer_surface(self, ang_deg):
        if self.center != (0, 0):
            raise ValueError("The tube must be centered at (0, 0)")
        else:
            return find_multiple_radii_from_ang(ang_deg, self.rdict, self.thetadict, surf_type="outer_surface")

    def fit(self):
        self.__create_outer_surface()
        self.__create_flaws()
        self.__create_inner_surface()
        return None

    def __create_outer_surface(self):
        theta_span = np.arange(self.angular_span[0], self.angular_span[1] + self.theta_discr, self.theta_discr)
        r = self.outer_radius * np.ones_like(theta_span)

        # Converts to cartesian coordinate system:
        x = r * np.sin(theta_span)
        z = r * np.cos(theta_span)
        transition_ang = (theta_span[0], theta_span[-1])

        # Appends to the dictionary:
        self.xdict['outer_surface'].append(x)
        self.zdict['outer_surface'].append(z)
        self.transition_angs['outer_surface'].append(transition_ang)
        return None

    def __create_flaws(self):
        if self.flaws_description is not None:
            for flaw_description in self.flaws_description:
                if flaw_description['type'] == types_of_flaw[0]:
                    # Flat bottom hole
                    self.__create_flat_bottom_hole(flaw_description)

                elif flaw_description['type'] == types_of_flaw[1]:
                    # Hemispherical bottom hole
                    self.__create_hemispherical_bottom_hole(flaw_description)
                else:
                    raise TypeError("Invalid flaw type.")
        else:
            pass

    def __create_inner_surface(self):
        inner_surf_angs = divide_angular_span(self.angular_span, self.flaws_transition_angs["inner_surface"])
        for segment in inner_surf_angs:
            theta = np.arange(segment[0], segment[1] + self.theta_discr, self.theta_discr)
            r = self.inner_radius * np.ones_like(theta)

            x = r * np.sin(theta)
            z = r * np.cos(theta)

            self.xdict["inner_surface"].append(x)
            self.zdict["inner_surface"].append(z)
            self.transition_angs["inner_surface"].append(segment)

    def __create_flat_bottom_hole(self, flaw_description):
        flaw_residual_thickness = flaw_description["residual_thickness"]
        flaw_rotation_ang_deg = flaw_description["rotation_ang_deg"]
        flaw_width = flaw_description["width"]

        rmax = self.outer_radius - flaw_residual_thickness
        x1 = -flaw_width / 2
        x2 = flaw_width / 2

        xtop = np.arange(x1, x2 + self.delta_x, self.delta_x)
        ztop = rmax + np.zeros_like(xtop)

        # Connects the top of the flaw to the inner surface with radius = self.inner_radius:
        self.__connect_top_to_bottom_flaw(rmax, x1, x2, xtop, ztop, flaw_rotation_ang_deg)

    def __create_hemispherical_bottom_hole(self, flaw_description):
        flaw_residual_thickness = flaw_description["residual_thickness"]
        flaw_rotation_ang_deg = flaw_description["rotation_ang_deg"]
        flaw_radius = flaw_description["radius"]
        flaw_width = flaw_description["width"]

        rmax = self.outer_radius - flaw_residual_thickness - flaw_radius
        x1 = -flaw_width
        x2 = flaw_width

        theta_top = np.arange(-np.pi / 2, np.pi / 2 + self.theta_discr, self.theta_discr)
        xtop = flaw_radius * np.sin(theta_top)
        ztop = flaw_radius * np.cos(theta_top) + rmax + np.zeros_like(xtop)

        # Connects the top of the flaw to the inner surface with radius = self.inner_radius:
        self.__connect_top_to_bottom_flaw(rmax, x1, x2, xtop, ztop, flaw_rotation_ang_deg)

    def __connect_top_to_bottom_flaw(self, rmax, x1, x2, xtop, ztop, ang_deg):
        zleft = np.arange(self.inner_radius, rmax + self.delta_z, self.delta_z)
        xleft = x1 + np.zeros_like(zleft)

        zright = np.arange(self.inner_radius, rmax + self.delta_z, self.delta_z)
        xright = x2 + np.zeros_like(zleft)

        # Rotate according to flaw location:
        xtop, ztop = rotate(xtop, ztop,
                            angle=np.deg2rad(ang_deg))
        xleft, zleft = rotate(xleft, zleft,
                              angle=np.deg2rad(ang_deg))
        xright, zright = rotate(xright, zright,
                                angle=np.deg2rad(ang_deg))

        xlist = [xleft, xright, xtop, xleft, xright]
        zlist = [zleft, zright, ztop, zleft, zright]
        theta_list = []
        for i in range(len(xlist)):
            x = xlist[i]
            z = zlist[i]
            ang1, ang2 = find_transition_angles(x, z)

            if ang1 > ang2:
                theta = np.arange(start=ang2, stop=ang1 + self.theta_discr, step=self.theta_discr)
                theta = theta[::-1]
            else:
                theta = np.arange(start=ang1, stop=ang2 + self.theta_discr, step=self.theta_discr)

            theta_list.append(theta)

        xlist, _ = adjust_the_phase_wrap(xlist, theta_list)
        zlist, theta_list = adjust_the_phase_wrap(zlist, theta_list)

        for x, z, theta in zip(xlist, zlist, theta_list):
            self.xdict["inner_surface"].append(x)
            self.zdict["inner_surface"].append(z)
            self.transition_angs["inner_surface"].append((theta[0], theta[-1]))

        a1, a2 = find_transition_angles(xleft, zleft)
        b1, b2 = find_transition_angles(xright, zright)
        if np.abs(a1 - b1) > (2 * np.pi) * .9:
            self.flaws_transition_angs["inner_surface"].append((a1, 2 * np.pi))
            self.flaws_transition_angs["inner_surface"].append((-2 * np.pi, b1))
        else:
            self.flaws_transition_angs["inner_surface"].append((a1, b1))


class PipeSBH:
    def __init__(self,
                 outer_radius=71.05e-3,
                 wall_thickness=18.67e-3,
                 height=250e-3,
                 ang_lim=45,
                 ystep=0.5e-3,
                 ang_step=0.5,
                 flaw_diameter=5e-3,
                 theta_offset=0,
                 includeEnd=True,):
        self.outer_radius = outer_radius
        self.wall_thickness = wall_thickness
        self.inner_radius = outer_radius - wall_thickness
        self.height = height
        self.ang_lim = ang_lim
        self.ystep = ystep
        self.ang_step = ang_step
        self.flaw_radius = flaw_diameter / 2

        if includeEnd:
            self.theta_span = np.radians(np.arange(-ang_lim, ang_lim + ang_step, ang_step) - theta_offset)
        else:
            self.theta_span = np.radians(np.arange(-ang_lim, ang_lim, ang_step) - theta_offset)

        self.y_span = np.arange(0, height, ystep)
        self.n_theta = len(self.theta_span)
        self.n_y = len(self.y_span)

        self.inner_surface = np.full((self.n_theta, self.n_y), self.inner_radius)
        self.outer_surface = np.full((self.n_theta, self.n_y), self.outer_radius)

        self.tt, self.yy = np.meshgrid(self.theta_span, self.y_span, indexing="ij")
        self.flaw_coords = []

    def add_flaw_region(self, nrows, ncols, y0, ystep_flaw, dist_flaw, intercalated):
        for i in range(nrows):
            y_flaw = y0 + i * ystep_flaw
            ncols_curr = ncols - 1 if intercalated and i % 2 == 1 else ncols

            arc_length_mm = (ncols_curr - 1) * dist_flaw
            arc_length_deg = np.degrees(arc_length_mm / self.inner_radius)

            ang_flaws = np.linspace(0, arc_length_deg, ncols_curr)
            ang_flaws -= np.mean(ang_flaws)

            self.flaw_coords.extend([(ang, y_flaw) for ang in ang_flaws])

    def carve_flaws(self):
        for theta_deg, y0 in self.flaw_coords:
            theta0_rad = np.radians(theta_deg)
            for i in range(self.n_theta):
                for j in range(self.n_y):
                    d_theta = (self.tt[i, j] - theta0_rad) * self.inner_radius
                    d_y = self.yy[i, j] - y0
                    dist2 = d_theta**2 + d_y**2
                    if dist2 <= self.flaw_radius**2:
                        dz = np.sqrt(self.flaw_radius**2 - dist2)
                        self.inner_surface[i, j] += dz

    def get_thickness_map(self, y_start=50e-3, y_end=200e-3):
        thickness_map = (self.outer_surface - self.inner_surface).T * 1e3
        idx_beg = bisect(self.y_span, y_start) - 1
        idx_end = bisect(self.y_span, y_end)
        return thickness_map[idx_beg:idx_end, :]

    def get_flaw_mask(self, flaw_number, y_start=0, y_end=None, radius_tol = 1e-3):



        beg_idx = bisect(self.y_span, y_start) - 1

        if y_end is not None:
            end_idx = bisect(self.y_span, y_end)

            tt, yy = np.meshgrid(self.theta_span, self.y_span[beg_idx:end_idx], indexing="ij")

        else:
            tt, yy = self.tt, self.yy


        yy_mask = yy
        xx_mask, zz_mask = self.inner_radius * np.sin(tt), self.inner_radius * np.cos(tt)

        theta_deg, y_flaw = self.flaw_coords[flaw_number]

        theta0_rad = np.radians(theta_deg)

        x_flaw, z_flaw = self.inner_radius * np.sin(theta0_rad), self.inner_radius * np.cos(theta0_rad)

        mask = (x_flaw - xx_mask)**2 + (z_flaw - zz_mask)**2 + (y_flaw - yy_mask)**2 <= (self.flaw_radius + radius_tol)**2

        return mask.T


    def get_slice(self, theta_deg, y_pos):
        theta_idx = bisect(np.degrees(self.theta_span), theta_deg) - 1
        y_idx = bisect(self.y_span, y_pos) - 1
        return self.inner_surface[theta_idx, y_idx], self.outer_surface[theta_idx, y_idx]



def get_specimen(searched_material_attribute, attribute_name="nicknames"):
    return get_class_by_attribute(specimens, searched_material_attribute, PipeWithGrooves,
                                  class_name="specimen", attribute_name=attribute_name)


types_of_flaw = ["flat bottom hole", "hemispherical bottom hole"]
specimens = [
    # Tubo com Sulcos. Projeto: "https://drive.google.com/drive/folders/1R2-djVlUA4pBR4HvYNucP0iLa6vFE_3w"
    {
        "nicknames": ["Pipe with grooves"],
        "type": "Tubular",
        "attributes": {
            "outer_radius": 139.61e-3/2,
            "wall_thickness": 18.19e-3,
            "cl": 5900,
            "rho": 7.85,
            "flaws_description":
                [
                    # Flaw 1:
                    {
                        "type": "flat bottom hole",
                        "width": 4e-3,  # in mm
                        "residual_thickness": 16.08e-3,  # in mm
                        "rotation_ang_deg": 0
                    },

                    # Flaw 2:
                    {
                        "type": "hemispherical bottom hole",
                        "radius": 2e-3,
                        "width": 2e-3,
                        "residual_thickness": 16.12e-3,
                        "rotation_ang_deg": 45
                    },

                    # Flaw 3:
                    {
                        "type": "hemispherical bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 14.08e-3,
                        "rotation_ang_deg": 90
                    },

                    # Flaw 4:
                    {
                        "type": "hemispherical bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 9.97e-3,
                        "rotation_ang_deg": 135
                    },

                    # Flaw 5:
                    {
                        "type": "hemispherical bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 6.14e-3,
                        "rotation_ang_deg": 180
                    },

                    # Flaw 6:
                    {
                        "type": "flat bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 6.07e-3,
                        "rotation_ang_deg": 225
                    },

                    # Flaw 7:
                    {
                        "type": "flat bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 10.00e-3,
                        "rotation_ang_deg": 270
                    },

                    # Flaw 8:
                    {
                        "type": "flat bottom hole",
                        "radius": 4e-3,
                        "width": 4e-3,
                        "residual_thickness": 14.02e-3,
                        "rotation_ang_deg": 305
                    },
                ],
            "origin": (0, 0),
            "rotation_around_origin_deg": 0,
            "angular_span_deg": (-180, 180)
        }

    },

    # ABC. Projeto: XYZ
]

