import numpy as np

__all__ = ["apply_operation_to_coord_dict", "adjust_the_phase_wrap", "phase_wrap_dict", "cart2polar_coord_dict",
           "shift_coord_dict", "rotate_coord_dict", "find_radius_from_ang", "find_multiple_radii_from_ang",
           "find_transition_angles", "divide_angular_span"]


def apply_operation_to_coord_dict(xdict, zdict, xfun, zfun):
    adict = {k: [] for k in xdict.keys()}
    bdict = {k: [] for k in xdict.keys()}
    for xkey, zkey in zip(xdict.keys(), zdict.keys()):
        for x, z in zip(xdict[xkey], zdict[zkey]):
            a = xfun(x, z)
            b = zfun(x, z)
            adict[xkey].append(a)
            bdict[zkey].append(b)
    return adict, bdict


def adjust_the_phase_wrap(vectors, thetas):
    output_vectors = vectors.copy()
    output_thetas = thetas.copy()
    extra_elements = 0
    i = 0
    for vector, theta in zip(vectors, thetas):
        diff = np.diff(theta)
        is_there_wrapping = np.abs(diff) > (2 * np.pi) * .9
        if np.any(is_there_wrapping):
            wrapping_idx = np.where(is_there_wrapping == True)[0][0]

            # Part A from wrapping:
            theta_a = theta[:wrapping_idx]
            vector_a = vector[:wrapping_idx]

            # Part B from wrapping:
            theta_b = theta[wrapping_idx + 1:]
            vector_b = vector[wrapping_idx + 1:]

            #
            output_vectors.pop(i + extra_elements)
            output_vectors.insert(i + extra_elements, vector_a)
            output_vectors.insert(i + extra_elements, vector_b)

            #
            output_thetas.pop(i + extra_elements)
            output_thetas.insert(i + extra_elements, theta_a)
            output_thetas.insert(i + extra_elements, theta_b)

            extra_elements += 1
        i += 1
    return output_vectors, output_thetas


def phase_wrap_dict(r_dict, theta_dict):
    new_rdict = r_dict.copy()
    new_thetadict = theta_dict.copy()
    for xkey, zkey in zip(r_dict.keys(), theta_dict.keys()):
        r_list = r_dict[xkey]
        theta_list = theta_dict[xkey]
        new_rdict[xkey], new_thetadict[xkey] = adjust_the_phase_wrap(r_list, theta_list)
    return new_rdict, new_thetadict


def cart2polar_coord_dict(xdict, zdict):
    abs_fun = lambda x, z: np.sqrt(x ** 2 + z ** 2)
    ang_fun = lambda x, z: np.arctan2(x, z)
    r_dict, theta_dict = apply_operation_to_coord_dict(xdict, zdict, abs_fun, ang_fun)
    r_dict, theta_dict = phase_wrap_dict(r_dict, theta_dict)
    return r_dict, theta_dict


def shift_coord_dict(xdict, zdict, new_origin, old_origin=(0, 0)):
    shift = (new_origin[0] - old_origin[0], new_origin[1] - old_origin[1])
    shiftx_fun = lambda x, z: x + shift[0]
    shiftz_fun = lambda x, z: z + shift[1]
    new_xdict, new_zdict = apply_operation_to_coord_dict(xdict, zdict, shiftx_fun, shiftz_fun)
    return new_xdict, new_zdict


def rotate_coord_dict(xdict, zdict, ang_deg):
    ang = np.deg2rad(ang_deg)
    rotatex_fun = lambda x, z: x * np.cos(ang) - z * np.sin(ang)
    rotatez_fun = lambda x, z: x * np.sin(ang) + z * np.cos(ang)
    new_xdict, new_zdict = apply_operation_to_coord_dict(xdict, zdict, rotatex_fun, rotatez_fun)
    return new_xdict, new_zdict


def find_radius_from_ang(ang, r_dict, theta_dict, surf_type):
    radii = r_dict[surf_type]
    thetas = theta_dict[surf_type]
    for radius, theta in zip(radii, thetas):
        if ang == -2.2794017484337536:
            1
        if theta[0] < ang < theta[-1]:
            return np.interp(ang, theta, radius)
        elif theta[-1] < ang < theta[0]:
            return np.interp(ang, theta[::-1], radius[::-1])
    raise ValueError(f"Impossible to find surface point at the given angle.")


def find_multiple_radii_from_ang(ang_deg, r_dict, theta_dict, surf_type):
    if type(ang_deg) == list:
        ang_deg = np.asarray(ang_deg)

    if type(ang_deg) == float or type(ang_deg) == int:
        ang = np.deg2rad(ang_deg)
        radius = find_radius_from_ang(ang, r_dict, theta_dict, surf_type)
        return radius
    elif type(ang_deg) == np.ndarray:
        angs = np.deg2rad(ang_deg)
        radii = np.asarray([
            find_radius_from_ang(ang, r_dict, theta_dict, surf_type) for ang in angs
        ])
        return radii


def find_transition_angles(x, z):
    x0, z0 = x[0], z[0]
    xf, zf = x[-1], z[-1]
    ang_beg = np.arctan2(x0, z0)
    ang_end = np.arctan2(xf, zf)
    return ang_beg, ang_end


def divide_angular_span(ang_span, angs_span):
    output = []
    start = ang_span[0]
    for span in sorted(angs_span):
        if start < span[0]:
            output.append((start, span[0]))
        start = span[1]
    if start < ang_span[1]:
        output.append((start, ang_span[1]))
    return output
