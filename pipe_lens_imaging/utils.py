import numpy as np
import open3d as o3d

from pipe_lens.imaging_utils import convert_corners_time2idx, moving_average

__all__ = [
    'api_func',
    'api_func_polar',
    'api',
    'fwhm',
    'pointlist_to_cloud',
    'pcd_to_mesh',
    'get_class_by_attribute',
]

def api_func(img, t, ang, corner_in_time, thresh=.5, drawSquare=True):
    corners = convert_corners_time2idx(t, ang, corner_in_time)
    north_east_corner = corners[0]
    south_west_corner = corners[1]
    img_cropped = img[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]
    local_max = np.max(img_cropped)
    maxLocationCoord = np.where(img_cropped == local_max)
    maxLocation = maxLocationCoord[1] + north_east_corner[1]
    img_cropped_masked = img_cropped > thresh * local_max
    img_masked = np.zeros_like(img)
    img_masked[north_east_corner[0]:south_west_corner[0],
    north_east_corner[1]:south_west_corner[1]] += img_cropped_masked
    api = np.sum(img_masked * 1.0) / len(img_masked)

    if drawSquare:
        width = 1
        scale_factor = int(img.shape[0] / img.shape[1])
        img_masked[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: north_east_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        south_west_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor: north_east_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[south_west_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxLocation, img_masked, local_max


def api_func_polar(sscan, r_span, theta_deg_span, corners_in, thresh=-6, drawSquare=True):
    # Corner in [(r1, theta1), (r2, theta2)] in (mm, deg)

    # Converts (mm, deg) to (index, index):
    corners = convert_corners_time2idx(r_span, theta_deg_span, corners_in)  # Corner in index
    north_east_corner = corners[0]
    south_west_corner = corners[1]

    # Crops only the API Region of Interest (ROI):
    img_cropped = sscan[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]

    # Obtain the maximum amplitude within the ROI:
    maximumAmplitude = np.max(img_cropped)

    # Creates a binary image where only pixels above thesh_db are highlighted
    pixels_above_threshold_within_roi = img_cropped >= maximumAmplitude * 10 ** (thresh / 20)
    pixels_above_threshold = np.zeros_like(sscan, dtype=bool)
    pixels_above_threshold[north_east_corner[0]:south_west_corner[0],
    north_east_corner[1]:south_west_corner[1]] += pixels_above_threshold_within_roi

    if np.any(pixels_above_threshold == True):
        maxLocationWithinROI = np.where(img_cropped == maximumAmplitude)
        maxAbsoluteLocation = maxLocationWithinROI[1] + north_east_corner[1]
    else:
        maxAbsoluteLocation = None

    # Compute the area of each pixel considering:
    # * All pixels above threshold belongs to the tube wall;
    # * The tube center is aligned with (0, 0) of the coordinate system;
    # * The sampling is constant along radius and angle axis;
    r_borders = moving_average(r_span)
    diff_i = r_borders[0] - r_span[0]
    diff_f = r_span[-1] - r_borders[-1]
    r_borders = np.append(r_span[0] - diff_i, r_borders)
    r_borders = np.append(r_borders, r_span[-1] + diff_f)

    #
    theta_borders = moving_average(theta_deg_span)
    diff_i = theta_borders[0] - theta_deg_span[0]
    diff_f = theta_deg_span[-1] - theta_borders[-1]
    theta_borders = np.append(theta_deg_span[0] - diff_i, theta_borders)
    theta_borders = np.append(theta_borders, theta_deg_span[-1] + diff_f)

    pixels_area = np.zeros_like(sscan)
    pixels_coords = np.zeros(shape=(*sscan.shape, 2))
    for i in range(sscan.shape[0]):
        for j in range(sscan.shape[1]):
            inner_radius = r_borders[i]
            outer_radius = r_borders[i + 1]
            theta1 = theta_borders[j]
            theta2 = theta_borders[j + 1]
            diff_theta = np.abs(theta2 - theta1)

            #
            pixels_area[i, j] = (outer_radius * 2 - inner_radius * 2) * np.pi * diff_theta / 360

            pixel_radius = (outer_radius + inner_radius) / 2
            pixel_ang = (theta1 + theta2) / 2
            pixels_coords[i, j, 0] = pixel_radius * np.cos(pixel_ang * np.pi / 360)
            pixels_coords[i, j, 1] = pixel_radius * np.sin(pixel_ang * np.pi / 360)
    pixels_area = np.abs(pixels_area)
    #
    api = np.sum(pixels_area[pixels_above_threshold])
    estimated_width = np.max(pixels_coords[pixels_above_threshold, 0]) - np.min(pixels_coords[pixels_above_threshold, 0])
    estimated_height = np.max(pixels_coords[pixels_above_threshold, 1]) - np.min(pixels_coords[pixels_above_threshold, 1])

    if drawSquare:
        width = 2
        scale_factor = int(sscan.shape[0] / sscan.shape[1])
        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: north_east_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        south_west_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: north_east_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[south_west_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxAbsoluteLocation, pixels_above_threshold, maximumAmplitude, estimated_width, estimated_height

def api(img, xaxis, zaxis, corners, thresh=-6, drawSquare=False):
    #corners = (superior-esquerdo, inferior-direito) considerando a imagem na visualizaçaõ eixo z para baixo e x para direita
    NE_corner, SW_corner = corners

    if (NE_corner[0] >= SW_corner[0] or NE_corner[1] <= SW_corner[0]):
        raise ValueError("Invalid Corners")

    xmin, xmax = NE_corner[0], SW_corner[0]
    zmin, zmax = NE_corner[1], SW_corner[1]

    i0, i1 = np.argmin(np.abs(xaxis - xmin)), np.argmin(np.abs(xaxis - xmax))
    j0, j1 = np.argmin(np.abs(zaxis - zmin)), np.argmin(np.abs(zaxis - zmax))

    pixel_area = (xaxis[1] - xaxis[0]) * (zaxis[1] - zaxis[0])

    croped_region = img[j0 : j1 + 1, i0 : i1 + 1]
    local_maximum = np.max(croped_region)
    binary_mask = croped_region >= local_maximum * 10 ** (thresh / 20)
    api = np.sum(binary_mask) * pixel_area

    return api, local_maximum, binary_mask

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def fwhm(signal, xspan=None):
    # Compute the Full Width Half Maximum assuming only peak exist within the signal.
    # If xspan is not given, FWHM will be computed in number of samples.

    if xspan is None:
        xspan = np.arange(len(signal))

    return half_max_x(xspan, signal)



def pointlist_to_cloud(points, steps, orient_tangent=False, xlen=None, radius_top=10, radius_bot=10):
    stepx, stepy, stepz = steps
    step = np.abs(steps).mean()
    surftop, surfbot, pfac1, pfac2, psid1, psid2 = points

    # Cria objeto PointCloud
    pcdtop = o3d.geometry.PointCloud()
    if surftop:
        # Transforma a lista de pontos em uma estrutura de pontos em 3D
        pcdtop.points = o3d.utility.Vector3dVector(surftop)
        # Usa uma busca em arvore com determinado raio para estimar as normais. Essas normais não podem ser muito dife-
        # rentes, como num spool, para dar problema.
        pcdtop.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_top, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdtop.orient_normals_consistent_tangent_plane(k=20)
            # pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))

        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdtop.orient_normals_to_align_with_direction()
            a = np.asarray(pcdtop.normals)
            coef = np.sign(a[:, 1])[:, np.newaxis]
            a *= - coef
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdtop.normals = o3d.utility.Vector3dVector(a * -coef)

        else:  # Top deve ter as normais invertidas para apontar para cima
            pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))
        pcdtop.orient_normals_to_align_with_direction(np.array([0, 0, 1]))

    pcdbot = o3d.geometry.PointCloud()
    if surfbot:
        pcdbot.points = o3d.utility.Vector3dVector(surfbot)
        pcdbot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_bot, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdbot.orient_normals_consistent_tangent_plane(20)
            pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdbot.orient_normals_to_align_with_direction()
            a = np.asarray(pcdbot.normals)
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdbot.normals = o3d.utility.Vector3dVector(a * coef)
            pcdbot.orient_normals_to_align_with_direction()
            pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
        else:
            pcdbot.normals = o3d.utility.Vector3dVector((np.asarray(pcdbot.normals)))

        pcdbot.orient_normals_to_align_with_direction(np.array([0, 0, -1]))

    pcdf1 = o3d.geometry.PointCloud()
    if pfac1:
        pcdf1.points = o3d.utility.Vector3dVector(pfac1)
        pcdf1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=20))
        # pcdf1.orient_normals_to_align_with_direction()
        # pcdf1.normals = o3d.utility.Vector3dVector(-np.asarray(pcdf1.normals)) # Face frontal deve apontar para fora da tela
        pcdf1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, -1, 0])),
                                                           [np.asarray(pcdf1.normals).shape[0], 1]))
        pcdf1.orient_normals_to_align_with_direction(np.array([0, -1, 0]))


    pcdf2 = o3d.geometry.PointCloud()
    if pfac2:
        pcdf2.points = o3d.utility.Vector3dVector(pfac2)
        pcdf2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=20))
        # pcdf2.orient_normals_to_align_with_direction()
        # pcdf2.normals = o3d.utility.Vector3dVector(np.asarray(pcdf2.normals)) # Face traseira deve apontar para dentro da tela
        pcdf2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 1, 0])),
                                                           [np.asarray(pcdf2.normals).shape[0], 1]))
        pcdf2.orient_normals_to_align_with_direction(np.array([0, 1, 0]))


    pcds1 = o3d.geometry.PointCloud()
    if psid1:
        pcds1.points = o3d.utility.Vector3dVector(psid1)
        pcds1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),
                                                           [np.asarray(pcds1.normals).shape[0], 1]))
        # Face lateral esquerda deve apontar para esquerda

    pcds2 = o3d.geometry.PointCloud()
    if psid2:
        pcds2.points = o3d.utility.Vector3dVector(psid2)
        pcds2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),
                                                           [np.asarray(pcds2.normals).shape[0], 1]))
        # Face lateral direita deve apontar para direita, junto do eixo.

    # Soma as nuvens de pontos em uma única nuvem
    return pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2


def pcd_to_mesh(pcd, depth=7, scale=1.1, smooth=0):
    print('Meshing')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=True)[0]
    mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    if smooth:
        mesh = mesh.filter_smooth_simple(smooth)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    print(f'Generated mesh with {len(mesh.triangles)} triangles')
    return mesh

def get_class_by_attribute(models, attribute, class_type, attribute_name="nicknames", class_name="transducer"):
    for model in models:
        if attribute in model[attribute_name]:
            return class_type(*model["attributes"].values())
    raise TypeError(f"There is no {class_name} with the given {attribute_name}.")