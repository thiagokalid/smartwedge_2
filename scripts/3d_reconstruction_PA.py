# === Third-Party Library Imports ===
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import matplotlib

# === Local Module Imports ===
from framework import file_m2k
from framework.post_proc import envelope, normalize
from tqdm import tqdm

from parameter_estimation.intsurf_estimation import profile_fadmm, img_line
from pipe_lens_imaging.intsurf_estimation import img_line_first_echoes
from pipe_lens_imaging.utils import pointlist_to_cloud as pl2pc
from pipe_lens_imaging.utils import pcd_to_mesh as p2m

# === Matplotlib Configuration ===
matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})
linewidth = 6.3091141732 # LaTeX linewidth


if __name__ == '__main__':
    # Inicializa as variáveis
    positions = [0]  # Encoder
    surftop = []  # Superfície externa
    surfbot = []  # Superfície interna
    pfac1 = []  # Frente
    pfac2 = []  # Fundo
    psid1 = []  # Lado direito
    psid2 = []  # Lado esquerdo
    step = 2 * 0.476
    stepx = 0.2  # Passo entre os pontos nas direções x, y, z
    stepy = 0.2
    stepz = 0.2
    c = 5813
    n_shots = 170

    ystep = 1e-3
    y_span = np.arange(0, n_shots * ystep, ystep)
    pos_idx = 0

    ang_span = np.arange(-45, 45, .2)
    thickness_map = np.zeros(shape=(n_shots, len(ang_span)))  # z-axis -> (thickness, theta)

    # ROI:
    delta_r = .05e-3
    delta_ang = .2
    radius = 140e-3 / 2
    waterpath = 32e-3
    wall_thickness = 17.23e-3 + 5e-3
    r_roi = np.arange(radius - wall_thickness - 10e-3, radius + 10e-3, delta_r)
    ang_roi = np.radians(np.arange(-45, 45, delta_ang))
    n_angs = len(ang_roi)

    # Read thickness map:
    idx_top_m = np.memmap("../test/outer_surface.dat", dtype='int32', mode='r', shape=(n_shots, n_angs))
    idx_bot_m = np.memmap("../test/inner_surface.dat", dtype='int32', mode='r', shape=(n_shots, n_angs))

    z_span = r_roi


    xstep = ystep = zstep = .5e-3
    rstep = .5e-3
    theta_step = .2

    from scipy.interpolate import RegularGridInterpolator

    interp1 = RegularGridInterpolator((y_span, ang_roi), idx_top_m, bounds_error=False, fill_value=None)
    interp2 = RegularGridInterpolator((y_span, ang_roi), idx_bot_m, bounds_error=False, fill_value=None)

    denser_ang_roi = np.arange(ang_roi[0], ang_roi[-1], np.radians(theta_step))
    denser_y_span = np.arange(y_span[0], y_span[-1], ystep)

    # Use 'ij' indexing to match (y, angle) order
    denser_yy, denser_xx = np.meshgrid(denser_y_span, denser_ang_roi, indexing='ij')

    # Stack in (n_points, 2) format with correct dimension order
    points = np.stack([denser_yy.ravel(), denser_xx.ravel()], axis=-1)

    interpolated_idx_top = np.int32(interp1(points).reshape(denser_yy.shape))
    interpolated_idx_bot = np.int32(interp2(points).reshape(denser_yy.shape))

    n_shots_interp = interpolated_idx_bot.shape[0]

    j = 0
    for k in tqdm(range(1, n_shots_interp)):

        idx_top = interpolated_idx_top[k, :]
        idx_bot = interpolated_idx_bot[k, :]

        rtop = z_span[::-1][idx_top]
        rbot = z_span[::-1][idx_bot]

        # Outer and inner surface:
        if denser_y_span[k] in y_span:
            thickness_map[j, :] = rtop - rbot
            j += 1


        y_slice = denser_y_span[k] * np.ones(n_angs)
        xtop, ztop = rtop * np.sin(denser_ang_roi), rtop * np.cos(denser_ang_roi)
        xbot, zbot = rbot * np.sin(denser_ang_roi), rbot * np.cos(denser_ang_roi)

        surftop.extend([(xt, yt, zt) for xt, yt, zt in zip(xtop, y_slice, ztop)])
        surfbot.extend([(xb, yb, zb) for xb, yb, zb in zip(xbot, y_slice, zbot)])

        ## Front Face or Back face
        if k == 1 or k == n_shots_interp - 1:
            idx = 0 if k == 1 else k == n_shots_interp - 1

            for i in range(n_angs):
                r_curr = np.arange(rbot[i], rtop[i] + rstep, rstep)
                theta_curr = np.radians(ang_span[i])

                # Convert to cartesian coordinates:
                xfront, zfront = r_curr * np.sin(theta_curr), r_curr * np.cos(theta_curr)

                pts_list = [(xf, denser_y_span[k], zf) for xf, zf in zip(xfront, zfront)]

                if idx == 1:
                    pfac2.extend(pts_list)
                else:
                    pfac1.extend(pts_list)

        # Side Face
        radius_side1 = np.arange(rbot[0], rtop[0] + rstep, rstep)

        xside1, zside1 = radius_side1 * np.sin(denser_ang_roi[0]), radius_side1 * np.cos(denser_ang_roi[0])

        psid1.extend([(xs1, denser_y_span[k], zs1) for xs1, zs1 in zip(xside1, zside1)])

        # Side 2 face:
        radius_side2 = np.arange(rbot[-1], rtop[-1] + rstep, rstep)

        xside2, zside2 = radius_side2 * np.sin(denser_ang_roi[-1]), radius_side2 * np.cos(denser_ang_roi[-1])

        psid2.extend([(xs2, denser_y_span[k], zs2) for xs2, zs2 in zip(xside2, zside2)])


    #%%

    #%%

    factor = 1

    fig, ax = plt.subplots(figsize=(linewidth*.5 * factor, 3 * factor))
    plt.imshow(thickness_map * 1e3, extent=[ang_span[0], ang_span[-1], y_span[-1] * 1e3 - 15, y_span[0] * 1e3 - 15], cmap="YlGnBu", aspect='auto', interpolation='None', vmin=15, vmax=19)
    plt.colorbar()
    plt.xlabel(r'$\alpha$ / (degrees)')
    plt.ylabel(r'Passive direction / (mm)')
    plt.tight_layout()

    # ytemp = np.arange(y_span[0] - 10e-3, y_span[-1] + 10e-3, 1e-3) * 1e3
    # xtemp = 1/2 * 60/(2 * np.pi * 51.55) * 360 * np.ones_like(ytemp)
    # plt.plot(-xtemp, ytemp, 'r--', linewidth=1)
    # plt.plot(+xtemp, ytemp, 'r--', linewidth=1)

    plt.yticks(np.arange(0, 150, 30))
    plt.xticks(np.linspace(-45, 45, 7))
    plt.grid(alpha=.25)

    plt.ylim([150, 0])
    plt.tight_layout()
    plt.savefig("../figures/corrosion_map_2d_PA.pdf")
    plt.show()

    #
    # #%%
    # pts = [False, False, pfac1, False, psid1, psid2]
    pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

    steps = (stepx, stepy, stepz)
    print(f'Forming Point Cloud with normals')
    pcd = pl2pc(pts, steps, orient_tangent=True, xlen=None, radius_top=20, radius_bot=12)

    o3d.visualization.draw_geometries([pcd], point_show_normal=False)


    # print(f'Generating Mesh')
    mesh = p2m(pcd, depth=10, smooth=False)
    mesh.compute_triangle_normals()

    # %%
    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set up camera parameters
    view = vis.get_view_control()
    camera = view.convert_to_pinhole_camera_parameters()
    # camera.intrinsic.set_intrinsics(
    #     width=640,
    #     height=480,
    #     fx=500,
    #     fy=500,
    #     cx=320,
    #     cy=240
    # )
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera)

    # Display the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)

    # Set rendering options
    camera_render = vis.get_render_option()
    camera_render.set_projection(
        fov=60.0,
        aspect=1.333,
        near=0.1,
        far=1000.0,
        fov_type=o3d.visualization.rendering.Camera.FovType.Vertical
    )

    # Update renderer and run the visualization
    vis.update_renderer()
    vis.run()

    # Destroy the window

    o3d.io.write_triangle_mesh(r'../meshs/half_pipe_RBH_PA.stl', mesh, print_progress=True)


