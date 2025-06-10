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
#%%

def compute_sscan(channels_insp, channels_ref):
    sscan = np.sum((channels_insp - channels_ref), axis=2)
    upper_bound = int(np.min(np.argmax(sscan, axis=0))-100)
    lower_bound = int(upper_bound + 1.1e3)
    normalized_windowed_sscan = normalize(envelope(sscan[upper_bound : lower_bound,:]))
    return normalized_windowed_sscan, upper_bound, lower_bound


def out_surfaces(img, lamb=100, rho=10):
    a = img_line(img)
    z = a[0].astype(int)
    w = np.diag((a[1]))
    idx, resf, kf, pk, sk = profile_fadmm(w, z, lamb=lamb, x0=z, rho=rho, eta=.999, itmax=25, tol=1e-3)
    idx = idx.astype(int)
    return idx

def int_surfaces(img, idx_cut=500, threshold=0.8, height=21e-3, lamb=0.1, rho=0.01):
    sscan_windowed = img[idx_cut:, :]

    a = img_line_first_echoes(
        img=normalize(sscan_windowed),
        threshold=threshold,
        height=height
    )

    z = a[0].astype(int)

    # plt.figure()
    # plt.imshow(img_cut, aspect='auto')
    # plt.plot(z)
    # plt.show()

    # lamb = 0.1
    # rho = 0.01
    w = np.diag((a[1]))
    idx, resf, kf, pk, sk = profile_fadmm(w, z, lamb=lamb, x0=z, rho=rho, eta=.999, itmax=25, tol=1e-3)
    return idx_cut+idx.astype(int)

if __name__ == '__main__':

    data_root = '../data/half_pipe_rbh/'
    filename_insp = data_root + "sweep_xl_1mm_step.m2k"
    data_insp = file_m2k.read(filename_insp, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
    data_ref = file_m2k.read(data_root + "ref.m2k", freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

    #
    channels_insp = data_insp.ascan_data[..., 0]
    channels_ref = data_ref.ascan_data[..., 0]


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
    n_shots = 149

    ystep = 1e-3
    y_span = np.arange(0, n_shots * ystep + ystep, ystep)
    pos_idx = 0

    ang_span = np.arange(-45, 45 + .5, .5)

    img, z0, z1 = compute_sscan(channels_insp, channels_ref)
    x = np.linspace(-45, 45, 181)
    h = np.max(img[800::, :], axis=0)
    curva_h = (h/h.max())

    thickness_map = np.zeros(shape=(n_shots, len(ang_span))) # z-axis -> (thickness, theta)

    generateVideo = False

    for k in tqdm(range(1, n_shots)):
        data_insp = file_m2k.read(filename_insp, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=k)
        channels_insp = data_insp.ascan_data[..., 0]

        img, z0, z1 = compute_sscan(channels_insp, channels_ref)

        idx_top = out_surfaces(img, lamb=100, rho=10)
        if k<=40:
            curva = np.cos(np.deg2rad(x / 0.38))
            curva[np.argwhere(curva <= 0.25)] = 0.25
            idx_bot = int_surfaces(img, idx_cut=600, threshold=0.9, height=0.35*curva, lamb=0.1, rho=0.01)
        else:
            curva = np.cos(np.deg2rad(x / 0.38))
            curva[np.argwhere(curva <= 0.2)] = 0.17
            idx_bot = int_surfaces(img, idx_cut=600, threshold=0.8, height=0.07*curva, lamb=0.1, rho=0.01)

        # plt.imshow(20 * np.log10(img + 1e-6), aspect='auto')
        # plt.plot(idx_top, color='r')
        # plt.plot(idx_bot, color='k')
        # plt.title(f"shot {k}")
        # plt.show()

        z_span = (c * (data_insp.time_grid[z0:z1]) / 2000)[:,0]

        top_plain = -z_span[idx_top]
        delta_cw = np.mean(top_plain)-70
        z_span = -z_span - delta_cw
        top_plain = z_span[idx_top]

        top = top_plain * np.cos(np.deg2rad(ang_span))
        surfx_top = top_plain * np.sin(np.deg2rad(ang_span))

        bot_plain = z_span[idx_bot]

        thickness_map[k, :] = top_plain - bot_plain

        # Interpolação das falhas - eixo z
        dist = bot_plain - np.roll(bot_plain, 1)
        dist_ = np.copy(dist)
        flaw = []
        for p in range(1, len(dist)):
            signal = np.sign(dist[p])
            if abs(dist[p]) >= 0.5:
                # print(p)
                flag = True
                np_y = int((abs(bot_plain[p - 1] - bot_plain[p]) - 0.2) / (stepy))
                if bot_plain[p - 1] <= bot_plain[p]:
                    flaw_aux = np.linspace(bot_plain[p - 1] - stepy, bot_plain[p] + stepy, np_y)
                else:
                    flaw_aux = np.linspace(bot_plain[p - 1] + stepy, bot_plain[p] - stepy, np_y)
                surfbot.extend([(flaw_aux[i]*np.sin(np.deg2rad(ang_span[p])), k, flaw_aux[i]*np.cos(np.deg2rad(ang_span[p]))) for i in range(np_y)])
                flaw.append([(ang_span[p], k, flaw_aux[i]) for i in range(np_y)])

        bot = bot_plain * np.cos(np.deg2rad(ang_span))
        surfx_bot = bot_plain * np.sin(np.deg2rad(ang_span))


        if k > 0:
            bot_ant = bot
            top_ant = top
            k_ant = k
            flaw_aux_ant = flaw_aux

        # XY Plane - aloca os pontos da superfície externa no formato [x, y, z]
        surftop.extend([(surfx_top[i], k, top[i]) for i in range(len(surfx_top))])
        # Plano XY - aloca os pontos da superfície interna no formato [x, y, z]
        surfbot.extend([(surfx_bot[i], k, bot[i]) for i in range(len(surfx_bot))])

        # XZ Plane - builds the wall on the left
        n_pts = 30
        aux_esq = np.linspace(surfx_top[0], surfx_bot[0], n_pts)
        psid1.extend([(aux_esq[i], k, j) for i, j in enumerate(np.linspace(top[0] + stepz, bot[0] - stepz, n_pts))])
        # XZ Plane - builds the wall on the right
        aux_dir = np.linspace(surfx_top[-1], surfx_bot[-1], n_pts)
        psid2.extend(
            [(aux_dir[i], k, j) for i, j in enumerate(np.linspace(top[-1] + stepz, bot[-1] - stepz, n_pts))])


        # Passive direction interpolation
        if k > 1:
            n_points = 3
            x_aux = np.linspace(k_ant + stepx, k - stepx, n_points)
            top_aux = np.linspace(top_ant, top, n_points)
            bot_aux = np.linspace(bot_ant, bot, n_points)
            for l in range(n_points):
                # Plano XY
                surftop.extend([(surfx_top[i], x_aux[l], top_aux[l, i]) for i in range(len(surfx_top))])
                surfbot.extend([(surfx_bot[i], x_aux[l], bot_aux[l, i]) for i in range(len(surfx_bot))])

        ## Front Face
        if k == 1:
            for l in range(1, len(ang_span) - 1):
                aux_front = np.linspace(surfx_top[l], surfx_bot[l], n_pts)
                pfac1.extend(
                    [(aux_front[i], k, j) for i, j in
                     enumerate(np.linspace(top[l] - stepz, bot[l] + stepz, n_pts))])
                pfac1.extend(
                    [(aux_front[i], k-0.2, j) for i, j in
                     enumerate(np.linspace(top[l] - stepz, bot[l] + stepz, n_pts))])
        ## Back Face
        elif k == n_shots - 1:
            for l in range(1, len(ang_span) - 1):
                aux_back = np.linspace(surfx_top[l], surfx_bot[l], n_pts)
                pfac2.extend(
                    [(aux_back[i], k, j) for i, j in enumerate(np.linspace(top[l] + stepz, bot[l] - stepz, n_pts))])
                pfac2.extend(
                    [(aux_back[i], k-0.2, j) for i, j in enumerate(np.linspace(top[l] + stepz, bot[l] - stepz, n_pts))])


        if generateVideo:
            plt.figure(figsize=(8, 6))
            plt.title(f"Normalized S-scan (log-scale) of a passive \n diretion sweep with acoustic lens. $y={y_span[k] * 1e3:.2f}$ mm.")
            plt.pcolormesh(ang_span, z_span, np.log10(img + 1e-6), cmap='inferno', vmin=-5, vmax=0)
            plt.plot(ang_span, top_plain, 'w-', linewidth=2)
            plt.plot(ang_span, bot_plain, 'w-', linewidth=2)

            plt.ylabel("Radial direction / (mm)")
            plt.xlabel(rf"$\alpha$-axis / (degrees)")
            plt.colorbar()
            plt.grid(alpha=.5)
            plt.ylim([48, 72])
            plt.xticks(np.arange(-45, 45 + 15, 15))
            plt.yticks(np.arange(50, 70 + 5, 5))
            plt.tight_layout()
            plt.savefig(f"../figures/lens_frames_surf/file{k:02d}.png")
            plt.close()

    pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

    #%%

    factor = 1

    fig, ax = plt.subplots(figsize=(linewidth*.5 * factor, 3 * factor))
    plt.imshow(thickness_map, extent=[ang_span[0], ang_span[-1], y_span[-1] * 1e3, y_span[0] * 1e3], cmap="YlGnBu", aspect='auto', interpolation='None', vmin=15, vmax=19)
    plt.colorbar()
    plt.xlabel(r'$\alpha$ / (degrees)')
    plt.ylabel(r'Passive direction / (mm)')

    ytemp = np.arange(y_span[0] - 10e-3, y_span[-1] + 10e-3, 1e-3) * 1e3
    xtemp = 1 / 2 * 60 / (2 * np.pi * 51.55) * 360 * np.ones_like(ytemp)
    # plt.plot(-xtemp, ytemp, 'r--', linewidth=1)
    # plt.plot(+xtemp, ytemp, 'r--', linewidth=1)

    plt.ylim([y_span[-1] * 1e3, y_span[0] * 1e3])
    plt.yticks(np.arange(0, 150, 30))
    plt.xticks(np.linspace(-45, 45, 7))

    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig("../figures/corrosion_map_2d_LENS.pdf")
    # plt.show()



    #%%

    # theta = np.linspace(-np.pi/4, np.pi/4, 181)
    # zz = np.linspace(0, n_shots * 1e-3, n_shots)
    # theta, zz = np.meshgrid(theta, zz)
    # xx = 70e-3 * np.cos(theta)
    # yy = 70e-3 * np.sin(theta)
    #
    # fig, ax = plt.subplots(figsize=(linewidth*.5 * factor, 3 * factor), subplot_kw={"projection": "3d"})
    # ax.view_init(elev=25, azim=-160, roll=0)
    #
    # cmap_thickness = thickness_map / 19
    # cmap = plt.cm.YlGnBu(cmap_thickness)
    # surf = ax.plot_surface(xx * 1e3, yy * 1e3, zz * 1e3,  rstride=1, cstride=1, facecolors=cmap, shade=False)
    #
    # ax.set_aspect('equal')
    # ax.set_xlabel(r'$x-axis / (\mathrm{mm})$')
    # ax.set_ylabel(r'$z-axis / (\mathrm{mm})$')
    # ax.set_zlabel(r'$y-axis / (\mathrm{mm})$')
    #
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.tight_layout()
    # plt.show()

    #%%

    steps = (stepx, stepy, stepz)
    print(f'Forming Point Cloud with normals')
    pcd = pl2pc(pts, steps, orient_tangent=True, xlen=None, radius_top=20, radius_bot=12)
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)


    # print(f'Generating Mesh')
    mesh = p2m(pcd, depth=10, smooth=False)
    mesh.compute_triangle_normals()

    #%%
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

    #o3d.io.write_triangle_mesh(r'../meshs/half_pipe_RBH.stl', mesh, print_progress=True)
