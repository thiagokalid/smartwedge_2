import open3d as o3d
import numpy as np

# Load mesh from ../meshs/half_pipe_RBH.stl
mesh = o3d.io.read_triangle_mesh("../meshs/half_pipe_RBH.stl")
mesh.compute_vertex_normals()  # Ensure normals are computed

angle_rad = np.radians(195)  # Convert degrees to radians
mesh.rotate(R=np.array([
    [1, 0, 0],
    [0, np.cos(angle_rad), -np.sin(angle_rad)],
    [0, np.sin(angle_rad), np.cos(angle_rad)]
]), center=(0, 0, 0))

# Initialize Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Mesh Viewer with 180° Rotation", width=250 * 3, height=400 * 3)
vis.add_geometry(mesh)

# Set up camera parameters
view = vis.get_view_control()
camera = view.convert_to_pinhole_camera_parameters()
# Set intrinsic parameters to more reasonable values
camera.intrinsic.set_intrinsics(
    width=250,  # Match the window width
    height=400, # Match the window height
    fx=200,    # Focal length close to window size
    fy=200,    # Focal length close to window size
    cx=125,    # Center x (width / 2)
    cy=200     # Center y (height / 2)
)

# Set render options
render_option = vis.get_render_option()
render_option.light_on = True
mesh.paint_uniform_color([.25, .25, .25])

# Set the camera field of view (FOV) or zoom level
view.set_zoom(.65)

# Update renderer and run the visualization
vis.update_renderer()
vis.run()

# Capture the visualization as an image
image = vis.capture_screen_float_buffer()

# Save the image as a PNG to the specified directory
o3d.io.write_image("../figures/3d_reconstruction_halfpipe_rbh.png", o3d.geometry.Image(np.asarray(image)))

# Clean up
vis.destroy_window()
