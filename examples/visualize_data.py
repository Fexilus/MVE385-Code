"""Visualize the data."""
import numpy as np
import open3d as o3d

from tracking.data.load import load_point_clouds


data_files = ["data/data_109.h5",
              "data/data_130.h5",
              "data/data_142.h5",
              "data/data_143.h5"]

# Set a world space bounding box for the point clouds
bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.asarray((-50, -50, -50)),
                                                   np.asarray(( 50,  50,  50)))


def normalize(vector):
    """Normalize a vector, i.e. make the norm 1 while preserving direction."""
    return vector / np.linalg.norm(vector)


pclouds = [load_point_clouds(file) for file in data_files]

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=800, height=600)

# Add point cloud objects to be used in render
geometries = [next(pcloud).crop(bounding_box) for pcloud in pclouds]
for geometry in geometries:
    visualizer.add_geometry(geometry)

# Set default camera view
front       = np.asarray((3.0, 4.0, -1.0))
lookat      = normalize(np.asarray((0.1, 0.0, 0.0)))
world_up    = np.asarray((0.0, 0.0, -1.0))
camera_side = normalize(np.cross(front, world_up))
camera_up   = np.cross(camera_side, front)
zoom        = 0.25

view_control = visualizer.get_view_control()
view_control.set_front(front)
view_control.set_lookat(lookat)
view_control.set_up(camera_up)
view_control.set_zoom(zoom)
view_control.translate(150, 0)

for i in range(800):
    # Loop over 800 frames
    for pcloud, geometry in zip(pclouds, geometries):
        cur_pcloud = next(pcloud).crop(bounding_box)
        geometry.points = cur_pcloud.points
        geometry.colors = cur_pcloud.colors
        visualizer.update_geometry(geometry)

    visualizer.poll_events()
    visualizer.update_renderer()
