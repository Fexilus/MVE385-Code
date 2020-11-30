"""Visualize the data."""
import numpy as np
import open3d as o3d

from tracking.data.load import load_point_clouds, load_detections, load_tracks


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


pclouds = [load_point_clouds(file, min_security=5) for file in data_files]
camera_detections = [load_detections(file) for file in data_files]
camera_tracks = [load_tracks(file) for file in data_files]

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=800, height=600)

# Add point cloud objects to be used in render
pc_geometries = [next(pcloud).crop(bounding_box) for pcloud in pclouds]
for geometry in pc_geometries:
    visualizer.add_geometry(geometry)

# Add detection objects to be used in render
det_geometries = [next(detections) for detections in camera_detections]
for geometries in det_geometries:
    for geometry in geometries:
        visualizer.add_geometry(geometry)

# Add original track objects to be used in render
tr_geometries = [next(tracks) for tracks in camera_tracks]
for geometries in tr_geometries:
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

# Set default render options
render_options = visualizer.get_render_option()
render_options.line_width = 15.0
render_options.point_size = 2.0

for i in range(800):
    # Update point clouds
    for pcloud, geometry in zip(pclouds, pc_geometries):
        cur_pcloud = next(pcloud).crop(bounding_box)
        geometry.points = cur_pcloud.points
        geometry.colors = cur_pcloud.colors
        visualizer.update_geometry(geometry)

    # Remove old detections
    for geometries in det_geometries:
        for geometry in geometries:
            visualizer.remove_geometry(geometry, False)

    det_geometries = []

    # Add new detections
    for detections in camera_detections:
        cur_detections = next(detections)
        for detection in cur_detections:
            visualizer.add_geometry(detection, False)

        det_geometries += [cur_detections]
    
    # Remove old tracks
    for geometries in tr_geometries:
        for geometry in geometries:
            visualizer.remove_geometry(geometry, False)

    tr_geometries = []

    # Add new detections
    for tracks in camera_tracks:
        cur_tracks = next(tracks)
        for track in cur_tracks:
            visualizer.add_geometry(track, False)

        tr_geometries += [cur_tracks]

    # Update frame
    visualizer.poll_events()
    visualizer.update_renderer()
