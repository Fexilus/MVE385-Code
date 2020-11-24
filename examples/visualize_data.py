"""Visualize the data from a specific frame."""
import numpy as np
import open3d as o3d

from tracking.data.load import load_point_cloud


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


pclouds = [load_point_cloud(file).crop(bounding_box) for file in data_files]

# Set default camera view
front       = np.asarray((2.0, 4.0, -1.0))
lookat      = normalize(np.asarray((0.1, 0.0, 0.0)))
world_up    = np.asarray((0.0, 0.0, -1.0))
camera_side = normalize(np.cross(front, world_up))
camera_up   = np.cross(camera_side, front)
zoom        = 0.25

o3d.visualization.draw_geometries(pclouds,
                                  width=800, height=600,
                                  front=front, lookat=lookat, up=camera_up,
                                  zoom=zoom)
