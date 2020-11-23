"""Visualize the data from a specific frame."""
import h5py
import numpy as np
import open3d as o3d
import seaborn as sns

import itertools


# Set up a color palette as an iterable
palette = itertools.cycle(sns.color_palette())

# Import camera data
camera0 = h5py.File('data/data_109.h5', 'r')
camera1 = h5py.File('data/data_130.h5', 'r')
camera2 = h5py.File('data/data_142.h5', 'r')
camera3 = h5py.File('data/data_143.h5', 'r')

# Set a world space bounding box for the point clouds
bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.asarray((-50, -50, -50)),
                                                   np.asarray(( 50,  50,  50)))


def get_point_cloud(camera):
    """Extract a single frame's point cloud from the camera data.
    The point cloud will be colored by the palette to differentiate between
    clouds."""

    cam2world = np.asarray(camera["/TMatrixCamToWorld"])

    # Transform coordinates to world space
    cs_points = np.reshape(np.asarray(camera["/Sequence/100/Points"]), (-1, 3))
    cs_points_4 = np.ones((int(np.size(cs_points) / 3), 4))
    cs_points_4[:,0:3] = cs_points
    ws_points_4 = (np.matmul(cam2world, (cs_points_4).T)).T
    ws_points = ws_points_4[:,0:3]

    # Color the points with image data, tinted by color from palette
    greyscale = np.reshape(np.asarray(camera["/Sequence/100/Image"]) / 255,
                           (-1, 1))
    colors = np.reshape(np.stack((greyscale, greyscale, greyscale), axis=2),
                        (-1, 3))
    colors_painted = colors * np.asarray(next(palette))

    # Add data to open3d object
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(ws_points)
    pcloud.colors = o3d.utility.Vector3dVector(colors_painted)

    pcloud = pcloud.crop(bounding_box)

    return pcloud


def normalize(vector):
    """Normalize a vector, i.e. make the norm 1 while preserving direction."""
    return vector / np.linalg.norm(vector)


pc0 = get_point_cloud(camera0)
pc1 = get_point_cloud(camera1)
pc2 = get_point_cloud(camera2)
pc3 = get_point_cloud(camera3)

# Set default camera view
front       = np.asarray((2.0, 4.0, -1.0))
lookat      = normalize(np.asarray((0.1, 0.0, 0.0)))
world_up    = np.asarray((0.0, 0.0, -1.0))
camera_side = normalize(np.cross(front, world_up))
camera_up   = np.cross(camera_side, front)
zoom        = 0.25

o3d.visualization.draw_geometries([pc0, pc1, pc2, pc3],
                                  width=800, height=600,
                                  front=front, lookat=lookat, up=camera_up,
                                  zoom=zoom)
