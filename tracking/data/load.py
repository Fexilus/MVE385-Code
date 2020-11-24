"""Load HDF5 data into open3d data structures."""
import itertools

import h5py
import numpy as np
import open3d as o3d
import seaborn as sns


# Set up a color palettes as iterables
seaborn_palette = itertools.cycle(sns.color_palette())
no_palette = itertools.cycle([(1.0, 1.0, 1.0)])


def load_point_cloud(hdf5file, frame=0, palette=seaborn_palette):
    """Extract a single frame's point cloud from camera data.
    The point cloud will be colored by the palette to differentiate between
    clouds."""

    camera = h5py.File(hdf5file, 'r')

    cam2world = np.asarray(camera["/TMatrixCamToWorld"])

    # Transform coordinates to world space
    cs_points = np.reshape(np.asarray(camera["Sequence"][str(frame)]["Points"]), (-1, 3))
    cs_points_4 = np.ones((int(np.size(cs_points) / 3), 4))
    cs_points_4[:,0:3] = cs_points
    ws_points_4 = (np.matmul(cam2world, (cs_points_4).T)).T
    ws_points = ws_points_4[:,0:3]

    # Color the points with image data, tinted by color from palette
    greyscale = np.reshape(np.asarray(camera["Sequence"][str(frame)]["Image"]) / 255,
                           (-1, 1))
    colors = np.reshape(np.stack((greyscale, greyscale, greyscale), axis=2),
                        (-1, 3))
    colors_painted = colors * np.asarray(next(palette))

    # Add data to open3d object
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(ws_points)
    pcloud.colors = o3d.utility.Vector3dVector(colors_painted)

    return pcloud