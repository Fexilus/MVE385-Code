"""Load HDF5 data into open3d data structures."""
import itertools

import h5py
import numpy as np
import open3d as o3d
import seaborn as sns


# Set up a color palettes as iterables
seaborn_palette = itertools.cycle(sns.color_palette())
no_palette = itertools.cycle([(1.0, 1.0, 1.0)])


def load_point_clouds(hdf5file, start_frame=0, palette=seaborn_palette):
    """Generator over point clouds per frame from camera data.
    The point cloud will be colored by the palette to differentiate between
    clouds."""

    camera = h5py.File(hdf5file, 'r')

    cam2world = np.asarray(camera["/TMatrixCamToWorld"])

    frames = itertools.cycle(range(start_frame, len(camera["Sequence"])))

    dataset_color = np.asarray(next(palette))

    num_points = camera["Sequence"]["0"]["Image"].size

    for frame in frames:
        # Transform coordinates to world space
        cs_points_4 = np.ones((num_points, 4))
        cs_points_4[:,0:3] = np.reshape(np.asarray(camera["Sequence"][str(frame)]["Points"]), (-1, 3))
        ws_points_4 = (np.matmul(cam2world, cs_points_4.T)).T

        # Color the points with image data, tinted by color from palette
        greyscale = np.reshape(np.asarray(camera["Sequence"][str(frame)]["Image"]) / 255,
                            (-1, 1))
        colors = np.reshape(np.stack((greyscale, greyscale, greyscale), axis=2),
                            (-1, 3))
        colors_painted = colors * dataset_color

        # Add data to open3d object
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(ws_points_4[:,0:3])
        pcloud.colors = o3d.utility.Vector3dVector(colors_painted)

        yield pcloud
