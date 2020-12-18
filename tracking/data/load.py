"""Load HDF5 data into open3d data structures."""
import itertools
from collections import defaultdict

import h5py
import numpy as np
import open3d as o3d
import seaborn as sns


# Set up a color palettes as iterables
point_cloud_palette = itertools.cycle(sns.color_palette("pastel"))
track_palette = itertools.cycle(sns.color_palette("bright"))
no_palette = itertools.cycle([(1.0, 1.0, 1.0)])


def load_point_clouds(hdf5file, start_frame=0, palette=point_cloud_palette, min_security=0):
    """Generator over frames loading point cloud hdf5 data into open3d.
    The point cloud will be colored by the palette to differentiate between
    clouds."""

    camera = h5py.File(hdf5file, 'r')

    cam2world = np.asarray(camera["/TMatrixCamToWorld"])

    frames = itertools.cycle(range(start_frame, len(camera["Sequence"])))

    dataset_color = np.asarray(next(palette))

    num_points = camera["Sequence"]["0"]["Image"].size

    for frame in frames:
        camera_frame = camera["Sequence"][str(frame)]

        # Load
        cs_points_raw = camera_frame["Points"]
        greyscale_raw = np.asarray(camera_frame["Image"])

        if min_security:
            # Filter points based on security parameter
            sec_filter = np.asarray(camera_frame["Security"]) >= min_security

            cs_points_raw = np.asarray(cs_points_raw)[sec_filter,:]
            greyscale_raw = greyscale_raw[sec_filter]

            # Update point amount
            num_points = int(cs_points_raw.size / 3)
        
        # Transform coordinates to world space
        cs_points_4 = np.ones((num_points, 4))
        cs_points_4[:,0:3] = np.reshape(cs_points_raw, (-1, 3))
        ws_points_4 = (np.matmul(cam2world, cs_points_4.T)).T

        # Color the points with image data, tinted by color from palette
        greyscale = np.reshape(greyscale_raw / 255, (-1, 1))
        colors = np.reshape(np.stack((greyscale, greyscale, greyscale), axis=2),
                            (-1, 3))
        colors_painted = colors * dataset_color

        # Add data to open3d object
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(ws_points_4[:,0:3])
        pcloud.colors = o3d.utility.Vector3dVector(colors_painted)

        yield pcloud


def load_detections(hdf5file, start_frame=0):
    """Generator over frames loading detections as open3d line sets."""

    camera = h5py.File(hdf5file, 'r')

    frames = itertools.cycle(range(start_frame, len(camera["Sequence"])))

    for frame in frames:
        detections = camera["Sequence"][str(frame)]["Detections"]

        lines = []
        for detection in detections:
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=detection["Length"],
                                                            height=detection["Width"],
                                                            depth=detection["Height"])

            # Adjust for center of mass vs corner definition of position
            mesh_box.translate((-detection["Length"] / 2,
                                -detection["Width"] / 2,
                                -detection["Height"] / 2))
            # Move box to correct position
            rotation = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle((0.0, 0.0, detection["Angle"]))
            mesh_box.rotate(rotation, (0.0, 0.0, 0.0))
            mesh_box.translate(np.ndarray(shape=(3,1), dtype=np.float64,
                                          buffer=detection["Pos"]))

            line_box = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_box)
            lines += [line_box]

        yield lines


def load_tracks(hdf5file, start_frame=0, palette=track_palette):
    """Generator over frames loading reference tracks."""

    camera = h5py.File(hdf5file, 'r')

    frames = itertools.cycle(range(start_frame, len(camera["Sequence"])))

    track_points = defaultdict(list)

    # First pass: load tracks
    for frame in range(start_frame, len(camera["Sequence"])):
        camera_frame = camera["Sequence"][str(frame)]

        for track in camera_frame["Tracks"]:
            track_points[track["TrackId"]] += [track["Pos"]]
    
    # Create open3d line sets for each track
    track_line_sets = {}

    for track_id, points in track_points.items():
        line_set = o3d.geometry.LineSet()

        points_struct = np.asarray(points)
        points_regular = points_struct.view(np.float64).reshape(points_struct.shape + (-1,))
        line_set.points = o3d.utility.Vector3dVector(points_regular)

        line_indices = np.column_stack((range(0, len(points) - 1),
                                        range(1, len(points))))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices))

        line_set.paint_uniform_color(np.asarray(next(palette)))

        track_line_sets[track_id] = line_set


    # Second pass: yield paths of detected objects
    for frame in frames:
        tracks = camera["Sequence"][str(frame)]["Tracks"]

        #TODO: Rewriting tracks is a bit ineffective
        cur_tracks = []

        for track_id in tracks["TrackId"]:
            cur_tracks += [track_line_sets[track_id]]

        yield cur_tracks
