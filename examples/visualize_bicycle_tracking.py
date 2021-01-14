"""Visualize created tracks."""
from math import pi
import itertools

import numpy as np
import open3d as o3d
import h5py
import seaborn as sns

from tracking.data.load import load_point_clouds, load_detections
from tracking.visualize.pointcloud import init_point_cloud, update_point_cloud
from tracking.visualize.detections import init_detections, update_detections
from tracking.visualize.tracks import init_tracks, update_tracks
from tracking.association.association_tracking import track_multiple_objects
from tracking.filter import bicycle


DATA_FILE = "data/data_109.h5"
MIN_SEC = 5
DEFAULT_ANGLE = -pi / 2

# Initiate data
bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.asarray((-50, -50, -50)),
                                                   np.asarray(( 50,  50,  10)))

pcloud_sequence = load_point_clouds(DATA_FILE, min_security=MIN_SEC)

det_sequence = load_detections(DATA_FILE)

# Initiate tracking

def specificdefault_state(state):
    return bicycle.default_state(state, DEFAULT_ANGLE)


camera = h5py.File(DATA_FILE, 'r')


def camera_detections():
    for frame in range(len(camera["Sequence"])):
        raw_detections = camera["Sequence"][str(frame)]["Detections"]
        positions = [list(pos) for pos in raw_detections["Pos"]]
        lengths = [max(l, w) for l, w
                    in zip(raw_detections["Length"], raw_detections["Width"])]

        detections = [np.ndarray((1, 4), buffer=np.asarray(pos + [leng]))
                      for pos, leng in zip(positions, lengths)]

        yield detections


timestamps = iter(np.asarray(camera["Timestamp"]))

tracks_sequence = track_multiple_objects(camera_detections(), timestamps,
                                         bicycle)

# Initiate visualization
o3d.visualization.gui.Application.instance.initialize()
visualizer = o3d.visualization.O3DVisualizer("Tracking: Bicycle model",
                                             width=800, height=600)

pcloud_geometry = init_point_cloud(visualizer, next(pcloud_sequence),
                                   bounding_box)

det_geometry = init_detections(visualizer, next(det_sequence))

# Initialize tracks geometry
cur_tracks, term_tracks = next(tracks_sequence)

cur_tracks_geometry = init_tracks(visualizer, cur_tracks)

term_track_palette = itertools.cycle(sns.color_palette("muted"))
term_tracks_geometry = init_tracks(visualizer, term_tracks,
                                   palette=term_track_palette)

# Set default camera view

def normalize(vector):
    """Normalize a vector, i.e. make the norm 1 while preserving direction."""
    return vector / np.linalg.norm(vector)


lookat      = np.asarray((0.0, 0.0, 0.0))
world_up    = np.asarray((0.0, 0.0, -1.0))
camera_pos  = np.asarray((9.5, 28.5, -4.70843))

visualizer.scene.camera.look_at(lookat, camera_pos, world_up)

# Set default render options
visualizer.line_width = 6
visualizer.point_size = 2


def next_frame(visualizer):
    """Callback function to progress in frame sequence."""
    global pcloud_geometry
    pcloud_geometry = update_point_cloud(visualizer, pcloud_geometry,
                                         next(pcloud_sequence), bounding_box)

    global det_geometry
    det_geometry = update_detections(visualizer, det_geometry,
                                     next(det_sequence))

    cur_tracks, term_tracks = next(tracks_sequence)
    global cur_tracks_geometry
    cur_tracks_geometry = update_tracks(visualizer, cur_tracks_geometry,
                                        cur_tracks)

    global term_tracks_geometry
    term_tracks_geometry = update_tracks(visualizer, term_tracks_geometry,
                                         term_tracks,
                                         palette=term_track_palette)


visualizer.add_action("Next frame", next_frame)

o3d.visualization.gui.Application.instance.add_window(visualizer)
o3d.visualization.gui.Application.instance.run()
