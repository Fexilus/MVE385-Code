"""Visualize tracks created using multiple object tracking with constant
acceleration motion model.
"""
import itertools

import numpy as np
import open3d as o3d
import h5py
import seaborn as sns

from tracking.data.loadmodels import load_point_clouds, load_detections
from tracking.visualize.pointcloud import init_point_cloud, update_point_cloud
from tracking.visualize.detections import init_detections, update_detections
from tracking.visualize.tracks import init_tracks, update_tracks
from tracking.association.association_tracking import track_multiple_objects
from tracking.filter import const_acceleration


DATA_FILE = "data/data_109.h5"
MIN_SEC = 5

# Initiate data
bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.asarray((-50, -50, -50)),
                                                   np.asarray(( 50,  50,  10)))

pcloud_sequence = load_point_clouds(DATA_FILE, min_security=MIN_SEC)

det_sequence = load_detections(DATA_FILE)

# Initiate tracking
camera = h5py.File(DATA_FILE, 'r')


def camera_detections():
    for i in range(len(camera["Sequence"])):
        raw_detections = camera["Sequence"][str(i)]["Detections"]
        positions = [list(pos) for pos in raw_detections["Pos"]]

        detections = [np.ndarray((1, 3), buffer=np.asarray(pos))
                      for pos in positions]

        yield detections


timestamps = iter(np.asarray(camera["Timestamp"]))

tracks_seq = track_multiple_objects(camera_detections(), timestamps,
                                    const_acceleration)

# Initiate visualization
visualizer = o3d.visualization.VisualizerWithKeyCallback()
visualizer.create_window(width=800, height=600)

pcloud_geometry = init_point_cloud(visualizer, next(pcloud_sequence), bounding_box)

det_geometry = init_detections(visualizer, next(det_sequence))

# Initialize tracks geometry
cur_tracks, term_tracks = next(tracks_seq)

cur_tracks_geometry = init_tracks(visualizer, cur_tracks)

term_track_palette = itertools.cycle(sns.color_palette("husl", 15))
term_tracks_geometry = init_tracks(visualizer, term_tracks,
                                   palette=term_track_palette)

# Set default camera view

def normalize(vector):
    """Normalize a vector, i.e. make the norm 1 while preserving direction."""
    return vector / np.linalg.norm(vector)


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


def next_frame(visualizer):
    """Callback function to progress in frame sequence."""
    global pcloud_geometry
    pcloud_geometry = update_point_cloud(visualizer, pcloud_geometry,
                                         next(pcloud_sequence), bounding_box)

    global det_geometry
    det_geometry = update_detections(visualizer, det_geometry,
                                     next(det_sequence))

    cur_tracks, term_tracks = next(tracks_seq)
    global cur_tracks_geometry
    cur_tracks_geometry = update_tracks(visualizer, cur_tracks_geometry,
                                        cur_tracks)

    global term_tracks_geometry
    term_tracks_geometry = update_tracks(visualizer, term_tracks_geometry,
                                         term_tracks,
                                         palette=term_track_palette)

    # Indicate that the geometry needs updating
    return True


visualizer.register_key_callback(32, next_frame)

visualizer.run()
