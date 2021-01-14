"""Visualization for sequential tracks."""
import numpy as np
import open3d as o3d

from ..data.load import track_palette


def make_geometries_from_dict(tracks, old_track_geometries=None,
                              palette=track_palette):
    track_line_sets = {}

    for track_id, points in tracks.items():
        points_pos = np.reshape([point[0] for point in points], (-1, 3))

        if points_pos.shape[0] > 1:
            line_set = o3d.geometry.LineSet()

            line_set.points = o3d.utility.Vector3dVector(points_pos)

            line_indices = np.column_stack((range(0, len(points) - 1),
                                            range(1, len(points))))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices))

            if old_track_geometries:
                if track_id in old_track_geometries:
                    old_color = np.asarray(old_track_geometries[track_id].colors)[0,:]
                    line_set.paint_uniform_color(old_color)
                else:
                    line_set.paint_uniform_color(np.asarray(next(palette)))
            else:
                line_set.paint_uniform_color(np.asarray(next(palette)))

            track_line_sets[track_id] = line_set

    return track_line_sets


def init_tracks(visualizer, tracks, palette=track_palette):
    """Add detections in proper form to open3d visualizer."""
    tracks_geometries = make_geometries_from_dict(tracks, palette=palette)

    for track_id, geometry in tracks_geometries.items():
        visualizer.add_geometry("Track " + str(track_id), geometry)

    return tracks_geometries


def update_tracks(visualizer, old_geometries, new_tracks,
                  palette=track_palette):
    """Update detections that has a previous geometry object."""
    for track_id, geometry in old_geometries.items():
        visualizer.remove_geometry("Track " + str(track_id))

    new_geometries = make_geometries_from_dict(new_tracks, old_geometries,
                                               palette=palette)

    for track_id, geometry in new_geometries.items():
        visualizer.add_geometry("Track " + str(track_id), geometry)

    return new_geometries
