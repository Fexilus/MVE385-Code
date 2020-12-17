"""Visualization for sequential detections."""

def init_detections(visualizer, detections_sequence):
    """Add detections in proper form to open3d visualizer."""
    det_geometries = next(detections_sequence)

    for geometry in det_geometries:
        visualizer.add_geometry(geometry)

    return det_geometries


def update_detections(visualizer, geometries, detections_sequence):
    """Update detections that has a previous geometry object."""
    for geometry in geometries:
        visualizer.remove_geometry(geometry, False)

    det_geometries = next(detections_sequence)
    for geometry in det_geometries:
        visualizer.add_geometry(geometry, False)

    return det_geometries
