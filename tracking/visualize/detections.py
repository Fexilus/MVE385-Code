"""Visualization for sequential detections."""

def init_detections(visualizer, detection_geometries):
    """Add detections in proper form to open3d visualizer."""

    for geometry in detection_geometries:
        visualizer.add_geometry(geometry)

    return detection_geometries


def update_detections(visualizer, old_det_geometries, new_det_geometries):
    """Update detections that has a previous geometry object."""
    for geometry in old_det_geometries:
        visualizer.remove_geometry(geometry, False)

    for geometry in new_det_geometries:
        visualizer.add_geometry(geometry, False)

    return new_det_geometries
