"""Visualization for sequential detections."""

def init_detections(visualizer, detection_geometries):
    """Add detections in proper form to open3d visualizer."""

    for geometry, det_num in zip(detection_geometries, range(len(detection_geometries))):
        visualizer.add_geometry("Detection " + str(det_num), geometry)

    return detection_geometries


def update_detections(visualizer, old_det_geometries, new_det_geometries):
    """Update detections that has a previous geometry object."""
    for det_num in range(len(detection_geometries)):
        visualizer.remove_geometry("Detection " + str(det_num))

    for geometry, det_num in zip(new_det_geometries, range(len(new_det_geometries))):
        visualizer.add_geometry("Detection " + str(det_num))

    return new_det_geometries
