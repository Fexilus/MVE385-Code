"""Visualization for sequential point clouds."""
import numpy as np
import open3d as o3d

def init_point_cloud(visualizer, point_cloud_sequence, bounding_box=None):
    """Add point clouds in proper form to open3d visualizer."""
    pcloud_geometry_full = next(point_cloud_sequence)
    max_size = np.asarray(pcloud_geometry_full.points).shape

    pcloud_geometry = o3d.geometry.PointCloud()
    pcloud_geometry.points = o3d.utility.Vector3dVector(np.zeros(max_size))

    if bounding_box:
        pcloud_geometry_cropped = pcloud_geometry_full.crop(bounding_box)
        pcloud_geometry.points = pcloud_geometry_cropped.points
        pcloud_geometry.colors = pcloud_geometry_cropped.colors
    else:
        pcloud_geometry = pcloud_geometry_full

    visualizer.add_geometry(pcloud_geometry)

    return pcloud_geometry


def update_point_cloud(visualizer, geometry, point_cloud_sequence,
                       bounding_box=None):
    """Update point cloud that has a previous geometry object."""
    if bounding_box:
        pcloud_geometry = next(point_cloud_sequence).crop(bounding_box)
    else:
        pcloud_geometry = next(point_cloud_sequence)

    geometry.points = pcloud_geometry.points
    geometry.colors = pcloud_geometry.colors

    return geometry
