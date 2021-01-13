"""Visualization for sequential point clouds."""
import numpy as np
import open3d as o3d

def init_point_cloud(visualizer, pcloud_geometry, bounding_box=None):
    """Add point clouds in proper form to open3d visualizer."""
    max_size = np.asarray(pcloud_geometry.points).shape

    pcloud_geometry_final = o3d.geometry.PointCloud()
    pcloud_geometry_final.points = o3d.utility.Vector3dVector(np.zeros(max_size))

    if bounding_box:
        pcloud_geometry_cropped = pcloud_geometry.crop(bounding_box)
        pcloud_geometry_final.points = pcloud_geometry_cropped.points
        pcloud_geometry_final.colors = pcloud_geometry_cropped.colors
    else:
        pcloud_geometry_final = pcloud_geometry

    visualizer.add_geometry(pcloud_geometry_final)

    return pcloud_geometry_final


def update_point_cloud(visualizer, old_pcloud_geometry, new_pcloud_geometry,
                       bounding_box=None):
    """Update point cloud that has a previous geometry object."""
    if bounding_box:
        pcloud_geometry = new_pcloud_geometry.crop(bounding_box)
    else:
        pcloud_geometry = new_pcloud_geometry

    old_pcloud_geometry.points = pcloud_geometry.points
    old_pcloud_geometry.colors = pcloud_geometry.colors

    return old_pcloud_geometry
