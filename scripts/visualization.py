import os
import numpy as np
import open3d as o3d

def visualization(filepath, *,
                  voxel_size=0.03,
                  nb_neighbors=24,
                  std_ratio=2.0,
                  use_radius_outlier=True,
                  radius_multiplier=3.0,   # radius = radius_multiplier * voxel_size
                  min_neighbors=8,
                  orient_k=50,
                  save_path=None):
    """
    Load a colored .ply, center it, denoise (SOR + optional radius), downsample,
    estimate & orient normals, then visualize (at the end). Optionally save.

    Parameters
    ----------
    filepath : str                Path to the .ply file (required).
    voxel_size : float            Voxel size for downsampling (meters).
    nb_neighbors : int            K for Statistical Outlier Removal.
    std_ratio : float             Std-dev multiplier for SOR (1.5–2.5 typical).
    use_radius_outlier : bool     Run radius outlier removal after SOR.
    radius_multiplier : float     Radius = multiplier * voxel_size.
    min_neighbors : int           Min neighbors within radius to keep a point.
    orient_k : int                K for consistent normal orientation.
    save_path : str or None       If set, writes cleaned PLY there.
    """
    if not filepath.lower().endswith(".ply"):
        raise ValueError("Please provide a .ply file.")

    # 1) Load
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError("Loaded point cloud is empty.")

    # 2) Center for stable visualization
    pcd.translate(-pcd.get_center())

    # 3) Statistical Outlier Removal
    pcd_sor, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # 4) Optional: radius outlier
    if use_radius_outlier:
        radius = max(1e-6, radius_multiplier * voxel_size)
        pcd_sor, _ = pcd_sor.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )

    # 5) Downsample
    pcd_ds = pcd_sor.voxel_down_sample(voxel_size=voxel_size) if voxel_size > 0 else pcd_sor

    # 6) Normals (improves the “building-like” shading)
    normal_radius = max(4 * voxel_size, 0.08)
    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    if len(pcd_ds.points) >= orient_k:
        pcd_ds.orient_normals_consistent_tangent_plane(k=orient_k)

    # 7) Save if requested (before visualization)
    if save_path:
        if not save_path.lower().endswith(".ply"):
            save_path += ".ply"
        o3d.io.write_point_cloud(save_path, pcd_ds, write_ascii=False, compressed=False)
        print(f"Saved cleaned point cloud to: {os.path.abspath(save_path)}")

    # 8) Visualization (AT THE END, single window)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Cleaned Building Point Cloud", width=1280, height=800)
    vis.add_geometry(pcd_ds)
    opt = vis.get_render_option()
    opt.point_size = 1.0              # smaller point size = less “blobby”
    opt.show_coordinate_frame = False
    vis.run()
    vis.destroy_window()

    return pcd_ds

if __name__ == "__main__":
    file_name = input("Enter filename (.ply): ")
    file_path = f"D:\Pointcloud data\processed data\{file_name}.ply"

    clean = visualization(
    file_path,
    voxel_size=0.03,
    nb_neighbors=24,
    std_ratio=2.0,
    use_radius_outlier=True,
    radius_multiplier=3.0,
    min_neighbors=8,
    orient_k=50,
    save_path=file_path
)