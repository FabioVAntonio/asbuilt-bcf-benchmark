import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualization(filepath):
    pcd = o3d.io.read_point_cloud(filepath)

    pcd_center = pcd.get_center()
    pcd.translate(-pcd_center)

    nn = 16
    std_multiplier = 10

    filtered_pcd = pcd.remove_statistical_outlier(nn,std_multiplier)

    outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
    outliers.paint_uniform_color([1, 0, 0])
    filtered_pcd = filtered_pcd[0]

    o3d.visualization.draw_geometries([filtered_pcd])
