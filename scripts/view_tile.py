import open3d as o3d
import numpy as np


def visualize_tile():
    file_name = input("Enter filename (.ply): ")
    file_path = f"D:\Pointcloud data\processed data\{file_name}.ply"

    pcd = initialize_pcd(file_path) #load .ply
    downpcd = voxel_downsampling(pcd) #voxel downsample .ply

    if len(downpcd.points) == 0:
        raise ValueError("Loaded point cloud is empty.")
    else:
        print(downpcd)
        print(np.asarray(downpcd.points))
        o3d.visualization.draw_geometries([downpcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
 
def initialize_pcd(file_path):
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def voxel_downsampling(pcd):
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    return downpcd
   
def tile_registration():
    src_file_name = input("Enter source filename (.ply): ")
    src_file_path = f"D:\Pointcloud data\processed data\{src_file_name}.ply"
    
    trg_file_name = input("Enter target filename (.ply): ")
    trg_file_path = f"D:\Pointcloud data\processed data\{trg_file_name}.ply"
    
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(src_file_path)
    target = o3d.io.read_point_cloud(trg_file_path)
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])


    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(trans_init)

    o3d.visualization.draw_geometries([source, target],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    