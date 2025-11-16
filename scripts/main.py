import open3d as o3d
import numpy as np


#GENERAL FILEPATH SETTING
def set_src_trg_paths():
    src_file_name = input("Enter source filename (.ply): ")
    src_file_path = f"D:\Pointcloud data\processed data\{src_file_name}.ply"
    
    trg_file_name = input("Enter target filename (.ply): ")
    trg_file_path = f"D:\Pointcloud data\processed data\{trg_file_name}.ply"
    return src_file_path, trg_file_path

def set_file_path():
    file_name = input("Enter filename (.ply): ")
    file_path = f"D:\Pointcloud data\processed data\{file_name}.ply"
    return file_path
#GENERAL FILEPATH SETTING


#INITIAL TRANSFORMATION
init_transformation = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
#INITIAL TRANSFORMATION


#POINT CLOUD OPERATIONS
def view_tile(voxel_size = 0.05): #voxel_size = 0.05 as a test
    file_path = set_file_path()

    pcd = initialize_pcd(file_path) #load .ply
    downpcd = voxel_downsampling(pcd, voxel_size) #voxel downsample .ply

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
 
def initialize_pcd(file_path: str):
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def voxel_downsampling(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

def geometric_features(voxel_size = 0.05):
    file_path = set_file_path

    pcd = initialize_pcd(file_path)
    pcd_down = voxel_downsampling(pcd, voxel_size)

    #estimate normals
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    #FPFH features
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    print(pcd_fpfh)
    return pcd_down, pcd_fpfh

def tile_reg():
    src_file_path, trg_file_path = set_src_trg_paths()
    
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(src_file_path)
    target = o3d.io.read_point_cloud(trg_file_path)
    threshold = 0.02

    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(init_transformation)

    o3d.visualization.draw_geometries([source, target],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
#POINT CLOUD OPERATIONS
    
if __name__ == "__main__":
    pass