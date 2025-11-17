import open3d as o3d
import numpy as np
import time
import copy

#START TIMER
start = time.time()
#START TIMER

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


#PREPROCESSING POINT CLOUD OPERATIONS
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

def geometric_features(pcd_down, voxel_size = 0.05):
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
    return pcd_fpfh
#PREPROCESSING POINT CLOUD OPERATIONS


#REGISTRATION POINT CLOUD OPERATIONS
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
            
    return result

def view_global_registration():
    reg_type = input("Regtype fast (F) / Regtype normal (N): ")
    source_pcd_path, target_pcd_path = set_src_trg_paths()
    source_pcd = initialize_pcd(source_pcd_path)
    target_pcd = initialize_pcd(target_pcd_path)

    voxel_size = 0.15

    source_down = voxel_downsampling(source_pcd, voxel_size)
    target_down = voxel_downsampling(target_pcd, voxel_size)

    source_fpfh = geometric_features(source_down, voxel_size)
    target_fpfh = geometric_features(target_down, voxel_size)

    if reg_type.lower() == "f":    
        result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        print(result_fast)
        draw_registration_result(source_down, target_down, result_fast.transformation)
    elif reg_type.lower() == "n":
        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        print("Global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)
    else:
        pass

def refine_registration():

    pass
#REGISTRATION POINT CLOUD OPERATIONS
    

if __name__ == "__main__":
    pass


#END TIMER
end = time.time()
runtime = end-start
if runtime <= 90:
    print('Runtime is:', f"{runtime:.3f}", 's')
else:
    minutes = int(runtime // 60)
    seconds = runtime % 60
    print(f"Runtime is: {minutes}min {seconds:06.3f}sec")
#END TIMER