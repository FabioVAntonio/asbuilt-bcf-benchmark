import open3d as o3d
import numpy as np
import time
import copy
import os

#CONSTANTS
PLY_DATA_FOLDER = r"D:\\Pointcloud data\\processed data\\"
#CONSTANTS

#GENERAL FILEPATH FUNCTIONS
def set_src_trg_paths():
    src_file_name = input("Enter source filename (.ply): ")
    src_file_path = PLY_DATA_FOLDER+f"{src_file_name}.ply"
    
    trg_file_name = input("Enter target filename (.ply): ")
    trg_file_path = PLY_DATA_FOLDER+f"{trg_file_name}.ply"
    return src_file_path, trg_file_path

def set_file_path():
    file_name = input("Enter filename (.ply): ")
    file_path = PLY_DATA_FOLDER+f"{file_name}.ply"
    return file_path

def get_pcd_paths(folder_path: str) -> list:
    pcd_file_names = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".ply"):
            pcd_file_names.append(file)

    pcd_file_names.sort()
    return pcd_file_names
#GENERAL FILEPATH FUNCTIONS

#PREPROCESSING POINT CLOUD OPERATIONS 
def get_pcd(file_path: str):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def get_pcds_down(voxel_size: float):
    pcds_down = []
    pcd_file_names = get_pcd_paths(PLY_DATA_FOLDER)
    for pcd in pcd_file_names:
        pcd = o3d.io.read_point_cloud(PLY_DATA_FOLDER+pcd)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcds_down.append(pcd_down)
    return pcds_down

def get_pcd_down(pcd, voxel_size: float):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

#NORMALS ESTIMATION
def get_normals(pcd, voxel_size: float):
    radius_normal = voxel_size * 2.0
    print(":: Estimated normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#FPFH features
def get_fpfh_features(pcd, voxel_size:float):
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_fpfh

def initialize_pcd_dataset(voxel_size: float):
    source_pcd_path, target_pcd_path = set_src_trg_paths()
    source_pcd = get_pcd(source_pcd_path)
    target_pcd = get_pcd(target_pcd_path)

    #NORMALS ESTIMATION
    for pcd in (source_pcd, target_pcd):
        get_normals(pcd, voxel_size)

    source_down = get_pcd_down(source_pcd, voxel_size)
    target_down = get_pcd_down(target_pcd, voxel_size)
    
    return source_pcd, target_pcd, source_down, target_down
#PREPROCESSING POINT CLOUD OPERATIONS

#GEOMETRY AND ADJACENCY CALCULATIONS
#TODO calculate geometric overlap between pcds, decide if its sufficient for global reg
#GEOMETRY AND ADJACENCY CALCULATIONS

#REGISTRATION POINT CLOUD OPERATIONS
def execute_global_reg(source_pcd, target_pcd, voxel_size: float):
    distance_threshold = voxel_size * 1.5
    source_fpfh = get_fpfh_features(source_pcd, voxel_size)
    target_fpfh = get_fpfh_features(target_pcd, voxel_size)

    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result_ransac

def execute_fast_global_reg(source_down, target_down, voxel_size: float):
    distance_threshold = voxel_size * 0.5
    source_fpfh = get_fpfh_features(source_down, voxel_size)
    target_fpfh = get_fpfh_features(target_down, voxel_size)

    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
            
    return result_fgr

def refine_registration(source_pcd, target_pcd, result_ransac, voxel_size: float):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result
#REGISTRATION POINT CLOUD OPERATIONS

#MULTIWAY REGISTRATION
def pairwise_registration(source_pcd, target_pcd, voxel_size,
                          max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
    
    #intitial ransac globalisation of pairs
    result_ransac = execute_global_reg(source_pcd, target_pcd,
                                        voxel_size)
    
    #refinement with ICP on pairs
    print("Apply point-to-plane ICP on pcds")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance_coarse, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_pcd, target_pcd, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    return transformation_icp, information_icp

def get_pos_graph(pcds, voxel_size:float):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    number_of_pcds = len(pcds)
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    for pcd in pcds:
        if len(pcd.normals) == 0:
            get_normals(pcd, voxel_size)

    for source_id in range(number_of_pcds):
        print(f"\n--- Processing cloud {source_id+1}/{number_of_pcds} ---")
        for target_id in range(source_id + 1, number_of_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], voxel_size,
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
            
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=True))
    return pose_graph

def get_optimized_pos_graph(pcds, voxel_size:float):
    max_correspondence_distance_fine = voxel_size * 1.5

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        optimized_pos_graph = get_pos_graph(pcds, voxel_size)
    
    option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            optimized_pos_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
        for point_id in range(len(pcds)):
            print(optimized_pos_graph.nodes[point_id].pose)
            pcds[point_id].transform(optimized_pos_graph.nodes[point_id].pose)

    return pcds, optimized_pos_graph
#MULTIWAY REGISTRATION

#MERGE PLY
def merge_and_save_ply(pcds, output_folder, voxel_size:float):
    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd

    pcd_merged = pcd_combined.voxel_down_sample(voxel_size)
    out_path = os.path.join(output_folder, "merged_scene_downsampled.ply")

    o3d.io.write_point_cloud(out_path, pcd_merged)
    print(f"Saved merged scene to: {out_path}")
#MERGE PLY

#VIEWING OPERATIONS
def view_data(pcds:list):
    try:
        o3d.visualization.draw_geometries(pcds,
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    except:
        pass

def draw_reg_result(source, target, transformation):
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
    
def view_fast_reg(voxel_size:float):
    _, _, source_down, target_down = initialize_pcd_dataset(voxel_size)
    
    result_fast = execute_fast_global_reg(source_down, target_down, voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_reg_result(source_down, target_down, result_fast.transformation)
    
def view_global_reg(voxel_size:float):
    _, _, source_down, target_down = initialize_pcd_dataset(voxel_size)
    
    result_ransac = execute_global_reg(source_down, target_down,
                                        voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))    
    draw_reg_result(source_down, target_down, result_ransac.transformation)

def view_refined_global_reg(voxel_size:float):
    source_pcd, target_pcd, source_down, target_down = initialize_pcd_dataset(voxel_size)
    
    result_ransac = execute_global_reg(source_down, target_down, voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    
    result_icp = refine_registration(source_pcd, target_pcd, result_ransac, voxel_size)
    print("Point-to-plane ICP registration refinement took %.3f sec.\n" % (time.time() - start))
    
    draw_reg_result(source_down, target_down, result_icp.transformation)

def view_multiway_registration(voxel_size:float):   #TODO: work on reg logic
    pcds_down = get_pcds_down(voxel_size)
    pcds_aligned, _ = get_optimized_pos_graph(pcds_down, voxel_size)
    merge_and_save_ply(pcds_aligned, PLY_DATA_FOLDER, voxel_size)

    view_data(pcds_aligned)
#VIEWING OPERATIONS
    
if __name__ == "__main__":
    #START TIMER
    start = time.time()
    #START TIMER
    
    #view_refined_global_reg(voxel_size=0.05)
    view_multiway_registration(voxel_size=0.05)

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