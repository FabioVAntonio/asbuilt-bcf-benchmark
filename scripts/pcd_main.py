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
    return pcd_file_names
#GENERAL FILEPATH FUNCTIONS


#INITIAL TRANSFORMATION
init_transformation = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
#INITIAL TRANSFORMATION


#PREPROCESSING POINT CLOUD OPERATIONS 
def get_pcd(file_path: str):
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def get_pcds_down(voxel_size):
    pcds_down = []
    pcd_file_names = get_pcd_paths(PLY_DATA_FOLDER)
    for pcd in pcd_file_names:
        pcd = o3d.io.read_point_cloud(PLY_DATA_FOLDER+pcd)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds_down.append(pcd_down)
    return pcds_down

def voxel_downsample_pcd(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

def geometric_features(pcd_down, voxel_size):
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

def initialize_dataset(voxel_size):
    source_pcd_path, target_pcd_path = set_src_trg_paths()
    source_pcd = get_pcd(source_pcd_path)
    target_pcd = get_pcd(target_pcd_path)

    #NORMALS ESTIMATION
    radius_normal = voxel_size * 2.0
    print(":: Estimate normals for full-res clouds with radius %.3f" % radius_normal)
    for pcd in (source_pcd, target_pcd):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal,
                max_nn=30
            )
        )

    source_down = voxel_downsample_pcd(source_pcd, voxel_size)
    target_down = voxel_downsample_pcd(target_pcd, voxel_size)

    source_fpfh = geometric_features(source_down, voxel_size)
    target_fpfh = geometric_features(target_down, voxel_size)
    return source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh

#PREPROCESSING POINT CLOUD OPERATIONS


#REGISTRATION POINT CLOUD OPERATIONS
def execute_global_reg(source_down, target_down, source_fpfh,
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

def execute_fast_global_reg(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
            
    return result

def refine_registration(source_pcd, target_pcd, result_ransac, voxel_size):
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
def multiway_registration():
    voxel_size = 0.5
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    pcds_down = get_pcds_down(voxel_size)

    for pcd in pcds_down:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0,
                max_nn=30
            )
        )

    def pairwise_registration(source, target):
        print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    def full_registration(pcds, max_correspondence_distance_coarse,
                        max_correspondence_distance_fine):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id])
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
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
    
    option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    
    return pcds_down
#MULTIWAY REGISTRATION

#SAVE PLY

#SAVE PLY

#VIEWING OPERATIONS
def view_data(voxel_size, pcds_down, multiway: bool): #voxel_size = 0.05 usually
    if multiway == False and pcds_down == []:
        file_path = set_file_path()
        pcd = get_pcd(file_path) #load .ply
        downpcds = [voxel_downsample_pcd(pcd, voxel_size)] #voxel downsample .ply
        print(np.asarray(downpcds[0].points))
        print(downpcds[0])
    elif multiway and pcds_down == []:
        downpcds = get_pcds_down(voxel_size)
        print(downpcds)
    else:
        downpcds = pcds_down
    try:
        o3d.visualization.draw_geometries(downpcds,
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    except:
        if multiway and len(downpcds.points) == 0:
            raise ValueError("Loaded point cloud is empty.")

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
    
def view_fast_reg():
    voxel_size = 0.05
    source_down, target_down, source_fpfh, target_fpfh = initialize_dataset(voxel_size)
    
    result_fast = execute_fast_global_reg(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_reg_result(source_down, target_down, result_fast.transformation)
    
def view_global_reg():
    voxel_size = 0.05
    source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh = initialize_dataset(voxel_size)
    
    result_ransac = execute_global_reg(source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))    
    draw_reg_result(source_down, target_down, result_ransac.transformation)

def view_refined_global_reg():
    voxel_size = 0.05
    source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh = initialize_dataset(voxel_size)
    
    result_ransac = execute_global_reg(source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    
    result_icp = refine_registration(source_pcd, target_pcd, result_ransac, voxel_size)
    print("Point-to-plane ICP registration refinement took %.3f sec.\n" % (time.time() - start))
    
    draw_reg_result(source_down, target_down, result_icp.transformation)
#VIEWING OPERATIONS
    

if __name__ == "__main__":
    #START TIMER
    start = time.time()
    #START TIMER
    
    pcds_down = get_pcds_down(voxel_size=0.05)
    view_data(0.05, pcds_down, multiway=False)

    #view_refined_global_reg()

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