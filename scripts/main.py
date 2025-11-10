import visualization as v
import preprocessing as p
from pathlib import Path

#1st execution: .zst -> .npy or other type of file
file_path_npys = p.execution()
file_npy_example = fr"{file_path_npys[0]}"
file_ply_example = file_npy_example.replace(".npy", ".ply") #output file of 2nd function

#2nd execution: .npy to .ply
p.npy_to_ply(Path(file_npy_example), Path(file_ply_example))

#3rd execution: postprocess the .ply, then visualize it
clean = v.visualization(
    file_ply_example,
    voxel_size=0.03,
    nb_neighbors=24,
    std_ratio=2.0,
    use_radius_outlier=True,
    radius_multiplier=3.0,
    min_neighbors=8,
    orient_k=50,
    save_path=file_ply_example
)