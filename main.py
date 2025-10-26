import visualization as v
import preprocessing as p
from pathlib import Path

root_filepath = r"D:\Pointcloud data\zst data"
specific_npy =  r"D:\Pointcloud data\zst data\site_11.coord.part003\site_11\scene_11018\coord.npy"
extracted_ply =  r"D:\Pointcloud data\zst data\site_11.coord.part003\site_11\scene_11018\coord.ply"


#p.execution() #execture this first, then the other functions
p.npy_to_ply(Path(specific_npy), Path(extracted_ply))
v.visualization(extracted_ply)