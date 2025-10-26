from pathlib import Path
import tarfile
import zstandard as zstd
import open3d as o3d
import subprocess
import numpy as np

# 1) Decompress .zst -> .tar
def decompress_zst_to_tar(zst_path: Path) -> Path:
    tar_path = zst_path.with_suffix("")  # drop .zst (leaves .tar)
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f, open(tar_path, "wb") as g, dctx.stream_reader(f) as r:
        for chunk in iter(lambda: r.read(1024 * 1024), b""):
            g.write(chunk)
    return tar_path

# 2) Extract .tar -> folder
def extract_tar(tar_path: Path) -> Path:
    out_dir = tar_path.with_suffix("")  # folder named like the tar (no .tar)
    out_dir.mkdir(exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tf:
        tf.extractall(out_dir)
    return out_dir

# 3) Convert extracted files to PLY (Open3D for xyz/pcd/ply; PDAL for las/laz/e57)
def convert_dir_to_ply(root: Path, voxel_size=0.03):
    o3d_exts = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
    lidar_exts = {".las", ".laz", ".e57"}

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()

        if ext in o3d_exts:
            pcd = o3d.io.read_point_cloud(str(f))
            if len(pcd.points) == 0:
                print(f"SKIP (0 pts): {f}")
                continue
            pcd = pcd.voxel_down_sample(voxel_size)
            out = f.with_suffix(".ply")
            o3d.io.write_point_cloud(str(out), pcd)
            print("Wrote:", out)

        elif ext in lidar_exts:
            # Requires PDAL installed and on PATH
            out = f.with_suffix(".ply")
            cmd = [
                "pdal", "translate", str(f), str(out),
                "filters.voxelgrid", f"--filters.voxelgrid.cell={voxel_size}"
            ]
            subprocess.run(cmd, check=True)
            print("Wrote via PDAL:", out)

        else:
            # Unknown file type inside the tar — just report it
            print("Unknown type, skipping:", f)

def execution():
    for file in Path(r"D:\Pointcloud data\zst data").glob("*.zst"):
        zst_file = Path(file)
        tar_file = decompress_zst_to_tar(zst_file)
        extracted_dir = extract_tar(tar_file)
        convert_dir_to_ply(extracted_dir, voxel_size=0.03)

def npy_to_ply(npy_path: Path, out_path: Path, voxel_size=0.03):
    arr = np.load(npy_path, allow_pickle=True)

    # Case A: directly an (N,3|6) float array
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 3:
        xyz = arr[:, :3].astype(np.float32)
        colors = None
        if arr.shape[1] >= 6:
            colors = arr[:, 3:6]
        return _write_ply_from_arrays(xyz, colors, out_path, voxel_size)

    # Case B: saved dict/object with keys
    if isinstance(arr, (dict, np.ndarray)) and arr.dtype == object:
        obj = arr.item() if isinstance(arr, np.ndarray) else arr
        # Try common keys
        xyz = obj.get("points") or obj.get("xyz") or obj.get("positions")
        rgb = obj.get("colors") or obj.get("rgb")
        if xyz is not None:
            xyz = np.asarray(xyz, dtype=np.float32)
            if rgb is not None:
                rgb = np.asarray(rgb)
            return _write_ply_from_arrays(xyz, rgb, out_path, voxel_size)

    # Case C: likely an image tensor (H,W,3) -> not a point cloud
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] in (1,3,4):
        raise ValueError(f"{npy_path.name} looks like an image (shape {arr.shape}). "
                         "You'd need depth + intrinsics to back-project to a point cloud.")

    raise ValueError(f"Unrecognized .npy layout for point cloud: {npy_path} (shape {getattr(arr, 'shape', None)})")

def _write_ply_from_arrays(xyz, colors, out_path: Path, voxel_size=0.03):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if colors is not None:
        # normalize to [0,1] if it looks like 0..255
        c = colors.astype(np.float32)
        if c.max() > 1.5: c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c[:, :3])
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path

if __name__ == "__main__":
    execution()


