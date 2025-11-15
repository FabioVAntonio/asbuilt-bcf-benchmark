from pathlib import Path
import tarfile
import zstandard as zstd
import open3d as o3d
import subprocess
import numpy as np
import shutil
import sys
import hashlib

PROCESSED_DIR = Path(r"D:\Pointcloud data\processed data")

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

# ---- Labeling helpers -------------------------------------------------------

def _short_hash_for_file(p: Path, extra: str = "") -> str:
    """Stable short hash from path + size + mtime (+ optional extra)."""
    try:
        st = p.stat()
        payload = f"{p.resolve()};{st.st_size};{st.st_mtime_ns};{extra}".encode("utf-8", "ignore")
    except FileNotFoundError:
        payload = f"{p};{extra}".encode("utf-8", "ignore")
    return hashlib.sha1(payload).hexdigest()[:8]  # short but robust

def _unique_labeled_target(src: Path, processed_dir: Path, ext: str) -> Path:
    """
    Build a labeled filename <stem>__<parent>__<hash>.<ext>, ensure uniqueness.
    """
    parent_label = src.parent.name or "root"
    h = _short_hash_for_file(src)
    base = f"{src.stem}__{parent_label}__{h}"
    candidate = processed_dir / f"{base}.{ext.lstrip('.')}"
    if not candidate.exists():
        return candidate
    # ultra-rare: if collision, add numeric suffix
    i = 1
    while True:
        cand = processed_dir / f"{base}_{i}.{ext.lstrip('.')}"
        if not cand.exists():
            return cand
        i += 1

# -----------------------------------------------------------------------------

# Helper for writing ply from arrays
def _write_ply_from_arrays(xyz, colors, out_path: Path, voxel_size=0.03):
    if xyz.size == 0:
        raise ValueError("No points to write.")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if colors is not None:
        c = colors.astype(np.float32)
        if c.max() > 1.5:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c[:, :3])
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path

# Convert a single .npy into a .ply (and return output path)
def npy_to_ply(npy_path: Path, out_path: Path, voxel_size=0.03) -> Path:
    arr = np.load(npy_path, allow_pickle=True)

    # Case A: directly an (N,3|6) float array
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 3:
        xyz = arr[:, :3].astype(np.float32)
        colors = None
        if arr.shape[1] >= 6:
            colors = arr[:, 3:6]
        return _write_ply_from_arrays(xyz, colors, out_path, voxel_size)

    # Case B: saved dict/object with keys
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.item()
    if isinstance(arr, dict):
        xyz = arr.get("points") or arr.get("xyz") or arr.get("positions")
        rgb = arr.get("colors") or arr.get("rgb")
        if xyz is not None:
            xyz = np.asarray(xyz, dtype=np.float32)
            rgb = None if rgb is None else np.asarray(rgb)
            return _write_ply_from_arrays(xyz, rgb, out_path, voxel_size)

    # Case C: likely an image tensor (H,W,3|4)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        raise ValueError(f"{npy_path.name} looks like an image (shape {arr.shape}). "
                         "Depth+intrinsics are required to back-project to a point cloud.")

    raise ValueError(f"Unrecognized .npy layout for point cloud: {npy_path} (shape {getattr(arr, 'shape', None)})")

# 3) Convert extracted files to PLY (handles Open3D types, PDAL types, and moves+labels+converts NPY)
def convert_dir_to_ply(root: Path, processed_dir: Path, voxel_size=0.03):
    processed_dir.mkdir(parents=True, exist_ok=True)

    o3d_exts = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
    lidar_exts = {".las", ".laz", ".e57"}

    file_paths = []

    for f in root.rglob("*"):
        if not f.is_file():
            continue

        ext = f.suffix.lower()

        # --- Handle .npy: MOVE (with label) to processed_dir and convert there
        if ext == ".npy":
            try:
                target_npy = _unique_labeled_target(f, processed_dir, ".npy")

                # Ensure parent exists (it will, it's processed_dir)
                target_npy.parent.mkdir(parents=True, exist_ok=True)

                # Move (not copy) to avoid twins
                shutil.move(str(f), str(target_npy))
                print("Moved .npy to processed:", target_npy.name)

                # Convert MOVED file to .ply in processed_dir with same labeled basename
                ply_out = target_npy.with_suffix(".ply")
                npy_to_ply(target_npy, ply_out, voxel_size=voxel_size)
                print("Converted .npy -> .ply in processed:", ply_out.name)

            except Exception as e:
                print(f"ERROR moving/converting NPY {f}: {e}", file=sys.stderr)

        elif ext in o3d_exts:
            try:
                pcd = o3d.io.read_point_cloud(str(f))
                if len(pcd.points) == 0:
                    print(f"SKIP (0 pts): {f}")
                else:
                    pcd = pcd.voxel_down_sample(voxel_size)
                    out = f.with_suffix(".ply")
                    o3d.io.write_point_cloud(str(out), pcd)
                    print("Wrote:", out)
            except Exception as e:
                print(f"ERROR reading/writing with Open3D for {f}: {e}", file=sys.stderr)

        elif ext in lidar_exts:
            # Requires PDAL installed and on PATH
            try:
                out = f.with_suffix(".ply")
                cmd = [
                    "pdal", "translate", str(f), str(out),
                    "filters.voxelgrid", f"--filters.voxelgrid.cell={voxel_size}"
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print("Wrote via PDAL:", out)
            except FileNotFoundError:
                print("ERROR: PDAL not found on PATH; cannot convert", f, file=sys.stderr)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
                print(f"ERROR from PDAL for {f}: {stderr}", file=sys.stderr)

        else:
            print("Unknown type, skipping:", f)

        file_paths.append(f)

    return file_paths

def execution():
    operation_input = input("Y/N? Enter here: ")
    if operation_input.lower()  == "y":
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        all_paths = []
        base = Path(r"D:\Pointcloud data\zst data")
        for zst in base.glob("*.zst"):
            zst_file = Path(zst)
            tar_file = decompress_zst_to_tar(zst_file)
            extracted_dir = extract_tar(tar_file)
            file_paths = convert_dir_to_ply(extracted_dir, PROCESSED_DIR, voxel_size=0.03)
            all_paths.extend(file_paths)

        return all_paths
    else:
        pass

if __name__ == "__main__":
    execution()
