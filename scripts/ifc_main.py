import ifcopenshell
import ifcopenshell.geom
import open3d as o3d
import numpy as np
import time


ifc_base_path = "C:/Users/fabio/OneDrive/Dokumente/Coding/asbuilt-bcf-benchmark/PLNs and IFCs/"
ifc_test = ifcopenshell.open(ifc_base_path+"Test.ifc")


def view_ifc(ifc):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)  # put everything in global coords

    global_mesh = o3d.geometry.TriangleMesh()

    element_types = [
        "IfcWall", "IfcWallStandardCase",
        "IfcSlab",
        "IfcBeam",
        "IfcColumn",
        "IfcDoor", "IfcWindow",
        "IfcStair", "IfcStairFlight",
    ]

    products = []
    for t in element_types:
        products.extend(ifc.by_type(t))

    for product in products:
        try:
            # create_shape generates triangulated geometry for this product
            shape = ifcopenshell.geom.create_shape(settings, product)
            geom = shape.geometry

            # verts is a flat list [x0, y0, z0, x1, y1, z1, ...]
            verts = np.array(geom.verts, dtype=np.float64).reshape(-1, 3)

            # faces is a flat list of indices [i0, j0, k0, i1, j1, k1, ...]
            faces = np.array(geom.faces, dtype=np.int32).reshape(-1, 3)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

            # optional: give each element a random color
            mesh.paint_uniform_color(np.random.rand(3))

            # append to global mesh
            global_mesh += mesh

        except Exception as e:
            print(f"Skipping {product.GlobalId} ({product.is_a()}): {e}")

    # Clean up and visualize in Open3D
    global_mesh.remove_duplicated_vertices()
    global_mesh.remove_duplicated_triangles()
    global_mesh.remove_degenerate_triangles()
    global_mesh.remove_unreferenced_vertices()
    global_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([global_mesh])

if __name__ == "__main__":
    #START TIMER
    start = time.time()
    #START TIMER
    
    view_ifc(ifc_test)

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