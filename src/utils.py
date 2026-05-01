import os
import igl
import trimesh
import numpy as np
from scipy.sparse import load_npz
from scipy.spatial.transform import Rotation as R

def load_random_scene(mesh_dir_full_path: str) -> list:

    num_fracs = 0
    compressed_mesh_path = os.path.join(mesh_dir_full_path,
                                        "compressed_mesh.obj")
    compressed_data_path = os.path.join(mesh_dir_full_path,
                                        "compressed_data.npz")
    fine_vertices, fine_triangles = igl.read_triangle_mesh(compressed_mesh_path)
    piece_to_fine_vertices_matrix = load_npz(compressed_data_path)

    frac_dirs = [d for d in os.listdir(mesh_dir_full_path) 
             if os.path.isdir(os.path.join(mesh_dir_full_path, d))]

    random_frac_dir = np.random.choice(frac_dirs)
    random_frac_dir_full_path = os.path.join(mesh_dir_full_path, random_frac_dir)

    random_frac_data_path = os.path.join(random_frac_dir_full_path,
                                      "compressed_fracture.npy")

    piece_labels_after_impact = np.load(random_frac_data_path)

    fine_vertex_labels_after_impact = piece_to_fine_vertices_matrix @ piece_labels_after_impact

    n_pieces_after_impact = int(np.max(piece_labels_after_impact) + 1)

    meshes = []

    for i in range(n_pieces_after_impact):
        tri_labels = fine_vertex_labels_after_impact[fine_triangles[:, 0]]

        if np.any(tri_labels == i):
            vi, fi = igl.remove_unreferenced(
                fine_vertices, fine_triangles[tri_labels == i, :])[:2]
        else:
            continue
        ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
        gi = J[fi]
        ffi, _ = igl.resolve_duplicated_faces(gi)
        nv, nf, _, _ = igl.remove_unreferenced(ui,ffi) # returns: nv, nf, IM, J
        
        mesh = trimesh.Trimesh(nv, nf)
        # mesh = mesh.subdivide()
        meshes.append(mesh)
    
    return meshes

def diffuse_fragments(fragments: list, mean_vec=(0,0,0), var_vec=(.75,.75,.75)) -> list:
    """
    Applies random SE(3) transformations to fragments.
    Ensures fragments are 'scattered' without excessive internal blending.
    """
    diffused_fragments = []
    # Calculate a global 'scale' to prevent overlap
    max_dim = max([f.extents.max() for f in fragments])
    
    for i, mesh in enumerate(fragments):
        m = mesh.copy()
        # Random Rotation
        rotation = R.random().as_matrix()
        
        # Random Translation (Wiener-like step)
        # We add an 'index-based' offset to push fragments in different directions
        translation = np.random.normal(mean_vec, var_vec, size=3) 
        translation += (np.random.standard_normal(3) * max_dim * .5) # Push out
        
        # Apply transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        m.apply_transform(matrix)
        
        diffused_fragments.append(m)
    return diffused_fragments
