# from tqdm import tqdm
import trimesh
import numpy as np
from scipy.sparse import lil_matrix
from collections import defaultdict


def extract_shell(
    mesh: trimesh.base.Trimesh, chunk_size: int=1000
    ) -> trimesh.base.Trimesh:
    """
    This function extracts the shell of fractured pieces.

    Arg:
        # mesh: a fractured piece
        # chunk_size: to prevent memory overflow; if you have a small RAM, set
        #             a lower value, e.g. 100.
    """

    centroids = mesh.triangles_center # Getting the centers of the face surfaces
    face_normals = mesh.face_normals # Getting the normal vectors to all faces
    visible_faces_mask = np.zeros(len(mesh.faces), dtype=bool)
    
    # Defining our 6 (or 18, or 26) specific viewpoints (extending beyond the bounding box)
    # We use a multiplier to ensure we are well outside the object
    extents = mesh.extents.max() * 8.5
    # For simple objects, such as bottles, the first 6 view points will suffice
    # for a little complex objects, you can include the next 12 view points
    # for super complex objects, however, you'll need either
    # all of them or just the last 8 view points.
    viewpoints = [
        np.array([extents, 0, 0]), np.array([-extents, 0, 0]),
        np.array([0, extents, 0]), np.array([0, -extents, 0]),
        np.array([0, 0, extents]), np.array([0, 0, -extents]),

        # (x, y, 0)
        np.array([(1/np.sqrt(2))*extents, (1/np.sqrt(2))*extents, 0]), np.array([-(1/np.sqrt(2))*extents, (1/np.sqrt(2))*extents, 0]),
        np.array([(1/np.sqrt(2))*extents, -(1/np.sqrt(2))*extents, 0]), np.array([-(1/np.sqrt(2))*extents, -(1/np.sqrt(2))*extents, 0]),
        # (x, 0 , z)
        np.array([(1/np.sqrt(2))*extents, 0, (1/np.sqrt(2))*extents]), np.array([-(1/np.sqrt(2))*extents, 0, (1/np.sqrt(2))*extents]),
        np.array([(1/np.sqrt(2))*extents, 0, -(1/np.sqrt(2))*extents]), np.array([-(1/np.sqrt(2))*extents, 0, -(1/np.sqrt(2))*extents]),
        # (0, y , z)
        np.array([0,( 1/np.sqrt(2))*extents, (1/np.sqrt(2))*extents]), np.array([0, -(1/np.sqrt(2))*extents, (1/np.sqrt(2))*extents]),
        np.array([0, (1/np.sqrt(2))*extents, -(1/np.sqrt(2))*extents]), np.array([0, -(1/np.sqrt(2))*extents, -(1/np.sqrt(2))*extents]),
        
        # (x, y , z) for finding faces that can barely be seen.
        # These view points require a lot of computation and memory.
        np.array([(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents]), 
        np.array([-(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents]),
        np.array([(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents]), 
        np.array([(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents]),
        np.array([-(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents]), 
        np.array([-(1/np.sqrt(3))*extents, (1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents]),
        np.array([(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents]), 
        np.array([-(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents, -(1/np.sqrt(3))*extents]),

    ]

    for view_pos in viewpoints:
        # Updating the list of faces we still haven't seen
        candidates = np.where(~visible_faces_mask)[0]
        if len(candidates) == 0: break
        
        # Determining which candidates are facing this specific camera
        to_camera = view_pos - centroids[candidates]
        to_camera /= np.linalg.norm(to_camera, axis=1)[:, np.newaxis]
        
        # Dot product to filter back-faces (saves half the rays)
        facing_camera = np.einsum('ij,ij->i', face_normals[candidates], to_camera) > 0.05
        active_candidates = candidates[facing_camera]
        
        # Processing in CHUNKS to avoid MemoryError
        for i in range(0, len(active_candidates), chunk_size):
            batch_indices = active_candidates[i:i + chunk_size]
            
            # Origin and direction for this chunk
            origins = np.tile(view_pos, (len(batch_indices), 1))
            directions = centroids[batch_indices] - view_pos
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

            # Ray intersection
            index_tri, index_ray = mesh.ray.intersects_id(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=False
            )

            # Verification: Did the ray hit the face we intended?
            # index_ray maps back to our batch_indices
            aimed_at = batch_indices[index_ray]
            was_visible = (index_tri == aimed_at)
            
            visible_faces_mask[aimed_at[was_visible]] = True

    # 3. Final Mesh Assembly
    shell = mesh.copy()
    shell.update_faces(visible_faces_mask)
    shell.remove_unreferenced_vertices()

    return shell


def find_neighbors(
    mesh: trimesh.base.Trimesh, min_common_vertices=2
    ) -> np.array:

    num_faces = mesh.faces.shape[0]
    W = lil_matrix((num_faces, num_faces), dtype=np.float32)
    M = lil_matrix((num_faces, num_faces), dtype=np.float32)
    vertex_to_faces = defaultdict(list)

    for fi, face in enumerate(mesh.faces):
        for v in face:
            vertex_to_faces[v].append(fi)

    pair_counter = defaultdict(int)

    for face_list in vertex_to_faces.values():

        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                f1 = face_list[i]
                f2 = face_list[j]
                if f1 > f2:
                    f1, f2 = f2, f1
                pair_counter[(f1, f2)] += 1
                
    for (f1, f2), shared_count in pair_counter.items():
        if shared_count >= min_common_vertices:
            cos_sim = np.dot(mesh.face_normals[f1], mesh.face_normals[f2])
            W[f1, f2] = cos_sim
            W[f2, f1] = cos_sim
            M[f1, f2] = 1
            M[f2, f1] = 1
                
    return W, M


def extract_fractures(mesh: trimesh.base.Trimesh) -> (trimesh.base.Trimesh):

    # M: adjacency matrix for fractured surfaces
    # W: weighted adjacency matrix, W_ij is the cosin similarity
    #    of the normal vectors of two adjacent faces i and j
    W, M = find_neighbors(mesh)

    W = W.toarray()
    M = M.toarray()

    Vs = [] # a list for the vertices of fractured surfaces
    Fs = [] # a list for the fractured surfaces
    
    for fi, m in enumerate(M):
        # fi: face index
        # Checking if a face has neighbors with cosin similarity
        # less than 0.9 
        qq = np.intersect1d(np.where(m == 1), np.where(np.abs(W[fi]) < .9))

        if len(qq) != 0:
            Vs.extend(mesh.faces[fi])
    uniq_Vs = list(set(Vs))

    frac_faces_mask = np.zeros(len(mesh.faces), dtype=bool)

    for i, face in enumerate(mesh.faces):
        if np.all(np.isin(face, uniq_Vs)):
            qq = np.intersect1d(np.where(M[i] == 1), np.where(np.abs(W[i]) < .9))
            to_include = sum(M[i]) // 2 <= len(qq)
            if to_include:
                Fs.append(i)


    for i, face in enumerate(mesh.faces):
        if np.all(np.isin(face, uniq_Vs)):
            Fs.append(i)

    
    frac_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[Fs])

    return frac_mesh