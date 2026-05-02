import os
import torch
import trimesh
import numpy as np
from torch_geometric.data import Data
from gpytoolbox.copyleft import mesh_boolean


def get_features(mesh: trimesh.base.Trimesh)-> Data:
    """
    Gets geometric features into a PyTorch Geometric Data object.
    
    Node Features (6D): [pos(3), norm(3)]
    Edge Features (10D): [len(1), mid(3), n1(3), n2(3)]
    """

    nodes_pos = torch.from_numpy(mesh.vertices).float()
    nodes_norm = torch.from_numpy(mesh.vertex_normals).float()
    x = torch.cat([nodes_pos, nodes_norm], dim=-1)

    
    edges_unique = mesh.edges_unique # (E, 2)
    edge_index = torch.from_numpy(edges_unique.astype(np.int64)).t().contiguous()
    
    
    adj_edges_v = mesh.face_adjacency_edges # face_adjacency_edges: the edges shared by two faces
    adj_faces = mesh.face_adjacency # face_adjacency: the two faces sharing those edges
    face_normals = torch.from_numpy(mesh.face_normals).float()

    # Create the lookup map for inner edges
    edge_to_faces = {tuple(sorted(edge)): faces for edge, faces in zip(adj_edges_v, adj_faces)}

    edge_feats = []

    for i, e_idx in enumerate(edges_unique):
        edge_key = tuple(sorted(e_idx))
        
        v0, v1 = nodes_pos[e_idx[0]], nodes_pos[e_idx[1]]
        length = torch.norm(v1 - v0).view(1)
        midpoint = (v0 + v1) / 2.0
        
        if edge_key in edge_to_faces:
            f1, f2 = edge_to_faces[edge_key]
            n1, n2 = face_normals[f1], face_normals[f2]
        else:
            face_idx = mesh.edges_face[mesh.edges_unique_inverse[i]]
            n1 = face_normals[face_idx]
            n2 = n1 # We duplicate n1 so the feature vector stays 10D
            
        edge_feats.append(torch.cat([length, midpoint, n1, n2]))

    edge_attr = torch.stack(edge_feats)

    num_edges = edge_index.size(1)
    
    # Transpose edge_index from (2, E) to (E, 2) before flattening
    # Example: [e0_v0, e0_v1, e1_v0, e1_v1, ...]
    row = edge_index.t().reshape(-1) 
    
    # Example: [0, 0, 1, 1, 2, 2, ...]
    col = torch.arange(num_edges, device=edge_index.device).repeat_interleave(2)
    
    inc_index = torch.stack([row, col], dim=0)

    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        inc_index=inc_index,
        num_nodes=x.size(0)
    )


def BreakingBadDataset():

    pass