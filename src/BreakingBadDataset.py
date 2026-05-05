import os
import torch
import trimesh
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from torch_geometric.data import Data
from gpytoolbox.copyleft import mesh_boolean

from helpers import get_random_directory

from src.utils import load_random_scene, diffuse_fragments
from src.geometry import extract_fractures

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

    # Creating the lookup map for inner edges
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
    
    # Transposing edge_index from (2, E) to (E, 2) before flattening
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


class BreakingBadDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data",
        return_meshes: bool = False,
        num_scenes: int = 4,
        diffuse: bool = True,
        extract_frac: bool = True
    ):
        self.root_dir = root_dir
        self.return_meshes = return_meshes
        self.num_scenes = num_scenes
        self.diffuse = diffuse
        self.extract_frac = extract_frac

    def __len__(self) -> int:
        return 10000

    def _merge_data_list(self, data_list: List[Data]) -> Data:

        node_offset = 0
        edge_offset = 0
        
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_inc_index = []
        all_batch = [] # Added for PyG compatibility
        
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            num_edges = data.edge_attr.size(0)
            
            # Shifting Adjacency
            all_edge_index.append(data.edge_index + node_offset)
            
            # Shifting Incidence (Nodes in Row 0, Edge IDs in Row 1)
            shifted_inc = data.inc_index.clone()
            shifted_inc[0] += node_offset
            shifted_inc[1] += edge_offset
            all_inc_index.append(shifted_inc)
            
            # Tracking which scene/fragment each node belongs to
            all_batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            
            all_x.append(data.x)
            all_edge_attr.append(data.edge_attr)
            
            node_offset += num_nodes
            edge_offset += num_edges

        return Data(
            x=torch.cat(all_x, dim=0).contiguous(),
            edge_index=torch.cat(all_edge_index, dim=1).contiguous(),
            edge_attr=torch.cat(all_edge_attr, dim=0).contiguous(),
            inc_index=torch.cat(all_inc_index, dim=1).contiguous(),
            batch=torch.cat(all_batch, dim=0).contiguous(),
            num_nodes=node_offset
        )

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single Data object containing num_scenes merged scenarios.
        """
        all_scene_graphs = []
        all_scene_meshes = []

        all_scene_diff_graphs = []
        all_scene_diff_meshes = []

        all_scene_graphs_frac = []
        all_scene_meshes_frac = []

        all_scene_diff_graphs_frac = []
        all_scene_diff_meshes_frac = []

        transformation_matrices = []

        for _ in range(self.num_scenes):
            scene_dir = get_random_directory(self.root_dir)
            fragment_meshes = load_random_scene(str(scene_dir))

            # Extracting features (6D Nodes, 10D Edges)
            frag_data_list = [get_features(m) for m in fragment_meshes]
            # Merging fragments into a Scene Graph
            all_scene_graphs.append(self._merge_data_list(frag_data_list))

            if self.extract_frac: # Doing the same thing but for extracted fracture surfaces of fragments
                frac_meshes = [extract_fractures(m) for m in fragment_meshes]
                frac_data_list = [get_features(m) for m in frac_meshes]
                all_scene_graphs_frac.append(self._merge_data_list(frac_data_list))
            
            if self.diffuse: # Doing the same thing but for diffused fragments
                diffused_fragments, t_matrices = diffuse_fragments(fragment_meshes)
                diff_data_list = [get_features(m) for m in diffused_fragments]
                all_scene_diff_graphs.append(self._merge_data_list(diff_data_list))
                transformation_matrices.append(t_matrices)

                if self.extract_frac:
                    diff_frac = [m.copy().apply_transform(t_matrices[i]) for i, m in enumerate(frac_meshes)]
                    diff_frac_data_list = [get_features(m) for m in diff_frac]
                    all_scene_diff_graphs_frac.append(self._merge_data_list(diff_frac_data_list))
            
            if self.return_meshes:
                all_scene_meshes.append(fragment_meshes)
                if self.diffuse:
                    all_scene_diff_meshes.append(diffused_fragments)
                if self.extract_frac:
                    all_scene_meshes_frac.append(frac_meshes)
                    if self.diffuse:
                        all_scene_diff_meshes_frac.append(diff_frac)

        # Merging all scenes into the Global Batch Graph
        global_batch_graph = self._merge_data_list(all_scene_graphs)
        if self.diffuse: # Doing the same for diffused Graph
            global_batch_diff_graph = self._merge_data_list(all_scene_diff_graphs)
            
            if self.extract_frac:
                 global_batch_frac_graph = self._merge_data_list(all_scene_graphs_frac)
                 global_batch_diff_frac_graph = self._merge_data_list(all_scene_diff_graphs_frac)


        return {
            "num_scenes": self.num_scenes,
            "graph": global_batch_graph,
            "meshes": all_scene_meshes if self.return_meshes else None,
            "diffused_graph": global_batch_diff_graph if self.diffuse else None,
            "diffused_meshes": all_scene_diff_meshes if self.return_meshes and self.diffuse else None,
            "frac_graph": global_batch_frac_graph if self.extract_frac else None,
            "diff_frac_graph": global_batch_diff_frac_graph if self.extract_frac and self.diffuse else None,
            "t_matrices": transformation_matrices if self.diffuse else None
        }