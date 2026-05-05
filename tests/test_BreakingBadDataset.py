import time
import torch

from src.BreakingBadDataset import BreakingBadDataset

def test_BreakingBadDataset():

    start = time.perf_counter()

    num_scenes_to_test = 2
    dataset = BreakingBadDataset(
                            root_dir="data", 
                            num_scenes=num_scenes_to_test, 
                            return_meshes=True,
                            extract_frac=False)

    output = dataset[0]
    graph = output["graph"]
    diff_graph = output["diffused_graph"]
    scenarios = output["meshes"]
    diff_scenarios = output["diffused_meshes"]

    # Total nodes for #num_scenes_to_test scenes
    total_num_nodes = sum([
        sum(
            [mesh.vertices.shape[0] for mesh in scene]
        )  for scene in scenarios
    ])
    # Total edges for #num_scenes_to_test scenes
    total_num_edges = sum([
        sum(
            [mesh.edges_unique.shape[0] for mesh in scene]
        )  for scene in scenarios
    ])

    # Checking basic shapes
    assert graph.x.shape[0] == diff_graph.x.shape[0] == total_num_nodes
    assert graph.x.shape[1] == diff_graph.x.shape[1] == 6
    assert graph.edge_attr.shape[1] == diff_graph.edge_attr.shape[1] == 10
    
    # Verifying node offsets in edge_index
    # The max value in edge_index must be < total_nodes
    assert graph.edge_index.max() < graph.num_nodes
    assert diff_graph.edge_index.max() < diff_graph.num_nodes

    assert graph.edge_index.min() == diff_graph.edge_index.min() == 0

    # Verifying edge offsets in inc_index
    # inc_index[1] contains the Edge IDs. 
    assert graph.edge_attr.shape[0] == diff_graph.edge_attr.shape[0] == total_num_edges
    assert graph.inc_index[1].max() == diff_graph.inc_index[1].max() == total_num_edges - 1
    
    # Checking that Row 0 of inc_index (node pointers) matches total nodes
    assert graph.inc_index[0].max() < graph.num_nodes
    assert diff_graph.inc_index[0].max() < diff_graph.num_nodes


    # Verifying batch vector
    assert graph.batch.shape[0] == diff_graph.batch.shape[0] == total_num_nodes
    # Since we merged num_scenes_to_test scenes, the batch should have values 0, 1, ..., num_scenes_to_test
    assert torch.unique(graph.batch).tolist() == torch.unique(diff_graph.batch).tolist() == list(range(num_scenes_to_test))

    # 5. Check Contiguity
    assert graph.x.is_contiguous()
    assert diff_graph.x.is_contiguous()

    assert graph.edge_index.is_contiguous()
    assert diff_graph.edge_index.is_contiguous()


    end = time.perf_counter()

    print("\n[✔] Shapes and Offsets Verified")
    print("[✔] Successful Diffusion")
    print(f"[✔] Total Nodes: {graph.num_nodes}")
    print(f"[✔] Total Edges: {total_num_edges}")
    print("[✔] Incidence Matrix Shifting: SUCCESS")
    print(f"[✔] It took {end - start:.4f}s for the whole test to end.")