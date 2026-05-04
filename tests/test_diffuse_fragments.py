import time
import numpy as np

from src.utils import load_random_scene, diffuse_fragments

from helpers import get_random_directory


def test_diffuse_fragments():

    random_dir = get_random_directory()

    start = time.perf_counter()
    
    loaded_mesh_fragments = load_random_scene(str(random_dir))
    diffused_mesh_fragments = diffuse_fragments(loaded_mesh_fragments)

    end = time.perf_counter()

    # The 'n'th fragment, # the 'k'the vertex
    n, k = 0, -1
    assert len(loaded_mesh_fragments) == len(diffused_mesh_fragments)
    assert loaded_mesh_fragments[n].vertices.shape == diffused_mesh_fragments[n].vertices.shape
    assert np.all(loaded_mesh_fragments[n].vertices[k] != diffused_mesh_fragments[n].vertices[k])

    print(f"Loaded and Diffused {len(loaded_mesh_fragments)} Fragments in {end-start:.4f}s")