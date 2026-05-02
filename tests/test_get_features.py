import time

from src.utils import load_random_scene
from src.BreakingBadDataset import get_features

from helpers import get_random_directory

def test_get_features():
    
    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))

    batch_0 = get_features(meshes[0])

    assert batch_0.x.size(0) == meshes[0].vertices.shape[0]
    assert batch_0.edge_attr.shape[0] == meshes[0].edges_unique.shape[0]

    v0_expected = batch_0.edge_index[0, 0]
    v1_expected = batch_0.edge_index[1, 0]
    assert batch_0.inc_index[0, 0] == v0_expected
    assert batch_0.inc_index[0, 1] == v1_expected
    assert batch_0.inc_index[1, 0] == 0
    assert batch_0.inc_index[1, 1] == 0