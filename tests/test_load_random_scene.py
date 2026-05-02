import time

from src.utils import load_random_scene

from helpers import get_random_directory


def test_load_random_scene():

    random_dir = get_random_directory()

    start = time.perf_counter()
    loaded_meshes = load_random_scene(str(random_dir))
    end = time.perf_counter()

    assert loaded_meshes is not None
    assert len(loaded_meshes) > 0

    print(f"Loaded {len(loaded_meshes)} meshes in {end - start:.4f}s")