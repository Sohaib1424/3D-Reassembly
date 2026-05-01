import time
import numpy as np

from src.utils import load_random_scene
from src.geometry import extract_fractures

from helpers import get_random_directory


def test_extract_fracture():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))

    max_num_vertices = -np.inf
    k = 0

    # For the sake of testing time, we only extract the fracture surface of the largest piece.
    for i, mesh in enumerate(meshes):
        if mesh.vertices.shape[0] > max_num_vertices:
            k = i
            max_num_vertices = mesh.vertices.shape[0]

    start = time.perf_counter()

    frac_mesh = extract_fractures(meshes[k])

    end = time.perf_counter()

    print(f"It took {end - start:.4f}s to extract the fracture surface of the largest fragment amoung the loaded meshes.")
    print(f"#Removed extra faces: {meshes[k].faces.shape[0] - frac_mesh.faces.shape[0]}, #Remaining faces: {frac_mesh.faces.shape[0]}")
    print(f"#Removed extra vertices: {meshes[k].vertices.shape[0] - frac_mesh.vertices.shape[0]}, #Remaining vertices: {frac_mesh.vertices.shape[0]}")

    assert frac_mesh.vertices.shape[0] <= meshes[k].vertices.shape[0]
    assert frac_mesh.faces.shape[0] <= meshes[k].faces.shape[0]