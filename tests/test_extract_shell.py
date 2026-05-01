import time
import numpy as np

from src.geometry import extract_shell
from src.utils import load_random_scene

from helpers import get_random_directory

def test_extract_shell():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))

    k = 0
    min_num_vertices = np.inf
    # Extracting the shell of fragments is time consuming.
    # For the sake of testing time, we only extract the shell of the smallest piece.
    for i, mesh in enumerate(meshes):
        if mesh.vertices.shape[0] < min_num_vertices:
            k = i
            min_num_vertices = mesh.vertices.shape[0]

    start = time.perf_counter()

    shell = extract_shell(meshes[k])

    end = time.perf_counter()

    assert shell.vertices.shape[0] <= meshes[k].vertices.shape[0]
    assert shell.faces.shape[0] <= meshes[k].faces.shape[0]

    print(f"It took {end - start:.4f}s to extract the shell of the smallest fragment amoung the loaded meshes.")
    print(f"#Removed extra faces: {meshes[k].faces.shape[0] - shell.faces.shape[0]}, #Remaining faces: {shell.faces.shape[0]}")
    print(f"#Removed extra vertices: {meshes[k].vertices.shape[0] - shell.vertices.shape[0]}, #Remaining vertices: {shell.vertices.shape[0]}")