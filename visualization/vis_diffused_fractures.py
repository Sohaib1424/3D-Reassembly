import trimesh

from src.utils import load_random_scene
from src.utils import diffuse_fragments

from src.geometry import extract_fractures

from helpers import get_random_directory

# Visualization might take some time since extracting fracture surfaces
# is time consuming. Be patient.
def Visualize_Diffused_Fractures():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))
    diffused_meshes, _ = diffuse_fragments(meshes)
    scene = trimesh.Scene()

    for mesh in diffused_meshes:

        frac_mesh = extract_fractures(mesh)
        frac_mesh.visual.face_colors = trimesh.visual.random_color()

        scene.add_geometry(frac_mesh)
    
    scene.show()

if __name__ == "__main__":
    Visualize_Diffused_Fractures()