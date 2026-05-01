import trimesh

from src.utils import load_random_scene
from src.geometry import extract_fractures

from helpers import get_random_directory

# Visualization might take some time since extracting fracture surfaces
# is time consuming. Be patient.
def Visualize_Fractures():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))
    scene = trimesh.Scene()

    for mesh in meshes:

        frac_mesh = extract_fractures(mesh)
        frac_mesh.visual.face_colors = trimesh.visual.random_color()

        scene.add_geometry(frac_mesh)
    
    scene.show()

if __name__ == "__main__":
    Visualize_Fractures()