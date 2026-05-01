import trimesh

from src.utils import load_random_scene
from src.utils import diffuse_fragments

from helpers import get_random_directory

def Visualize_Diffused():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))
    diffused_meshes = diffuse_fragments(meshes)
    scene = trimesh.Scene()

    for mesh in diffused_meshes:

        mesh.visual.face_colors = trimesh.visual.random_color()
        scene.add_geometry(mesh)
    
    scene.show()


if __name__ == "__main__":
    Visualize_Diffused()