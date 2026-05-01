import trimesh

from src.utils import load_random_scene

from helpers import get_random_directory


def Visualize():

    random_dir = get_random_directory()

    meshes = load_random_scene(str(random_dir))
    scene = trimesh.Scene()

    for mesh in meshes:

        mesh.visual.face_colors = trimesh.visual.random_color()
        scene.add_geometry(mesh)
    
    scene.show()


if __name__ == "__main__":
    Visualize()