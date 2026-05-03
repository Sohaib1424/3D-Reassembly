import random
from pathlib import Path


def get_random_directory(root:str = "data") -> str:

    paths = [
    {"path": f"{root}/everyday_compressed/everyday_compressed", "depth": 1},
    {"path": f"{root}/artifact_compressed/artifact_compressed", "depth": 0},
]

    random_path = random.choice(paths)
    base = Path(random_path["path"])

    dirs = [p for p in base.iterdir() if p.is_dir()]
    random_dir = random.choice(dirs)

    if random_path["depth"] == 1:
        subdirs = [p for p in random_dir.iterdir() if p.is_dir()]
        random_dir = random.choice(subdirs)

    return random_dir