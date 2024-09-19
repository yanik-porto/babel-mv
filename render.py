import argparse
import os
from renderer import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create babel-mv dataset")
    parser.add_argument("meshes_path", type=str, help="Path to the folder containing mesh files")
    parser.add_argument('--convention', type=str, choices=['LSP', 'COCO'], default='COCO', help="Skeleton convention to use")
    parser.add_argument('--method', type=str, choices=['BLENDER', 'PYRENDER', 'JOINTS2D', 'JOINTS3D'], default='BLENDER', help="Method of rendering")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    renderer = create_renderer(args.method.lower(), args.convention)

    for root, _, files in os.walk(args.meshes_path):
        if root != args.meshes_path:
            continue
        for f in tqdm(files):
            if f.endswith('.npz'):
                renderer.render_animation(root, f, ["Camera1", "Camera2", "Camera3"])