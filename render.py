import argparse
import os
import numpy as np
from smpl import SMPLX, SMPL
import torch
from tools.renderer import Renderer
import cv2
import time
from renderer.animation_renderer_pyrender import AnimationRendererPyrender
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create babel-mv dataset")
    parser.add_argument("meshes_path", type=str, help="Path to the folder containing mesh files")
    parser.add_argument('--convention', type=str, choices=['LSP', 'COCO'], default='LSP', help="Skeleton convention to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    renderer = AnimationRendererPyrender(args.convention)

    for root, _, files in os.walk(args.meshes_path):
        if root != args.meshes_path:
            continue
        for f in tqdm(files):
            if f.endswith('.npz'):
                renderer.render_animation(root, f, ["Camera1", "Camera2", "Camera3"])