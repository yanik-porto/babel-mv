import sys
import os
sys.path.insert(0, os.getcwd())
from tools.geometry import xz_to_xy_ground_plane
import pyrender
from tools.renderer import Renderer
import numpy as np
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D
import math
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test 2d projection of a 3D squeleton sequence")
    parser.add_argument('seq_path', help="path to the file containing the squleton sequence")
    parser.add_argument('--yaw', type=int, default=0, help='yaw rotation of the body in the first frame')
    parser.add_argument('--inverse_axes', action="store_true", help="Inverse the squeleton axes to be z up")
    parser.add_argument('--xz_to_xy', action='store_true', help="rotate the squeleton sequence so that it lies on the xy plan")
    parser.add_argument('--camera', default='Camera1', help="Name of the camera to project on")
    return parser.parse_args()

mesh_obj_p = "/home/yanik/Documents/datasets/BABEL_MV/val/10031_d9d6c092-adef-44c5-8d99-bfe6bf18377f_sit.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/ixmas/smpls/demo_alba1_05_get-up_cam2_frames_0350_0424.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/ixmas/smpls/demo_alba1_01_check-watch_cam0_frames_0053_0097.npz"
# mesh_obj_p = "tests/smpl_rest_pose.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/BABEL_MV/val/504_ebff062e-f0d3-49f8-9af2-1d78fbc80065_t pose.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/BABEL_MV/val/51_193fb406-8223-4567-a94e-0934085603e8_t pose.npz"

if __name__ == '__main__':
    args = parse_args()
    width = 640
    height = 480
    offscreen = pyrender.OffscreenRenderer(viewport_width=width,
                            viewport_height=height,
                            point_size=10.0)

    # generate vertices from pose
    renderer = AnimationRendererJoints3D()
    _, all_verts = renderer.load_animation(args.seq_path)
    anim_len = len(all_verts)

    scene3d = Renderer(viewport_width=width, viewport_height=height, faces=renderer.bm.faces)
    cam = renderer.cameras[args.camera]
    camera_translation = cam[0]
    camera_angles = cam[1]


    # Rotation matrix for axes inversion
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, -1.0, 0.0]])


    for im in range(anim_len):
        # redo smpl forward if global orient has to be changed
        if args.xz_to_xy:
            global_orient = renderer.poses[im, :3]
            global_orient[0] = 0
            global_orient = xz_to_xy_ground_plane(global_orient)
            renderer.poses[im, :3] = global_orient
            global_orient = [a * 180 / math.pi for a in global_orient]
            print("global_orient: ", global_orient)
            _, verts = renderer.joints_from_pose(im)
            verts = verts[0].detach().cpu().numpy()

        else:
            verts = all_verts[im]

        if args.inverse_axes:
            verts = np.dot(verts, np.transpose(trans_matrix))

        # render
        outimg = scene3d(verts, camera_translation, camera_angles)
        outimg *= 255 # or any coefficient
        outimg = outimg.astype(np.uint8)

        cv2.imshow("img", outimg[:, :, :3])
        cv2.waitKey(0)
