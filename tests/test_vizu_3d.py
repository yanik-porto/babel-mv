import sys
import os
import numpy as np
import time
import argparse
sys.path.insert(0, os.getcwd())

from tools.scene3d import Scene3D, random_camera_pose, sampled_camera_poses
from tools.viz_utils import add_camera_mesh_like_blender
from tools.visualization3d import Visualization
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D


# jp =  "tests/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_Camera0_A032_0_joints.npz"
# jp = "/home/yanik/Documents/datasets/ixmas/joints/demo_alba1_01_check-watch_cam1_frames_0053_0097.npz"
# jp = "/home/yanik/Documents/datasets/ixmas/joints/demo_alba1_05_get-up_cam3_frames_0350_0424.npz"
# jp = "tests/joints_rest_pose.npz"
# jp = "tests/babel_25j_sample_1.npy"
jp = "/home/yanik/repos/mar/mar-features-fusion/encoder/dataset/data/ntu/first_joints_original.npz"

def parse_args():
    parser = argparse.ArgumentParser(description='Test sequence loading and visualize it in 3D')
    parser.add_argument('seq_path', help="path to the file containing the squleton sequence")
    parser.add_argument('--inverse_axes', action="store_true", help="Inverse the squeleton axes to be z up")
    parser.add_argument('--append_random_camera_poses', action="store_true", help="Append random camera poses")
    parser.add_argument('--append_sampled_camera_poses', action="store_true", help="Append random camera poses")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    width = 640
    height = 480
    scene3d = Scene3D(viewport_width=width, viewport_height=height)
    data = np.load(args.seq_path)
    if args.seq_path.endswith('.npz'):
        data = dict(data)
        joints = data["joints"]
    else:
        joints = data
    print(joints.shape)
    if len(joints.shape) > 3:
        joints = joints[0]

    visualizer = Visualization()

    renderer = AnimationRendererJoints3D()
    cams = renderer.cameras

    for cam in cams.values():
        camera_translation = cam[0]
        camera_angles = cam[1]
        camera_pose = scene3d.camera_pose(camera_translation, camera_angles, inverse=False)
        cam = add_camera_mesh_like_blender(camera_pose, camerascale=0.1, doinverse=False)
        visualizer.visualize_cameras(cam.T, [1,0,0])

    if args.append_random_camera_poses:
        for i in range(100):
            random_pose = random_camera_pose()
            random_cam = add_camera_mesh_like_blender(random_pose, camerascale=0.1, doinverse=False)
            visualizer.visualize_cameras(random_cam.T, [0,1,0])

    if args.append_sampled_camera_poses:
        cams = sampled_camera_poses()
        for cam in cams:
            camera_pose = scene3d.camera_pose(cam[0], cam[1], inverse=False)
            sampled_cam = add_camera_mesh_like_blender(camera_pose, camerascale=0.1, doinverse=False)
            visualizer.visualize_cameras(sampled_cam.T, [0,0,1])

    eye = scene3d.camera_pose([0, 0, 0], [0, 0, 0], inverse=False)
    zero_cam = add_camera_mesh_like_blender(eye, camerascale=0.1, doinverse=False)
    visualizer.visualize_cameras(zero_cam.T, [1,0.5,0])

    if args.inverse_axes:
        joints[..., 0], joints[..., 1], joints[..., 2] = joints[..., 0].copy(), joints[..., 2].copy(), -joints[..., 1].copy()

    joints = joints - joints[0:1, 0:0+1]

    point_cloud = visualizer.visualize_points(joints[0], [1,0,1])
    time.sleep(1)

    for ip in range(1, len(joints)):
        time.sleep(0.1)
        visualizer.update_point_cloud(point_cloud, joints[ip])

    visualizer.show()
