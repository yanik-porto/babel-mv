import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.getcwd())

from tools.scene3d import Scene3D, random_camera_pose
from tools.viz_utils import add_camera_mesh_like_blender
from tools.visualization3d import Visualization
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D


class TestVizu3d(unittest.TestCase):
    def test_show_scene(self):
        width = 640
        height = 480
        scene3d = Scene3D(viewport_width=width, viewport_height=height)
        jp =  "tests/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_Camera0_A032_0_joints.npz"
        data = dict(np.load(jp))
        joints = data["joints"]
        visualizer = Visualization()

        renderer = AnimationRendererJoints3D()
        cams = renderer.cameras

        for cam in cams.values():
            camera_translation = cam[0]
            camera_angles = cam[1]
            camera_pose = scene3d.camera_pose(camera_translation, camera_angles, inverse=False)
            cam = add_camera_mesh_like_blender(camera_pose, camerascale=0.1, doinverse=False)
            visualizer.visualize_cameras(cam.T, [1,0,0])

        for i in range(10):
            random_pose = random_camera_pose()
            random_cam = add_camera_mesh_like_blender(random_pose, camerascale=0.1, doinverse=False)
            visualizer.visualize_cameras(random_cam.T, [0,1,0])

        eye = scene3d.camera_pose([0, 0, 0], [0, 0, 0], inverse=False)
        zero_cam = add_camera_mesh_like_blender(eye, camerascale=0.1, doinverse=False)
        visualizer.visualize_cameras(zero_cam.T, [1,0.5,0])


        visualizer.visualize_points(joints[0], [1,0,1])
        visualizer.show()

if __name__ == '__main__':
    unittest.main()