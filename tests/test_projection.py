import unittest
import pickle
import sys
import os
import numpy as np
import cv2
sys.path.insert(0, os.getcwd())
from tools.matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z
from tools.scene3d import Scene3D
from tools.visualize import add_estimation
from tools.geometry import vector_mesh_intersection, create_realistic_mask
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D

class TestProjection(unittest.TestCase):
    def test_project_joints_to_camera(self):
        width = 640
        height = 480
        scene3d = Scene3D(viewport_width=width, viewport_height=height)
        jp =  "tests/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_Camera0_A032_0_joints.npz"
        data = dict(np.load(jp))
        joints = data["joints"]

        renderer = AnimationRendererJoints3D()
        cameras = renderer.cameras
        cam = cameras["Camera1"]

        joints = create_realistic_mask(np.expand_dims(joints, axis=0), cam[0])[0]

        camera_translation = cam[0]
        camera_angles = cam[1]
        rotz = rotation_3d_z(np.radians(0))
        for joints_frame in joints:

            jHomo = np.concatenate((joints_frame, np.ones((joints_frame.shape[0], 1))), axis=1).transpose()
            jRotated = rotz @ jHomo
            joints_frame = jRotated.transpose()[:, :3]

            joints2d_frame = scene3d.project_joints(joints_frame, camera_translation, camera_angles)
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            img = add_estimation(img, joints2d_frame)

            cv2.imshow("keypoints", img)
            cv2.waitKey(100)

if __name__ == '__main__':
    unittest.main()