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

class TestProjection(unittest.TestCase):
    def test_project_joints_to_camera(self):
        width = 640
        height = 480
        renderer = Scene3D(viewport_width=width, viewport_height=height)
        jp = "/home/yanik/Documents/datasets/BABEL_MV/kick_joints/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_kick/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_Camera0_A032_0_joints.npz"
        data = dict(np.load(jp))
        joints = data["joints"]
        camera_translation = (7.35889, -6.92579, 4.95831)
        camera_angles = (63.5593, 0, 46.6919)
        camera_pose = renderer.camera_pose([-camera_translation[0], camera_translation[1], camera_translation[2]], [camera_angles[0], -camera_angles[1], -camera_angles[2]], inverse=True)
        print(camera_pose)
        for joints_frame in joints:

            rotz = rotation_3d_z(np.radians(45))
            # breakpoint()
            jHomo = np.concatenate((joints_frame, np.ones((joints_frame.shape[0], 1))), axis=1).transpose()
            jRotated = rotz @ jHomo
            # breakpoint()
            joints_frame = jRotated.transpose()[:, :3]

            joints2d_frame = renderer.project_joints(joints_frame, camera_translation, camera_angles)
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            img = add_estimation(img, joints2d_frame)

            cv2.imshow("keypoints", img)
            cv2.waitKey(100)

if __name__ == '__main__':
    unittest.main()