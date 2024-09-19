# import trimesh
import numpy as np

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z

class Scene3D:
    def __init__(self, focal_length_mm=50, viewport_width=224, viewport_height=224):
        sensor_width = 36
        self.focal_length_mm = focal_length_mm
        self.focal_length = focal_length_mm / sensor_width * viewport_width
        self.camera_center = [viewport_width // 2, viewport_height // 2]

    def camera_pose(self, camera_translation, camera_angles, inverse=False):
        # camera_rotation = trimesh.transformations.rotation_matrix(
        #     np.radians(camera_angles[2]), [0, 0, 1])
        # camera_rotation = camera_rotation @ trimesh.transformations.rotation_matrix(
        #     np.radians(camera_angles[1]), [0, 1, 0])
        # camera_rotation = camera_rotation @ trimesh.transformations.rotation_matrix(
        #     np.radians(camera_angles[0]), [1, 0, 0])
        
        camera_rotation = rotation_3d_z(np.radians(camera_angles[2]))
        camera_rotation = camera_rotation @ rotation_3d_y(np.radians(camera_angles[1]))
        camera_rotation = camera_rotation @ rotation_3d_x(np.radians(camera_angles[0]))
        Rc = camera_rotation[:3, :3]

        if not inverse:
            camera_pose = camera_rotation
            camera_pose[:3, 3] = camera_translation

            Tc = np.eye(4)
            Tc[:3, 3] = camera_translation
            
        else:
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = Rc.transpose()

            Tc = np.asarray(camera_translation).transpose()
            trans_after_rot = -1. * Rc.transpose() @ Tc
            camera_pose[:3, 3] = trans_after_rot

        return camera_pose

    def project_joints(self, joints, camera_translation, camera_angles):
        camera_pose = self.camera_pose([-camera_translation[0], camera_translation[1], camera_translation[2]], [camera_angles[0], -camera_angles[1], -camera_angles[2]], inverse=True)

        K = np.eye(3)
        K[0][0] = self.focal_length
        K[1][1] = self.focal_length
        K[0][2] = self.camera_center[0]
        K[1][2] = self.camera_center[1]

        Khomo = K @ np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        P = Khomo @ camera_pose

        joints[:, 0] *= -1.
        # joints[:, 1] *= -1.
        # joints[:, 2] *= -1.

        jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        joints2d = P @ jHomo
        joints2d[:, :] /= joints2d[2, :]
        return joints2d[:2, :].transpose()
