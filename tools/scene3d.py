import numpy as np
import math

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z

def randint_exclude_range(min_value, max_value, forbidden_range):
    """Generates a random integer within the specified range, excluding the forbidden range.

    Args:
        min_value: The minimum value for the random integer.
        max_value: The maximum value for the random integer.
        forbidden_range: A tuple representing the forbidden range (start, end).

    Returns:
        A random integer within the specified range, excluding the forbidden range.
    """

    while True:
        random_number = np.random.randint(min_value, max_value)
        if random_number < forbidden_range[0] or random_number > forbidden_range[1]:
            return random_number

def random_camera_translation():
    x = randint_exclude_range(-6500, 7500, (-1000, 1000)) / 1000.
    y = randint_exclude_range(-6500, 7500, (-1000, 1000)) / 1000.
    z = np.random.randint(3000, 5000) / 1000.
    return np.array([x, y, z], dtype=float)

def compute_camera_rotation(translation):
    # pich angle
    x = translation[0]
    y = translation[1]
    z = translation[2]
    tanx = math.sqrt(x**2 + y**2) / z
    x_angle = math.atan(tanx) * 180. / math.pi

    # roll angle
    y_angle = 0

    # yaw angle
    tanz = x / -y
    z_angle = math.atan(tanz) * 180. / math.pi
    if y > 0:
        z_angle += 180.

    return x_angle, y_angle, z_angle

def camera_pose(camera_translation, camera_angles, inverse=False, right2left=False):    
    camera_rotation = rotation_3d_z(np.radians(camera_angles[2]))
    camera_rotation = camera_rotation @ rotation_3d_y(np.radians(camera_angles[1]))
    camera_rotation = camera_rotation @ rotation_3d_x(np.radians(camera_angles[0]))
    Rc = camera_rotation[:3, :3]

    if right2left:
        camera_rotation = camera_rotation @ rotation_3d_x(np.radians(180))
        camera_translation = [camera_translation[0], camera_translation[1], -camera_translation[2]]


    if not inverse:
        camera_pose = camera_rotation
        camera_pose[:3, 3] = camera_translation

    elif False: # estimation of inverse
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = Rc.transpose()

        Tc = np.asarray(camera_translation).transpose()
        trans_after_rot = -1. * Rc.transpose() @ Tc
        camera_pose[:3, 3] = trans_after_rot
    else:
        camera_pose = camera_rotation
        camera_pose[:3, 3] = camera_translation
        camera_pose = np.linalg.inv(camera_pose)  

    return camera_pose

def random_camera_pose():
    rand_trans = random_camera_translation()
    cam_angles = compute_camera_rotation(rand_trans)
    random_pose = camera_pose(rand_trans, cam_angles, inverse=False)
    return random_pose

class Scene3D:
    def __init__(self, focal_length_mm=50, viewport_width=224, viewport_height=224):
        sensor_width = 36
        self.focal_length_mm = focal_length_mm
        self.focal_length = focal_length_mm / sensor_width * viewport_width
        self.camera_center = [viewport_width // 2, viewport_height // 2]

    def camera_pose(self, camera_translation, camera_angles, inverse=False, right2left=False):
        return camera_pose(camera_translation, camera_angles, inverse, right2left)

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