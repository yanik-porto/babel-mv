import numpy as np
import math

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z

def xy_randint(min_value, max_value, forbidden_range):
    min_f_range = forbidden_range[0]
    max_f_range = forbidden_range[1]
    while True:
        x = np.random.randint(min_value, max_value)
        y = np.random.randint(min_value, max_value)
        if x < min_f_range or y < min_f_range or x > max_f_range or y > max_f_range:
            return x, y

def random_camera_translation():
    x1000, y1000 = xy_randint(-6500, 7500, (-1000, 1000))
    x, y = x1000 / 1000., y1000 / 1000.
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
    tanz = x / -y if abs(y) > 1e-3 else -math.inf
    z_angle = math.atan(tanz) * 180. / math.pi
    if y >= 0.:
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

def sampled_camera_poses(step_degree = 30):
    trans_z = 4
    radius = 7
    angles = range(-180, 180, step_degree)
    angles_rad = [math.radians(a) for a in angles]
    vertices = [[math.cos(a) * radius, math.sin(a) * radius, trans_z] for a in angles_rad]

    print(vertices)
    cams = [(t, compute_camera_rotation(t)) for t in vertices]
    return cams

class Scene3D:
    def __init__(self, focal_length_mm=50, img_res=224, img_res_height=None, sensor_width=36):
        self.viewport_height = img_res if img_res_height is None else img_res_height
        self.viewport_width = img_res
        self.focal_length_mm = focal_length_mm
        self.focal_length = focal_length_mm / sensor_width * self.viewport_width
        self.camera_center = [self.viewport_width // 2, self.viewport_height // 2]

    def camera_pose(self, camera_translation, camera_angles, inverse=False, right2left=False):
        return camera_pose(camera_translation, camera_angles, inverse, right2left)

    def project_joints(self, joints, camera_translation, camera_angles=[0, 0, 0]):
        if camera_angles == [0, 0, 0]:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
        else:
            camera_pose = self.camera_pose([-camera_translation[0], camera_translation[1], camera_translation[2]], [camera_angles[0], -camera_angles[1], -camera_angles[2]], inverse=True)

        K = np.eye(3)
        K[0][0] = self.focal_length
        K[1][1] = self.focal_length
        K[0][2] = self.camera_center[0]
        K[1][2] = self.camera_center[1]

        Khomo = K @ np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        P = Khomo @ camera_pose

        # joints[:, 0] *= -1. # TODO : check with rendering in babel_mv, useless in live but it was for babel_mv
        # joints[:, 1] *= -1.
        # joints[:, 2] *= -1.

        jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        joints2d = P @ jHomo
        joints2d[:, :] /= joints2d[2, :]
        return joints2d[:2, :].transpose()
    
    # def render_joints(self, joints, camera_translation, camera_angles, image):
    #     joints2d = self.project_joints(joints, camera_translation, camera_angles)
    #     imageOverlay = image.copy()
    #     for ikpt in range(joints2d.shape[0]):
    #         kptInt = (int(joints2d[ikpt][0]), int(joints2d[ikpt][1]))
    #         cv2.circle(imageOverlay, kptInt, radius=2, color=(0,0,255), thickness=3)

    #     return imageOverlay