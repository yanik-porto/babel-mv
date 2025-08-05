import sys
import os
import numpy as np
import cv2
import math
import argparse

sys.path.insert(0, os.getcwd())
from tools.matrix import rotation_3d_z
from tools.scene3d import Scene3D
from tools.visualize import add_estimation
from tools.geometry import create_realistic_mask, get_unit_vector
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D

def rotate_joints(joints, rotmat):
    for jf, joints_frame in enumerate(joints):
        jHomo = np.concatenate((joints_frame, np.ones((joints_frame.shape[0], 1))), axis=1).transpose()
        jRotated = rotmat @ jHomo
        joints[jf] = jRotated.transpose()[:, :3]
    return joints

def rotate_points(points, rotmat):
    vHomo = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).transpose()
    vRotated = rotmat @ vHomo
    return vRotated.transpose()[:, :3]

def AlignHips(joints_frame, angle_hips, camera_angles, offset=0):
    # fix mesh rotation with hips
    rotz_fix = rotation_3d_z(-angle_hips + np.radians(camera_angles[2]) + np.radians(offset))
    # rotz_fix = rotation_3d_z(np.radians(camera_angles[2]))
    joints_frame = rotate_points(joints_frame, rotz_fix)
    return joints_frame

def AlignHipsAfterward(joints_frame, offset):
    # fix mesh rotation with hips
    rotz_fix = rotation_3d_z(np.radians(offset))
    # rotz_fix = rotation_3d_z(np.radians(camera_angles[2]))
    joints_frame = rotate_points(joints_frame, rotz_fix)
    return joints_frame

def find_closest_angle(random_angle):
    """
    Finds the closest angle from the predefined list to a given random angle.

    Args:
        random_angle: A float representing an angle in the range [-180, 150].

    Returns:
        A float representing the closest angle from the predefined list.
    """
    angle_degrees = list(range(-180, 180, 30))
    closest_angle = angle_degrees[0]
    min_difference = abs(random_angle - closest_angle)

    for angle in angle_degrees[1:]:
        difference = abs(random_angle - angle)
        if difference < min_difference:
            min_difference = difference
            closest_angle = angle

    return closest_angle

# jp =  "tests/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_Camera0_A032_0_joints.npz"
# jp = "/home/yanik/Documents/datasets/ixmas/joints/demo_alba1_03_scratch-head_cam0_frames_0192_0261.npz"
# jp = "/home/yanik/repos/HumanML3D/P01G01R01F0001T0064A0101.npy"
# jp = "/home/yanik/Documents/datasets/ixmas/ixmas_coco.pkl"
# jp = "/home/yanik/Documents/datasets/ixmas/joints/demo_alba1_01_check-watch_cam1_frames_0053_0097.npz"
jp = "/home/yanik/Documents/datasets/ixmas/joints_no_cam4/demo_alba1_06_turn-around_cam0_frames_0425_0497.npz"

def parse_args():
    parser = argparse.ArgumentParser(description="Test 2d projection of a 3D squeleton sequence")
    parser.add_argument('seq_path', help="path to the file containing the squleton sequence")
    parser.add_argument('--face_hips_front', action='store_true', help="faces the hips in front of the camera")
    parser.add_argument('--closest_node', action='store_true', help="rotate the hips to the closest yaw node")
    parser.add_argument('--yaw', type=int, default=0, help='yaw rotation of the body in the first frame')
    parser.add_argument('--inverse_axes', action="store_true", help="Inverse the squeleton axes to be z up")
    parser.add_argument('--camera', default='Camera1', help="Name of the camera to project on")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    width = 640
    height = 480
    scene3d = Scene3D(viewport_width=width, viewport_height=height)

    data = np.load(args.seq_path, allow_pickle=True)
    if args.seq_path.endswith('.npz'):
        data = dict(data)
        joints = data["joints"]
    else:
        joints = data
    print(joints.shape)
    if len(joints.shape) > 3:
        joints = joints[0]



    renderer = AnimationRendererJoints3D()
    cameras = renderer.cameras
    cam = cameras[args.camera]
    camera_translation = cam[0]
    camera_angles = cam[1]

    rotz = rotation_3d_z(np.radians(args.yaw))

    if args.inverse_axes:
        trans_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0]])
        joints = np.dot(joints, np.transpose(trans_matrix))

        # joints[..., 0], joints[..., 1], joints[..., 2] = joints[..., 0].copy(), joints[..., 2].copy(), -joints[..., 1].copy()


    
    offset = 0
    if args.face_hips_front or args.closest_node:
        convention = 'tsu'
        if convention == 'coco':
            hips_idx = (12, 11)
        elif convention == 'tsu':
            hips_idx = (4, 3)
        lhip = joints[0, hips_idx[1]]
        rhip = joints[0, hips_idx[0]]
        unit_vec = get_unit_vector(rhip, lhip)
        angle_hips = math.atan2(unit_vec[1], unit_vec[0])
        print("angle hips : ", np.degrees(angle_hips), ' (unit :', unit_vec, ')')
        print("camera z angle : ", camera_angles[2])

        z_cam_angle = camera_angles[2]
        # z_cam_angle = z_cam_angle % 180. if z_cam_angle > 0 else z_cam_angle % -180.
        z_cam_angle = (z_cam_angle + 180) % 360 - 180

        z_hips_angle = (np.degrees(angle_hips) + 180) % 360 - 180
        print(z_hips_angle)

        if args.closest_node:
            closest_camera_angle = find_closest_angle(z_cam_angle)
            print("closest_camera_angle : ", closest_camera_angle)

            closest_angle = find_closest_angle(z_hips_angle)

            offset = z_cam_angle - z_hips_angle + closest_angle
        
        else:
            offset = z_cam_angle + z_hips_angle

        print("offset:", offset)


    for joints_frame in joints:

        if args.face_hips_front or args.closest_node:
            # joints_frame = AlignHips(joints_frame, angle_hips, camera_angles=[0, 0, 0])#)
            joints_frame = AlignHipsAfterward(joints_frame, offset)


        jHomo = np.concatenate((joints_frame, np.ones((joints_frame.shape[0], 1))), axis=1).transpose()
        jRotated = rotz @ jHomo
        joints_frame = jRotated.transpose()[:, :3]

        joints_frame = create_realistic_mask(joints_frame,  cam[0])

        joints2d_frame = scene3d.project_joints(joints_frame, camera_translation, camera_angles)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        img = add_estimation(img, joints2d_frame)

        cv2.imshow("keypoints", img)
        cv2.waitKey(100)

