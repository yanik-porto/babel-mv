import sys
import os
sys.path.insert(0, os.getcwd())
from tools.matrix import rotation_3d_x, rotation_3d_z
import pyrender
from tools.scene3d import Scene3D, camera_pose
import trimesh
import numpy as np
import argparse
import math

from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D
from tools.geometry import xz_to_xy_ground_plane

def parse_args():
    parser = argparse.ArgumentParser(description="Test 2d projection of a 3D squeleton sequence")
    parser.add_argument('seq_path', help="path to the file containing the squleton sequence")
    parser.add_argument('--yaw', type=int, default=0, help='yaw rotation of the body in the first frame')
    parser.add_argument('--inverse_axes', action="store_true", help="Inverse the squeleton axes to be z up")
    parser.add_argument('--xz_to_xy', action='store_true', help="rotate the squeleton sequence so that it lies on the xy plan")
    parser.add_argument('--camera', default='Camera1', help="Name of the camera to project on")
    return parser.parse_args()


mesh_obj_p = "/home/yanik/Documents/datasets/BABEL_MV/val/10031_d9d6c092-adef-44c5-8d99-bfe6bf18377f_sit.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/ixmas/smpls/demo_alba1_05_get-up_cam3_frames_0350_0424.npz"
# mesh_obj_p = "/home/yanik/Documents/datasets/ixmas/smpls/demo_alba1_01_check-watch_cam0_frames_0053_0097.npz"
# mesh_obj_p = "tests/smpl_rest_pose.npz"

if __name__ == '__main__':
    args = parse_args()

    width = 640
    height = 480

    # generate vertices from pose
    renderer = AnimationRendererJoints3D()
    anim_len = renderer.load_animation(args.seq_path)

    global_orient = renderer.poses[0, :3]
    global_orient = [a * 180 / math.pi for a in global_orient]
    print("global_orient: ", global_orient)

    # create scene
    scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = rotation_3d_x(1.57)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose = rotation_3d_x(-1.57)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    scene3d = Scene3D(viewport_width=width, viewport_height=height)
    cam = renderer.cameras[args.camera]
    camera_translation = cam[0]
    camera_angles = cam[1]
    campose = camera_pose(camera_translation, camera_angles, inverse=False)

    camera = pyrender.IntrinsicsCamera(fx=scene3d.focal_length, fy=scene3d.focal_length,
                                        cx=scene3d.camera_center[0], cy=scene3d.camera_center[1])

    scene.add(camera, pose=campose)

    if args.xz_to_xy:
        renderer.poses[0, :3] = xz_to_xy_ground_plane(renderer.poses[0, :3])


    # add mesh
    _, verts = renderer.joints_from_pose(0)
    verts = verts[0].detach().cpu().numpy()

    # Rotate around z axis
    rotz = rotation_3d_z(np.radians(args.yaw))
    vHomo = np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1).transpose()
    vRotated = rotz @ vHomo
    verts = vRotated.transpose()[:, :3]

    if args.inverse_axes:
        verts[..., 0], verts[..., 1], verts[..., 2] = verts[..., 0].copy(), -verts[..., 2].copy(), verts[..., 1].copy()
        verts[..., 0], verts[..., 1], verts[..., 2] = verts[..., 0].copy(), verts[..., 2].copy(), -verts[..., 1].copy()


    mesh = trimesh.Trimesh(verts, renderer.bm.faces)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=(0.5, 0.3, 0.7, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene.add(mesh, 'mesh')

    pyrender.Viewer(scene)#, viewport_size=(500, 500), use_raymond_lighting=True)

    # # scene.add(camera, pose=random_camera_pose())
