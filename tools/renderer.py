import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = "surfaceless"
import pyrender
import trimesh
import numpy as np
import cv2
import time
from copy import deepcopy

from .matrix import *
from .scene3d import Scene3D
from .utils import AverageMeter
from .viz_utils import create_cube_mesh


class Renderer(Scene3D):
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length_mm=50, img_res=224, img_res_height=None, faces=None, sensor_width=36):
        super(Renderer, self).__init__(focal_length_mm, img_res, img_res_height, sensor_width)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.viewport_width,
                                       viewport_height=self.viewport_height,
                                       point_size=1.0)
        self.faces = faces

        self.load_mesh_time = AverageMeter()
        self.load_scene_time = AverageMeter()
        self.render_time = AverageMeter()
        self.camerapose_time = AverageMeter()

        st = time.time()
        self.camera_node = None
        self.mesh_node = None
        self.joints_nodes = []
        self.scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.3, 0.7, 1.0))
        self.material_joints = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.2, 0.3, 0.8, 1.0))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        # light_pose = np.eye(4)
        light_pose = rotation_3d_x(1.57)
        light_pose[:3, 3] = np.array([0, -1, 1])
        self.scene.add(light, pose=light_pose)

        light_pose = rotation_3d_x(-1.57)
        light_pose[:3, 3] = np.array([0, 1, 1])
        self.scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)
        self.load_scene_time.update(time.time() - st)


    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def reset_time(self):
        self.load_mesh_time.reset()
        self.load_scene_time.reset()
        self.render_time.reset()
        self.camerapose_time.reset()

    def load_mesh(self, vertices, from_left_hand=False):

        st = time.time()

        mesh = trimesh.Trimesh(vertices, self.faces)
        if from_left_hand:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)
        
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh, 'mesh')

        self.load_mesh_time.update(time.time() - st)

    def render_mesh(self, camera_pose, image=None, add_specific_light=False):
       
        st = time.time()
        if self.camera_node is not None:
            self.scene.set_pose(self.camera_node, pose=camera_pose)
        else:
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                            cx=self.camera_center[0], cy=self.camera_center[1])
            self.camera_node = self.scene.add(camera, pose=camera_pose, name='camera')
        if add_specific_light:
            self.add_lighting(self.scene, self.camera_node)
        self.camerapose_time.update(time.time() - st)

        st = time.time()
        color, rend_depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        if image is not None:
            output_img = (color[:, :, :3] * valid_mask +
                    (1 - valid_mask) * image)
            output_img = cv2.normalize(output_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            
        else:
            output_img = color
        
        self.render_time.update(time.time() - st)

        return output_img

    def __call__(self, vertices, camera_translation, camera_angles=[0., 0., 0.], image=None, joints=None, from_left_hand=False):
        self.load_mesh(vertices, from_left_hand)
        if joints is not None:
            self.load_joints(joints, from_left_hand)
        if from_left_hand:
            camera_translation = deepcopy(camera_translation)
            camera_translation[0] *= -1.
        camera_pose = self.camera_pose(camera_translation, camera_angles, inverse=False)
        return self.render_mesh(camera_pose, image)

    def load_joints(self, joints, from_left_hand=False):

        if from_left_hand:
            rot = rotation_3d_x(np.radians(180))
            jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
            jRotated = rot @ jHomo
            joints = jRotated.transpose()[:, :3]

        # remove previous joints
        if len(self.joints_nodes) > 0:
            for node in self.joints_nodes:
                self.scene.remove_node(node)
            self.joints_nodes = []
        # add new joints to the scene
        for i, joint in enumerate(joints):
            mesh = create_cube_mesh(joint)
            mesh = pyrender.Mesh.from_trimesh(mesh, self.material_joints)
            self.scene.add(mesh, 'joint_' + str(i))

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses