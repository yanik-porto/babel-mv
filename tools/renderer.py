import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = "surfaceless"
import pyrender
import trimesh
import numpy as np
# import cv2
import time

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z
from .scene3d import Scene3D
from tools.utils import AverageMeter

class Renderer(Scene3D):
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length_mm=50, viewport_width=224, viewport_height=224, faces=None):
        super(Renderer, self).__init__(focal_length_mm, viewport_width, viewport_height)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=viewport_width,
                                       viewport_height=viewport_height,
                                       point_size=1.0)
        self.faces = faces

        self.load_mesh_time = AverageMeter()
        self.load_scene_time = AverageMeter()
        self.render_time = AverageMeter()
        self.camerapose_time = AverageMeter()

        st = time.time()
        self.camera_node = None
        self.mesh_node = None
        self.scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.3, 0.7, 1.0))

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

    def reset_time(self):
        self.load_mesh_time.reset()
        self.load_scene_time.reset()
        self.render_time.reset()
        self.camerapose_time.reset()

    def load_mesh(self, vertices):

        st = time.time()

        mesh = trimesh.Trimesh(vertices, self.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)
        
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh, 'mesh')

        self.load_mesh_time.update(time.time() - st)

    def render_mesh(self, camera_pose, image=None, joints=None):
        # st = time.time()

        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(0.5, 0.3, 0.7, 1.0))

        # # camera_translation[0] *= -1.

        # mesh = trimesh.Trimesh(vertices, self.faces)

        # if False: #hmr from image mode
        #     rot = trimesh.transformations.rotation_matrix(
        #         np.radians(180), [1, 0, 0])
        #     mesh.apply_transform(rot)
        # else:
        #     # rotx = trimesh.transformations.rotation_matrix(
        #     #     np.radians(-90), [1, 0, 0])
        #     # mesh.apply_transform(rotx)
        #     # roty = trimesh.transformations.rotation_matrix(
        #     #     np.radians(-90), [0, 1, 0])
        #     # mesh.apply_transform(roty)

        #     # rotx = trimesh.transformations.rotation_matrix(
        #     #     np.radians(-90), [1, 0, 0])
        #     # mesh.apply_transform(rotx)
        #     # roty = trimesh.transformations.rotation_matrix(
        #     #     np.radians(-90), [0, 1, 0])
        #     # mesh.apply_transform(roty)

        #     # mesh.apply_transform(camera_rotation)
        #     dfdf = 0

        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        # self.load_mesh_time.update(time.time() - st)

        # scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        # scene.add(mesh, 'mesh')

        # build camera transformation matrix
        # camera_pose = self.camera_pose([-camera_translation[0], camera_translation[1], camera_translation[2]], [camera_angles[0], -camera_angles[1], -camera_angles[2]], inverse=True)
        
        st = time.time()
        # camera_pose = self.camera_pose(camera_translation, camera_angles, inverse=False)
        if self.camera_node is not None:
            self.scene.set_pose(self.camera_node, pose=camera_pose)
        else:
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                            cx=self.camera_center[0], cy=self.camera_center[1])
            self.camera_node = self.scene.add(camera, pose=camera_pose, name='camera')
        self.camerapose_time.update(time.time() - st)

        # V, P = self.renderer._renderer._get_camera_matrices(scene)
        # print(V)
        # print(P)

        # pyrender.Viewer(scene, viewport_size=(1000, 1000))


        st = time.time()
        # color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color, rend_depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        if image is not None:
            output_img = (color[:, :, :3] * valid_mask +
                    (1 - valid_mask) * image)
        else:
            output_img = color
        
        # if joints is not None:
        #     output_img = self.render_joints(joints, camera_translation, camera_angles, output_img)
        self.render_time.update(time.time() - st)

        return output_img

    def __call__(self, vertices, camera_translation, camera_angles=[0., 0., 0.], image=None, joints=None):
        self.load_mesh(vertices)
        camera_pose = self.camera_pose(camera_translation, camera_angles, inverse=False)
        return self.render_mesh(camera_pose, image, joints)

    # def render_joints(self, joints, camera_translation, camera_angles, image):
    #     joints2d = self.project_joints(joints, camera_translation, camera_angles)
    #     imageOverlay = image.copy()
    #     for ikpt in range(joints2d.shape[0]):
    #         kptInt = (int(joints2d[ikpt][0]), int(joints2d[ikpt][1]))
    #         cv2.circle(imageOverlay, kptInt, radius=2, color=(0,0,255), thickness=3)

    #     return imageOverlay