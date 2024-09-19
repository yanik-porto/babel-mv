import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
import numpy as np
import cv2

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z
from .scene3d import Scene3D

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

    def __call__(self, vertices, camera_translation, camera_angles=[0., 0., 0.], image=None, joints=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        # camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)

        if False: #hmr from image mode
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
        else:
            dfdf = 0

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        scene.add(mesh, 'mesh')

        # build camera transformation matrix
        camera_pose = self.camera_pose(camera_translation, camera_angles)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        # light_pose = np.eye(4)
        light_pose = rotation_3d_x(1.57)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose = rotation_3d_x(-1.57)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        if image is not None:
            output_img = (color[:, :, :3] * valid_mask +
                    (1 - valid_mask) * image)
        else:
            output_img = color
        
        if joints is not None:
            output_img = self.render_joints(joints, camera_translation, camera_angles, output_img)

        return output_img

    def render_joints(self, joints, camera_translation, camera_angles, image):
        joints2d = self.project_joints(joints, camera_translation, camera_angles)
        imageOverlay = image.copy()
        for ikpt in range(joints2d.shape[0]):
            kptInt = (int(joints2d[ikpt][0]), int(joints2d[ikpt][1]))
            cv2.circle(imageOverlay, kptInt, radius=2, color=(0,0,255), thickness=3)

        return imageOverlay