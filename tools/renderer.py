import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
import numpy as np
import cv2
import math

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, viewport_width=224, viewport_height=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=viewport_width,
                                       viewport_height=viewport_height,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [viewport_width // 2, viewport_height // 2]
        # self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def __call__(self, vertices, camera_translation, image=None, joints=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)

        if False: #hmr from image mode
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
        else:
            rotx = trimesh.transformations.rotation_matrix(
                np.radians(-90), [1, 0, 0])
            mesh.apply_transform(rotx)
            roty = trimesh.transformations.rotation_matrix(
                np.radians(-90), [0, 1, 0])
            mesh.apply_transform(roty)
            dfdf = 0

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        if image is not None:
            output_img = (color[:, :, :3] * valid_mask +
                    (1 - valid_mask) * image)
        else:
            output_img = color
        
        if joints is not None:
            output_img = self.render_joints(joints, camera_translation, output_img)

        return output_img
    
    def project_joints(self, joints, camera_translation):
        
        # move joints in the coordinate system
        # rot = trimesh.transformations.rotation_matrix(
        #         np.radians(180), [1, 0, 0])
        # joints = trimesh.transformations.transform_points(joints, rot)
        
        rotx = trimesh.transformations.rotation_matrix(
                np.radians(-270), [1, 0, 0])
        roty = trimesh.transformations.rotation_matrix(
                np.radians(90), [0, 1, 0])
        joints = trimesh.transformations.transform_points(joints, rotx)
        joints = trimesh.transformations.transform_points(joints, roty)


        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation

        K = np.eye(3)
        K[0][0] = self.focal_length
        K[1][1] = self.focal_length
        K[0][2] = self.camera_center[0]
        K[1][2] = self.camera_center[1]

        Khomo = K @ np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        P = Khomo @ camera_pose
        jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        # jHomo = roty @ jHomo
        joints2d = P @ jHomo
        joints2d[:, :] /= joints2d[2, :]
        return joints2d[:2, :].transpose()

    def render_joints(self, joints, camera_translation, image):
        joints2d = self.project_joints(joints, camera_translation)
        imageOverlay = image.copy()
        for ikpt in range(joints2d.shape[0]):
            kptInt = (int(joints2d[ikpt][0]), int(joints2d[ikpt][1]))
            cv2.circle(imageOverlay, kptInt, radius=2, color=(0,0,255), thickness=3)

        return imageOverlay
    
    def rotation_3d_x(alpha):
        R = np.eye(4)
        R[1, 1] = math.cos(alpha)
        R[1, 2] = -math.sin(alpha)
        R[2, 1] = math.sin(alpha)
        R[2, 2] = math.cos(alpha)
        return R
    
    def rotation_3d_y(alpha):
        R = np.eye(4)
        R[0, 0] = math.cos(alpha)
        R[0, 2] = math.sin(alpha)
        R[2, 0] = -math.sin(alpha)
        R[2, 2] = math.cos(alpha)
        return R
    
    def rotation_3d_z(alpha):
        R = np.eye(4)
        R[0, 0] = math.cos(alpha)
        R[0, 1] = -math.sin(alpha)
        R[1, 0] = math.sin(alpha)
        R[1, 1] = math.cos(alpha)
        return R