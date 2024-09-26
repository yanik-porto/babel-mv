import unittest
import sys
import os
sys.path.insert(0, os.getcwd())
from tools.matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z
import pyrender
from tools.scene3d import Scene3D
import trimesh
import numpy as np
from renderer.animation_renderer_joints_3D import AnimationRendererJoints3D


class TestPyrender(unittest.TestCase):
    def test_pyrender(self):

        width = 640
        height = 480
        # self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
        #                         viewport_height=height,
        #                         point_size=10.0)

        mesh_obj_p = "tests/1761_7df9f1da-613e-458b-973f-12bf8f0569b4_kick.npz"
        renderer = AnimationRendererJoints3D()
        renderer.load_animation(mesh_obj_p)
        _, verts = renderer.joints_from_pose(0)
        mesh = trimesh.Trimesh(verts[0].detach().cpu().numpy(), renderer.bm.faces)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.3, 0.7, 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        scene.add(mesh, 'mesh')
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = rotation_3d_x(1.57)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose = rotation_3d_x(-1.57)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        scene3d = Scene3D(viewport_width=width, viewport_height=height)
        cam = renderer.cameras["Camera1"]
        camera_translation = cam[0]
        camera_angles = cam[1]
        camera_pose = scene3d.camera_pose(camera_translation, camera_angles, inverse=False)

        camera = pyrender.IntrinsicsCamera(fx=scene3d.focal_length, fy=scene3d.focal_length,
                                           cx=scene3d.camera_center[0], cy=scene3d.camera_center[1])

        scene.add(camera, pose=camera_pose)

        pyrender.Viewer(scene)#, viewport_size=(500, 500), use_raymond_lighting=True)


if __name__ == '__main__':
    unittest.main()