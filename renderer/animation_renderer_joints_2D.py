from .animation_renderer_joints_3D import AnimationRendererJoints3D
from tools.renderer import Renderer
import os
import numpy as np
import torch
import time
from tools.utils import AverageMeter

class AnimationRendererJoints2D(AnimationRendererJoints3D):
    def __init__(self, convention='LSP', skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererJoints2D, self).__init__(convention, skip_existing, strict_label, n_classes, only_some_actions)

        focal_length_mm = 70
        self.image_width = 640
        self.image_height = 480

        # focal_length_mm = 50
        # self.image_width = 1920
        # self.image_height = 1080

        st = time.time()
        # self.renderer = Renderer(focal_length_mm=focal_length_mm, viewport_width=1920, viewport_height=1080, faces=self.bm.faces)
        self.renderer = Renderer(focal_length_mm=focal_length_mm, img_res=self.image_width, img_res_height=self.image_height, faces=self.bm.faces)
        print("total init renderer : ", time.time() - st, " sec")

    def render_animation_in_camera(self, camera_name, animation_filename, animation_folder):        
        an_f_noext, _ = os.path.splitext(animation_filename)
        out_folder = os.path.join(animation_folder, an_f_noext)
        os.makedirs(out_folder, exist_ok=True)

        stdname = self.babel_to_stdname(an_f_noext, camera_name)
        if stdname == "":
            return
        
        kpts_file_path = os.path.join(out_folder, stdname + '_0_gt.npz')
        if os.path.exists(kpts_file_path) and self.skip_existing:
            print(kpts_file_path, " already exists")
            return
        
        self.load_animation(os.path.join(animation_folder, animation_filename))

        keypoints = []

        render_time = AverageMeter()

        batch = range(self.poses.shape[0])
        torch.no_grad()
        for ib in batch:
            st = time.time()

            joints, _ = self.joints_from_pose(ib)
                
            render_time.update(time.time() - st)

            assert(camera_name in self.cameras)
            camera_translation = self.cameras[camera_name][0]
            camera_angles = self.cameras[camera_name][1]

            keypoints.append(self.renderer.project_joints(joints[0].detach().cpu().numpy(), camera_translation, camera_angles))

        np.savez(kpts_file_path, keypoint=keypoints)

        print("avg time render : {est_time.avg:.3f}\t".format(est_time=render_time))
