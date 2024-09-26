from .animation_renderer_joints_2D import AnimationRendererJoints2D
from smpl import SMPLX, SMPL
from tools.renderer import Renderer
import os
import numpy as np
import torch
import cv2
import time
from tools.utils import AverageMeter

class AnimationRendererPyrender(AnimationRendererJoints2D):
    def __init__(self, convention='LSP', skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererPyrender, self).__init__(convention, skip_existing, strict_label, n_classes, only_some_actions)


    def render_animation_in_cameras(self, cams, animation_filename, animation_folder):
        # TODO : list all cameras in the scene
        for cam in cams:
            self.render_animation_in_camera(cam, animation_filename, animation_folder)

    def render_animation_in_camera(self, camera_name, animation_filename, animation_folder):        
        an_f_noext, _ = os.path.splitext(animation_filename)
        out_folder = os.path.join(animation_folder, an_f_noext)
        os.makedirs(out_folder, exist_ok=True)

        stdname = self.babel_to_stdname(an_f_noext, camera_name)
        if stdname == "":
            return
        
        render_file_path = os.path.join(out_folder, stdname + '.avi')
        if os.path.exists(render_file_path) and self.skip_existing:
            print(render_file_path, " already exists")
            return
        
        self.load_animation(os.path.join(animation_folder, animation_filename))
        
        video=cv2.VideoWriter(render_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920,1080))

        keypoints = []

        render_time = AverageMeter()

        batch = range(self.poses.shape[0])
        torch.no_grad()
        for ib in batch:
            st = time.time()

            joints, verts = self.joints_from_pose(ib)
                
            render_time.update(time.time() - st)

            assert(camera_name in self.cameras)
            camera_translation = self.cameras[camera_name][0]
            camera_angles = self.cameras[camera_name][1]
            img_rendered = self.renderer(verts[0].detach().cpu().numpy(), camera_translation, camera_angles)#, joints=joints[0].detach().cpu().numpy())
            img_rendered *= 255 # or any coefficient
            img_rendered = img_rendered.astype(np.uint8)
            video.write(img_rendered[:, :, :3])

            if False:
                cv2.imshow("smplx", img_rendered)
                cv2.waitKey(10)

            keypoints.append(self.renderer.project_joints(joints[0].detach().cpu().numpy(), camera_translation, camera_angles))

        video.release()
        np.savez(os.path.join(out_folder, stdname + '_0_gt.npz'), keypoint=keypoints)

        print("avg time render : {est_time.avg:.3f}\t".format(est_time=render_time))
