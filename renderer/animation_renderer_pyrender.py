from .animation_renderer import AnimationRenderer
from smpl import SMPLX, SMPL
from tools.renderer import Renderer
import os
import numpy as np
import torch
import cv2
import time
from tools.utils import AverageMeter

class AnimationRendererPyrender(AnimationRenderer):
    def __init__(self, convention='LSP', skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererPyrender, self).__init__(skip_existing, strict_label, n_classes, only_some_actions)
        assert(convention in ['LSP', 'COCO'])
        self.convention = convention

        self.bm = None
        if self.convention == 'LSP':
            self.bm = SMPLX('/home/yanik/Documents/models/smplx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.pkl',
                            batch_size=1,
                            create_transl=False,
                            ext='pkl',
                            use_pca = False,
                            num_expression_coeffs=16,
                            num_betas=16).cuda()

        elif convention == 'COCO':
            self.bm = SMPL('/home/yanik/Documents/models/smpl/SMPL_NEUTRAL.pkl',
                            batch_size=1,
                            create_transl=False).cuda()

        focal_length_mm = 50

        self.renderer = Renderer(focal_length_mm=focal_length_mm, viewport_width=1920, viewport_height=1080, faces=self.bm.faces)

    def load_animation(self, animation_path):
        if animation_path == self.animation_loaded:
            return
        
        data = dict(np.load(animation_path))
        self.betas = torch.from_numpy(data['betas']).float().cuda()
        self.poses = torch.from_numpy(data['poses']).float()#.cuda()
        self.trans = torch.from_numpy(data['trans']).float()#.cuda()
        if len(self.betas.shape) == 1:
            self.betas = self.betas.unsqueeze(0)

        self.animation_loaded = animation_path

    def clear(self):
        self.betas = None
        self.poses = None

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

        render_time = AverageMeter()

        batch = range(self.poses.shape[0])
        torch.no_grad()
        for ib in batch:
            st = time.time()

            joints, verts = None, None
            if self.convention == 'LSP':
                global_orient = self.poses[ib:ib+1, :3]
                body_pose = self.poses[ib:ib+1, 3:3+21*3]
                jaw_pose = self.poses[ib:ib+1, 22*3:23*3]
                leye_pose = self.poses[ib:ib+1, 23*3:24*3]
                reye_pose = self.poses[ib:ib+1, 24*3:25*3]
                left_hand_pose = self.poses[ib:ib+1, 25*3:25*3 + 15*3]
                right_hand_pose = self.poses[ib:ib+1, 25*3 + 15*3:25*3 + 15*3 + 15*3]
                joints, verts = self.bm(betas=self.betas,  global_orient=global_orient.cuda(),
                                                    body_pose=body_pose.cuda(),
                                                    jaw_pose=jaw_pose.cuda(),
                                                    leye_pose=leye_pose.cuda(),
                                                    reye_pose=reye_pose.cuda(),
                                                    left_hand_pose=left_hand_pose.cuda(),
                                                    right_hand_pose=right_hand_pose.cuda(),
                                                    transl=self.trans[ib:ib+1].cuda())
            elif self.convention == 'COCO':
                global_orient = self.poses[ib:ib+1, :3]
                body_pose = self.poses[ib:ib+1, 3:3+21*3].reshape(1, 21, 3)
                left_hand_pose = self.poses[ib:ib+1, 25*3:25*3 + 3].reshape(1, 1, 3)
                right_hand_pose = self.poses[ib:ib+1, 25*3 + 15*3:25*3 + 15*3 + 3].reshape(1, 1, 3)
                body_pose = torch.cat((body_pose, left_hand_pose), axis=1)
                body_pose = torch.cat((body_pose, right_hand_pose), axis=1)
                body_pose = body_pose.reshape(1, 23*3)
                joints, verts = self.bm(betas=self.betas[:, :10],  global_orient=global_orient.cuda(),
                                                    body_pose=body_pose.cuda())
                
            render_time.update(time.time() - st)


            camera_translation = [-2.,     2.,      10.]
            camera_angles = [2, 5, 90]
            assert(camera_name in self.cameras)
            camera_translation = self.cameras[camera_name][0]
            camera_angles = self.cameras[camera_name][1]
            img_rendered = self.renderer(verts[0].detach().cpu().numpy(), camera_translation, camera_angles, joints=joints[0].detach().cpu().numpy())
            img_rendered *= 255 # or any coefficient
            img_rendered = img_rendered.astype(np.uint8)
            video.write(img_rendered[:, :, :3])

            if False:
                cv2.imshow("smplx", img_rendered)
                cv2.waitKey(10)

        video.release()

        print("avg time render : {est_time.avg:.3f}\t".format(est_time=render_time))
