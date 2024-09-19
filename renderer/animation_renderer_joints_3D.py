from .animation_renderer import AnimationRenderer
from smpl import SMPLX, SMPL
import os
import numpy as np
import torch
import time
from tools.utils import AverageMeter

class AnimationRendererJoints3D(AnimationRenderer):
    def __init__(self, convention='LSP', skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererJoints3D, self).__init__(skip_existing, strict_label, n_classes, only_some_actions)
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
        self.trans = None

    def render_animation(self, animation_folder, animation_filename, cams):
        self.clear()

        classidx = self.get_classidx_from_filename(os.path.splitext(animation_filename)[0])
        if classidx >= self.n_classes:
            print("Skipped animation rendering because class index is out of range : {}".format(classidx))
            return

        print("Render animation {}".format(animation_filename))

        self.regress_joints(animation_filename, animation_folder)

        self.clear()


    def regress_joints(self, animation_filename, animation_folder):        
        an_f_noext, _ = os.path.splitext(animation_filename)
        out_folder = os.path.join(animation_folder, an_f_noext)
        os.makedirs(out_folder, exist_ok=True)

        stdname = self.babel_to_stdname(an_f_noext, "Camera0")
        if stdname == "":
            return
        
        joints_file_path = os.path.join(out_folder, stdname + '_0_joints.npz')
        if os.path.exists(joints_file_path) and self.skip_existing:
            print(joints_file_path, " already exists")
            return
        
        self.load_animation(os.path.join(animation_folder, animation_filename))

        joints_sequence = []

        render_time = AverageMeter()

        batch = range(self.poses.shape[0])
        torch.no_grad()
        for ib in batch:
            st = time.time()

            joints, _ = self.joints_from_pose(ib)
                
            render_time.update(time.time() - st)

            joints_sequence.append(joints[0].detach().cpu().numpy())

        np.savez(joints_file_path, joints=joints_sequence)

        print("avg time render : {est_time.avg:.3f}\t".format(est_time=render_time))

    def joints_from_pose(self, ib):
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
                                                body_pose=body_pose.cuda(), transl=self.trans[ib:ib+1].cuda())
            
        return joints, verts
    
    def render_animation_in_camera(self, camera_name, animation_filename, animation_folder):
        raise NotImplementedError    