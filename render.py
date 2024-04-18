import argparse
import os
import numpy as np
from smpl import SMPLX, SMPL
import torch
from tools.renderer import Renderer
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Create babel-mv dataset")
    parser.add_argument("meshes_path", type=str, help="Path to the folder containing mesh files")
    parser.add_argument('--convention', type=str, choices=['LSP', 'COCO'], default='LSP', help="Skeleton convention to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    bm = None
    if args.convention == 'LSP':
        bm = SMPLX('/home/yanik/Documents/models/smplx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.pkl',
                        batch_size=1,
                        create_transl=False,
                        ext='pkl',
                        use_pca = False,
                        num_expression_coeffs=16,
                        num_betas=16).cuda()

    elif args.convention == 'COCO':
        bm = SMPL('/home/yanik/Documents/models/smpl/SMPL_NEUTRAL.pkl',
                        batch_size=1,
                        create_transl=False).cuda()

    focal_length = 5000.

    renderer = Renderer(focal_length=focal_length, viewport_width=640, viewport_height=480, faces=bm.faces)

    for root, _, files in os.walk(args.meshes_path):
        for f in files:
            if f.endswith('.npz'):
                animation_path = os.path.join(root, f)
                data = dict(np.load(animation_path))
                print(data.keys())
                print(data['poses'].shape)

                betas = torch.from_numpy(data['betas']).float().cuda()
                poses = torch.from_numpy(data['poses']).float()#.cuda()

                if len(betas.shape) == 1:
                    betas = betas.unsqueeze(0)
                print(poses[:, 3:3+21*3].shape)

                # batch= [0:150]
                batch = range(poses.shape[0])

                torch.no_grad()

                for ib in batch:
                    st = time.time()

                    joints, verts = None, None
                    if args.convention == 'LSP':
                        global_orient = poses[ib:ib+1, :3]
                        body_pose = poses[ib:ib+1, 3:3+21*3]
                        jaw_pose = poses[ib:ib+1, 22*3:23*3]
                        leye_pose = poses[ib:ib+1, 23*3:24*3]
                        reye_pose = poses[ib:ib+1, 24*3:25*3]
                        left_hand_pose = poses[ib:ib+1, 25*3:25*3 + 15*3]
                        right_hand_pose = poses[ib:ib+1, 25*3 + 15*3:25*3 + 15*3 + 15*3]
                        joints, verts = bm(betas=betas,  global_orient=global_orient.cuda(),
                                                            body_pose=body_pose.cuda(),
                                                            jaw_pose=jaw_pose.cuda(),
                                                            leye_pose=leye_pose.cuda(),
                                                            reye_pose=reye_pose.cuda(),
                                                            left_hand_pose=left_hand_pose.cuda(),
                                                            right_hand_pose=right_hand_pose.cuda())
                    elif args.convention == 'COCO':
                        global_orient = poses[ib:ib+1, :3]
                        body_pose = poses[ib:ib+1, 3:3+21*3].reshape(1, 21, 3)
                        left_hand_pose = poses[ib:ib+1, 25*3:25*3 + 3].reshape(1, 1, 3)
                        right_hand_pose = poses[ib:ib+1, 25*3 + 15*3:25*3 + 15*3 + 3].reshape(1, 1, 3)
                        body_pose = torch.cat((body_pose, left_hand_pose), axis=1)
                        body_pose = torch.cat((body_pose, right_hand_pose), axis=1)
                        body_pose = body_pose.reshape(1, 23*3)
                        joints, verts = bm(betas=betas[:, :10],  global_orient=global_orient.cuda(),
                                                            body_pose=body_pose.cuda())
                    else:
                        exit()
                    
                    
                    print("time estimation : {est_time:.3f}\t".format(est_time=time.time() - st))

                    # print("joints: ", joints.shape)
                    # print("verts: ", verts.shape)
                    # breakpoint()
                    camera_translation = [0.,     0.,      40.]
                    # camera_translation = [1.1297, 0.1291, 0.1926]
                    # camera_translation = [0.1291,     0.19265,      39.518]

                    img_rendered = renderer(verts[0].detach().cpu().numpy(), camera_translation, joints=joints[0].detach().cpu().numpy())

                    cv2.imshow("smplx", img_rendered)
                    cv2.waitKey(10)