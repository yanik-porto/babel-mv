import argparse
import os
import numpy as np
from smpl import SMPLX
import torch
from tools.renderer import Renderer
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Create babel-mv dataset")
    parser.add_argument("meshes_path", type=str, help="Path to the folder containing mesh files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    smplx = SMPLX('/home/yanik/Documents/models/smplx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.pkl',
                    batch_size=150,
                    create_transl=False,
                    ext='pkl',
                    use_pca = False,
                    num_expression_coeffs=16,
                    num_betas=16).cuda()
                    # create_expression=False).cuda()

    focal_length = 5000.
    # img_res = 224
    renderer = Renderer(focal_length=focal_length, viewport_width=640, viewport_height=480, faces=smplx.faces)

    for root, _, files in os.walk(args.meshes_path):
        for f in files:
            if f.endswith('.npz'):
                animation_path = os.path.join(root, f)
                data = dict(np.load(animation_path))
                print(data.keys())
                print(data['poses'].shape)

                # TODO: convert to tensors
                betas = torch.from_numpy(data['betas']).float().cuda()
                poses = torch.from_numpy(data['poses']).float().cuda()

                if len(betas.shape) == 1:
                    betas = betas.unsqueeze(0)
                print(poses[:, 3:3+21*3].shape)

                # batch= [0:150]
                batch= range(150)
                global_orient = poses[batch, :3]
                body_pose = poses[batch, 3:3+21*3]
                jaw_pose = poses[batch, 22*3:23*3]
                leye_pose = poses[batch, 23*3:24*3]
                reye_pose = poses[batch, 24*3:25*3]
                left_hand_pose = poses[batch, 25*3:25*3 + 15*3]
                right_hand_pose = poses[batch, 25*3 + 15*3:25*3 + 15*3 + 15*3]

                # global_orient = global_orient[:, [0, 2, 1]]

                torch.no_grad()
                joints, verts = smplx(betas=betas,  global_orient=global_orient,
                                                    body_pose=body_pose,
                                                    jaw_pose=jaw_pose,
                                                    leye_pose=leye_pose,
                                                    reye_pose=reye_pose,
                                                    left_hand_pose=left_hand_pose,
                                                    right_hand_pose=right_hand_pose)
                
                print("joints: ", joints.shape)
                print("verts: ", verts.shape)
                
                # breakpoint()
                camera_translation = [0.,     0.,      40.]
                # camera_translation = [1.1297, 0.1291, 0.1926]
                # camera_translation = [0.1291,     0.19265,      39.518]

                for ib in batch:
                    img_rendered = renderer(verts[ib].detach().cpu().numpy(), camera_translation, joints=joints[ib].detach().cpu().numpy())
                    print("img_rendered: ", img_rendered.shape)

                    cv2.imshow("smplx", img_rendered)
                    cv2.waitKey(10)

                exit()