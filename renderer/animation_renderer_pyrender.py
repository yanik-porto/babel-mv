from .animation_renderer_joints_2D import AnimationRendererJoints2D
from smpl import SMPLX, SMPL
from tools.renderer import Renderer
import os
import numpy as np
import torch
import cv2
import time
from tools.utils import AverageMeter
from tools.geometry import get_ori_in_cam, find_closest_angle
import math

class AnimationRendererPyrender(AnimationRendererJoints2D):
    def __init__(self, convention='LSP', skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererPyrender, self).__init__(convention, skip_existing, strict_label, n_classes, only_some_actions)

        self.all_oris = []

    def render_animation_in_cameras(self, cams, animation_filename, animation_folder):
        start = time.perf_counter()
        self.render_animation_in_cameras_simultaneously(cams, animation_filename, animation_folder)
        print("total animation : ", time.perf_counter() - start, " sec")
        return

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
        
        st = time.time()
        all_joints, all_verts = self.load_animation(os.path.join(animation_folder, animation_filename))
        animation_loading_time = time.time() - st

        video=cv2.VideoWriter(render_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920,1080))
        # video=cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), 30, (640,480))

        keypoints = []

        assert(camera_name in self.cameras)
        camera_translation = self.cameras[camera_name][0]
        camera_angles = self.cameras[camera_name][1]

        render_time = AverageMeter()
        writer_time = AverageMeter()

        # batch = range(self.poses.shape[0])
        # torch.no_grad()
        for ib, _ in enumerate(all_verts):
            st = time.time()

            joints, verts = all_joints[ib], all_verts[ib]
            # joints, verts = self.joints_from_pose(ib)
                

            # camera_translation = [-2.,     0.,      10.]
            # camera_angles = [0, 0, 180]

            # camera_translation = [-2.,     2.,      10.]
            # camera_angles = [2, 5, 90]

            # camera_angles = [0, 0, 180]
            # camera_translation = [0.,     2.,      10.]
            # camera_angles = [5, 0, 45]
            # camera_rotation = trimesh.transformations.rotation_matrix(
            #     np.radians(camera_angles[2]), [0, 0, 1])
            # camera_rotation = camera_rotation @ trimesh.transformations.rotation_matrix(
            #     np.radians(camera_angles[1]), [0, 1, 0])
            # camera_rotation = camera_rotation @ trimesh.transformations.rotation_matrix(
            #     np.radians(camera_angles[0]), [1, 0, 0])
            # camera_translation = [0.,     0.,      40.]

            # print(verts.shape)
            # print(joints.shape)
            img_rendered = self.renderer(verts, camera_translation, camera_angles)#, joints=joints[0].detach().cpu().numpy())
            render_time.update(time.time() - st)
            st = time.time()
            img_rendered *= 255 # or any coefficient
            img_rendered = img_rendered.astype(np.uint8)
            video.write(img_rendered[:, :, :3])
            writer_time.update(time.time() - st)

            if False:
                cv2.imshow("smplx", img_rendered)
                cv2.waitKey(10)

            keypoints.append(self.renderer.project_joints(joints, camera_translation, camera_angles))

        video.release()
        np.savez(os.path.join(out_folder, stdname + '_0_gt.npz'), keypoint=keypoints)

        print("total time loading : {est_time:.3f}\t sec".format(est_time=animation_loading_time))
        print("avg time render : {est_time.avg:.3f}\t sec".format(est_time=render_time))
        print("avg time writer : {est_time.avg:.3f}\t sec".format(est_time=writer_time))
        print("total time render : {est_time.sum:.3f}\t sec".format(est_time=render_time))
        print("total time writer : {est_time.sum:.3f}\t sec".format(est_time=writer_time))

        print("avg time loading mesh only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.load_mesh_time))
        print("avg time loading scene only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.load_scene_time))
        print("avg time rendering only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.render_time))


    def render_animation_in_cameras_simultaneously(self, cams, animation_filename, animation_folder):
        an_f_noext, _ = os.path.splitext(animation_filename)
        out_folder = os.path.join(animation_folder, an_f_noext)
        os.makedirs(out_folder, exist_ok=True)

        self.renderer.reset_time()

        # create video writer for each cam
        vrws = {}
        for camera_name in cams:
            stdname = self.babel_to_stdname(an_f_noext, camera_name)
            if stdname == "":
                print("Action is not in the list for animation ", animation_filename)
                return
            vrw = VideoRenderingWriter(stdname, out_folder, self.skip_existing, (self.image_width, self.image_height))
            if not vrw.initialized:
                print("No VideoRenderingWriter for animation ", animation_filename)
                return
            vrws[camera_name] = vrw

        # load animation
        st = time.perf_counter()
        all_joints, all_verts = self.load_animation(os.path.join(animation_folder, animation_filename))
        animation_loading_time = time.perf_counter() - st

        print("animation duration : ", len(all_verts) / 30., " sec")
        

        for ib, _ in enumerate(all_verts):

            joints, verts = all_joints[ib], all_verts[ib]

            self.renderer.load_mesh(verts)

            render_time = AverageMeter()
            writer_time = AverageMeter()

            for camera_name in cams:
                assert camera_name in self.cameras, camera_name + " vs " + str(self.cameras)
                camera_translation = self.cameras[camera_name][0]
                camera_angles = self.cameras[camera_name][1]

                camera_pose = self.camera_poses[camera_name]

                st = time.perf_counter()
                img_rendered = self.renderer.render_mesh(camera_pose)
                render_time.update(time.perf_counter() - st)

                st = time.perf_counter()
                img_rendered *= 255 # or any coefficient
                img_rendered = img_rendered.astype(np.uint8)
                vrws[camera_name].write(img_rendered)
                writer_time.update(time.perf_counter() - st)

                vrws[camera_name].add_keypoints(self.renderer.project_joints(joints, camera_translation, camera_angles))

                if ib == 0:
                    vrws[camera_name].define_ori(get_ori_in_cam(joints, camera_angles, convention=self.convention))

        for vrw in vrws.values():
            vrw.close()
                self.all_oris.append(vrws[camera_name].closest_node)

        print("total time loading : {est_time:.3f}\t sec".format(est_time=animation_loading_time))
        print("avg time render : {est_time.avg:.3f}\t sec".format(est_time=render_time))
        print("avg time writer : {est_time.avg:.3f}\t sec".format(est_time=writer_time))
        print("total time render : {est_time.sum:.3f}\t sec".format(est_time=render_time))
        print("total time writer : {est_time.sum:.3f}\t sec".format(est_time=writer_time))

        print("avg time loading camera only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.camerapose_time))
        print("total time loading camera only : {est_time.sum:.3f}\t sec".format(est_time=self.renderer.camerapose_time))
        # print("avg time loading mesh only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.load_mesh_time))
        # print("avg time loading scene only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.load_scene_time))
        # print("avg time rendering only : {est_time.avg:.3f}\t sec".format(est_time=self.renderer.render_time))


class VideoRenderingWriter():
    def __init__(self, stdname, out_folder, skip_existing, image_wh):
        render_file_path = os.path.join(out_folder, stdname + '.avi')
        if os.path.exists(render_file_path) and skip_existing:
            print(render_file_path, " already exists")
            self.initialized = False
            return
        
        self.video=cv2.VideoWriter(render_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, image_wh)
        self.kpts_path = os.path.join(out_folder, stdname + '_0_gt.npz')

        self.keypoints = []
        self.ori = (0, 1) # default orientation is front (sin(0°), cos(0°))
        self.closest_node = 6 # 6th node in the range (-180, 180, 30)
        self.angles_range = list(range(-180, 180, 30))

        self.initialized = True

    def write(self, img_rendered):
        self.video.write(img_rendered[:, :, :3])

    def add_keypoints(self, kpts):
        self.keypoints.append(kpts)

    def define_ori(self, ori):
        self.ori = ori

        ori_deg = math.degrees(math.atan2(self.ori[0], self.ori[1]))
        self.closest_node = find_closest_angle(ori_deg, angles_list=self.angles_range)


    def close(self):
        self.video.release()
        np.savez(self.kpts_path, keypoint=self.keypoints, orientation=self.ori, closest_node=self.closest_node)
        