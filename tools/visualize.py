import argparse
import glob
import os
import numpy as np
import cv2
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="visualize estimated pose and gt on video source")
    parser.add_argument('folder', type=str, help="path to the folder containing the input files to visualize")
    parser.add_argument('--save', action='store_true', default=False, help="save output to file")
    parser.add_argument('--from_single_file', required=False, default=None, help='single file containing the poses for all videos')
    parser.add_argument('--estim_id', type=int, default=0, required=False, help="id of the estimated pose")
    return parser.parse_args()

def add_keypoints(image, keypoints, color):
    imgOverlay = image.copy()
    for i, kpt in enumerate(keypoints):
        kptInt = kpt.astype('int32')
        cv2.circle(imgOverlay, kptInt, radius=2, color=color, thickness=3)
    return imgOverlay

def add_estimation(image, keypoints):
    return add_keypoints(image, keypoints, (255, 0, 0))

def add_gt(image, keypoints):
    return add_keypoints(image, keypoints, (0, 255, 0))

if __name__ == '__main__':
    args = parse_args()

    estimations = None
    if args.from_single_file is not None:
        print("loading from single file : ", args.from_single_file, " ...")
        with open(args.from_single_file, 'rb') as f:
            estimations = pickle.load(f)['annotations']
        print("...done loading")

    videos = glob.glob(os.path.join(args.folder, '*.avi'))

    for video in videos:
        vidnoext, _ = os.path.splitext(video)

        if estimations is not None:
            # load estimated pose from loaded data
            estim = next(ann for ann in estimations if ann['frame_dir'] == os.path.basename(vidnoext))
            gt = None
        else:
            # load estimated pose from a file joint to the video
            estim = None
            estim_path = vidnoext + '_' + str(args.estim_id) + '.npz'
            if os.path.exists(estim_path):
                estim = dict(np.load(estim_path))
                print(estim['keypoint'].shape)

            # load gt pose
            gt = None
            gt_path = vidnoext + '_' + str(args.estim_id) + '_gt.npz'
            if os.path.exists(gt_path):
                gt = dict(np.load(gt_path))
                print(gt['keypoint'].shape)

        # prepare video streamer
        cap = cv2.VideoCapture(video)
        assert(cap.isOpened())

        # prepare video file
        if args.save:
            video_out = cv2.VideoWriter(os.path.join("output", os.path.basename(video)), cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920,1080))

        # iterate over video frames
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("video is over")
                break

            estim_kpt = None
            gt_kpt = None

            # display estimated pose if exists
            if estim is not None:
                estim_keypoints = estim['keypoint'] if len(estim['keypoint'].shape) == 3 else estim['keypoint'][0] # keep first character if more than one
                if estim_keypoints.shape[0] > i:
                    estim_kpt = estim_keypoints[i, :, :]
                    frame = add_estimation(frame, estim_kpt)

            # display gt pose if exists
            if gt is not None:
                if gt['keypoint'].shape[0] > i:
                    gt_kpt = gt['keypoint'][i, :, :]
                    frame = add_gt(frame, gt_kpt)

            # compute error between estimation and gt pose
            if gt_kpt is not None and estim_kpt is not None:
                mpjpe = np.mean(np.sum(np.square(estim_kpt - gt_kpt), axis=-1))
                frame = cv2.putText(frame, "mpjpe : " + str(int(mpjpe)), (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("frame", frame)
            cv2.waitKey(100)
            if args.save:
                video_out.write(frame)

            i=i+1

        if args.save:
            video_out.release()

        cap.release()
        cv2.destroyAllWindows()