import argparse
import glob
import os
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="visualize estimated pose and gt on video source")
    parser.add_argument('folder', type=str, help="path to the folder containing the input files to visualize")
    parser.add_argument('--save', action='store_true', default=False, help="save output to file")
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

    videos = glob.glob(os.path.join(args.folder, '*.avi'))

    for video in videos:
        vidnoext, _ = os.path.splitext(video)

        # load estimatied pose
        estim = None
        estim_path = vidnoext + '_0.npz'
        if os.path.exists(estim_path):
            estim = dict(np.load(estim_path))
            print(estim['keypoint'].shape)

        # load gt pose
        gt = None
        gt_path = vidnoext + '_0_gt.npz'
        if os.path.exists(gt_path):
            gt = dict(np.load(gt_path))
            print(gt['keypoint'].shape)

        cap = cv2.VideoCapture(video)
        assert(cap.isOpened())

        if args.save:
            video_out = cv2.VideoWriter(os.path.join("output", os.path.basename(video)), cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920,1080))

        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("video is over")
                break

            estim_kpt = None
            gt_kpt = None

            if estim is not None:
                if estim['keypoint'].shape[0] > i:
                    estim_kpt = estim['keypoint'][i, :, :]
                    frame = add_estimation(frame, estim_kpt)

            if gt is not None:
                if gt['keypoint'].shape[0] > i:
                    gt_kpt = gt['keypoint'][i, :, :]
                    frame = add_gt(frame, gt_kpt)

            if gt_kpt is not None and estim_kpt is not None:
                mpjpe = np.mean(np.sum(np.square(estim_kpt - gt_kpt), axis=-1))
                frame = cv2.putText(frame, "mpjpe : " + str(int(mpjpe)), (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("frame", frame)
            cv2.waitKey(10)

            if args.save:
                video_out.write(frame)

            i=i+1

        if args.save:
            video_out.release()

        cap.release()
        cv2.destroyAllWindows()