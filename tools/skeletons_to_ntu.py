import argparse
import os
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Compress a list of skeleton files into a ntu dataset format")
    parser.add_argument("in_path", type=str, help="Path to input folder")
    parser.add_argument("--out_filename", type=str, default="ntu_custom.pkl", help="Name of the output file")
    parser.add_argument("--unknown_label", action="store_true", default=False, help="set if the label is unknown")
    parser.add_argument("--verbose", action="store_true", default=False, help="set if need to print more information")
    parser.add_argument("--only_suffix", type=str, default="_0.npz", help="Set if only files with the given suffix have to be taken into account")
    parser.add_argument("--train_folder", type=str, default="train", help="Name of the training folder")
    parser.add_argument("--val_folder", type=str, default="val", help="Name of the validation folder")
    parser.add_argument("--append", action="store_true", default=False, help="if set, append sequences to existing output file")
    return parser.parse_args()

def fill_split(args, ntu_format, folder_path, split_name="xsub_val"):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(args.only_suffix):
                if args.verbose:
                    print(f)
                skel_path = os.path.join(root, f)
                skel = dict(np.load(skel_path))

                if "keypoint" not in skel:
                    continue

                annot = {}
                keypoint = skel["keypoint"]
                if len(keypoint.shape) == 3:
                    keypoint = np.expand_dims(keypoint, axis=0)
                annot["keypoint"] = keypoint

                if "keypoint_score" in skel:
                    keypoint_score = skel["keypoint_score"]
                    if len(keypoint_score.shape) == 2:
                        keypoint_score = np.expand_dims(keypoint_score, axis=0)
                    annot["keypoint_score"] = keypoint_score

                frame_dir = f.split(".")[0]
                if args.unknown_label:
                    annot["label"] = 0
                else:
                    splits = frame_dir.split("_")
                    if len(splits) == 1:
                        frame_dir = frame_dir[0]
                    elif len(splits) > 1:
                        frame_dir = frame_dir[:-2]
                    annot["label"] = int(frame_dir[-3:]) - 1
                
                annot["frame_dir"] = frame_dir
                annot["img_shape"] = (1080, 1920)
                annot["original_shape"] = (1080, 1920)
                annot["total_frames"] = keypoint.shape[1]

                ntu_format["annotations"].append(annot)
                ntu_format["split"][split_name].append(frame_dir)

    seq_len = len(ntu_format["split"][split_name])
    print(f"Export {seq_len} sequences in {split_name}")

def append_to_existing(new_data, old_data, split, args):
    print("before appening, existing data ", split, " contains : ", len(old_data["split"][split]))
    for frame_dir in new_data["split"][split]:
        if frame_dir in old_data["split"][split]:
            if args.verbose:
                print(frame_dir, " already in existing data")
            continue
        annot = next((a for a in new_data["annotations"] if a["frame_dir"] == frame_dir), None)
        if annot is None:
            print("weired")
            continue
        old_data["split"][split].append(frame_dir)
        old_data["annotations"].append(annot)

    print("after appening, existing data ", split, " contains : ", len(old_data["split"][split]))

if __name__ == '__main__':
    args = parse_args()

    ntu_format = {}
    ntu_format["split"] = {}
    ntu_format["split"]["xsub_val"] = []
    ntu_format["split"]["xsub_train"] = []
    ntu_format["annotations"] = []

    train_folder = os.path.join(args.in_path, args.train_folder)
    val_folder = os.path.join(args.in_path, args.val_folder)

    if os.path.isdir(train_folder):
        fill_split(args, ntu_format, train_folder, "xsub_train")
    if os.path.isdir(val_folder):
        fill_split(args, ntu_format, val_folder, "xsub_val")

    out_path = os.path.join(args.in_path, args.out_filename)
    if os.path.exists(out_path):
        if not args.append:
            print("Output file already exists!")
            print("Set arg --append if data has to be append to existing file")
            exit()
        with open(out_path, "rb") as outf:
            existing_data = pickle.load(outf)
            append_to_existing(ntu_format, existing_data, "xsub_val", args)
            append_to_existing(ntu_format, existing_data, "xsub_train", args)
        with open(out_path, "wb") as outf:
            pickle.dump(existing_data, outf)
    else:
        with open(out_path, "wb") as outf:
            pickle.dump(ntu_format, outf)
        