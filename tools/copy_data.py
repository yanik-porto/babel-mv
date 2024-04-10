import argparse
import os
import glob
import shutil
from tqdm import tqdm

subset = [15, 16, 19, 21, 23, 26, 30, 31, 37, 52, 62, 64]

map_what = {"renders": "*/*.avi", "meshs": "*.npz", "projections": "*/*.npz"}

def parse_args():
    parser = argparse.ArgumentParser(description="Copy required data files")
    parser.add_argument('src', type=str, help="Path to source files to copy")
    parser.add_argument('dst', type=str, help="Path to destination folder where copying")
    parser.add_argument('--what', nargs='+', type=str, default='renders', help="suffixes to copy")
    parser.add_argument('--only_few', action='store_true', default=False, help="copy only few actions")
    return parser.parse_args()

def get_action_from_path(filepath, action_position=-1):
    mnoext, _ = os.path.splitext(m)
    words = os.path.basename(mnoext).split('_')
    action_str = words[action_position]
    action_id = int(action_str[1:]) - 1
    return action_id

if __name__ == '__main__':

    args = parse_args()

    suffixes = [map_what[w] for w in args.what]

    for suf in suffixes:
        print("copy all ", suf)
        matches = glob.glob(os.path.join(args.src, suf))
        
        for m in tqdm(matches):
            if args.only_few:
                action_id = None
                if suf == "*/*.avi":
                    action_id = get_action_from_path(m, -1)
                elif suf == "*/*.npz":
                    action_id = get_action_from_path(m, -2)
                elif suf == "*.npz":
                    print("not implemented yet!")
                    continue

                if action_id not in subset:
                    continue

            dst = m.replace(args.src, args.dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(m, dst)
