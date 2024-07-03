import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

# import sys
# sys.path.append(os.getcwd())
# print(sys.path)
# from renderer.map_to_babel import ntu2babel_few_few

ntu2babel_few_few = {
    6: 23, #throw
    # 7: 18, #sitting down
    8: 26, #standing up (from sitting position)
    9: 64, #clapping
    23: 31, #kicking something
    26: 16, #jump up
    50: 31, #kicking other person
    63: 30, #bounce ball
    79: 52, #squat down
    91: 37, #lift something
    92: 62, #shake fist
    98: 21, #running on the spot
    99: 31, #butt kicks (kick backward)
    101: 31, #side kick
    103: 15 #stretch oneself
}

def parse_args():
    parser = argparse.ArgumentParser(description="Display plots on the database")
    parser.add_argument("poses_path", type=str, help="Path to the file containing the poses")
    parser.add_argument("--ntu_path", type=str, help="Path to the ntu dataset")
    return parser.parse_args()

def get_dur_by_action(poses_path, max_dur=200):
    dur_by_action = {}

    with open(poses_path, 'rb') as f:
        data = pickle.load(f)

        for ann in data['annotations']:
            seq = ann["frame_dir"]
            actidx = int(seq[-3:]) - 1
            duration = ann['total_frames']

            if not actidx in dur_by_action:
                dur_by_action[actidx] = np.zeros(max_dur // 10)
            
            if duration >= max_dur:
                continue

            dur_by_action[actidx][duration // 10] += 1
    return dur_by_action

def dur_by_action_to_plot(dur_by_action, labels, figname):
    fig, ax = plt.subplots()
    bottom = np.zeros(max_dur // 10)
    for action, durs in dur_by_action.items():
        ax.bar(range(0, max_dur, 10), durs, 5, label=labels[action] + " (" + str(int(sum(durs))) + ")", bottom=bottom)
        bottom += durs

    ax.set_title("duration by action")
    ax.legend(loc="upper right")
    plt.savefig(figname + '.png')

if __name__ == "__main__":
    args = parse_args()

    max_dur = 250
    dur_by_action = get_dur_by_action(args.poses_path, max_dur)

    # plot durations in babel_mv
    labels = [x.strip() for x in open("tools/babel.txt").readlines()]
    dur_by_action_to_plot(dur_by_action, labels, "duration_babel_mv")

    # plot durations in ntu
    if args.ntu_path:
        dur_ntu = get_dur_by_action(args.ntu_path, max_dur)
        dur_ntu = {key : val for key,val in dur_ntu.items() if key in ntu2babel_few_few.keys()}
        labels_ntu = [x.strip() for x in open("tools/nturgbd_120.txt").readlines()]
        dur_by_action_to_plot(dur_ntu, labels_ntu, "duration_ntu")