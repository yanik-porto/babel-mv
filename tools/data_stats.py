import argparse
import os
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Display statistics on the database")
    parser.add_argument("poses_path", type=str, help="Path to the file containing the poses")
    parser.add_argument("--babel_path", type=str, default=None, help="Path to the babel dataset with split files, if comparison is needed")
    return parser.parse_args()

def compute_babel_split_statistics(babel_split):
    nseqbyaction = {}
    for sid in babel_split:
        sid_descr = babel_split[sid]
        annotation = sid_descr['frame_ann'] if sid_descr['frame_ann'] is not None else babel_split[sid]['seq_ann']
        if annotation is None:
            print("No annotation for sid %s" % sid)
            continue
        for label in annotation["labels"]:
            cats = label['act_cat']
            for cat in cats:
                if cat not in nseqbyaction:
                    nseqbyaction[cat] = 0
                nseqbyaction[cat] += 1
    return nseqbyaction

def compute_babel_statistics(babel_path):
    babel_train_path = os.path.join(babel_path, "train.json")
    babel_val_path = os.path.join(babel_path, "val.json")
    with open(babel_train_path) as file: babel_train = json.load(file)
    with open(babel_val_path) as file: babel_val = json.load(file)
    nseqbyaction_train = compute_babel_split_statistics(babel_train)
    nseqbyaction_val = compute_babel_split_statistics(babel_val)
    return {'train': nseqbyaction_train, 'val': nseqbyaction_val}

def print_statsbysplit(statsbysplit, babel_stats=None):
    for split, nseqbyaction in statsbysplit.items():
        print("number of actions:", len(nseqbyaction))
        full_babel_split = None
        if babel_stats is not None:
            for babel_split in babel_stats:
                if babel_split in split:
                    full_babel_split = babel_stats[babel_split]

        for action, n in nseqbyaction.items():
            total = "?"
            if full_babel_split is not None:
                if action in full_babel_split:
                    total = str(full_babel_split[action]) + " ( " + str(int(n / full_babel_split[action] * 100)) + "%)"
            print(action, " : ", n, " / ", total)

if __name__ == "__main__":
    args = parse_args()

    labels = [x.strip() for x in open("tools/babel.txt").readlines()]

    babel_stats = None
    if args.babel_path is not None:
        babel_stats = compute_babel_statistics(args.babel_path)

    with open(args.poses_path, 'rb') as f:
        data = pickle.load(f)

        statsbysplit = {}
        for split, seqs in data['split'].items():
            nseqbyaction = {}   
            print(split, " (", len(seqs), " sequences)")
            for seq in seqs:
                actidx = int(seq[-3:]) - 1
                action = labels[actidx]
                if not action in nseqbyaction:
                    nseqbyaction[action] = 0
                nseqbyaction[action] += 1
            statsbysplit[split] = nseqbyaction

        print("***BABEL_MV***")
        print_statsbysplit(statsbysplit, babel_stats)
