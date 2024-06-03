import argparse
import os
import pickle
import sys
sys.path.insert(0, os.getcwd())
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/babel_mv_0.1.pkl", help="Path to the data file containing 2d skeletons.")
    parser.add_argument("--out_path", type=str, default="data/babel_mv_0.1_by_views.pkl", help="Path to the output data where 2d skeletons will be grouped by views.")
    parser.add_argument("--only_print", action="store_true", default=False, help="If set, print dataset information, without grouping data and saving it")
    parser.add_argument("--do_not_equalize_views", action="store_true", default=False, help="If set, do not delete sequences with missing views")
    parser.add_argument("--only_cam", type=str, default=None, choices=[None, 'C001', 'C002', 'C003'], help="The uinique camera to keep")
    opts = parser.parse_args()
    return opts

def split_dataname(dataname):
    splits = dataname.split('_')
    idx = splits[0]
    uid = splits[1]
    c = splits[2]
    a = splits[3]
    return idx, uid, c, a

def parse_data(dataset):
    idxset = set()
    uidset = set()
    cset = set()
    aset = set()

    n = len(dataset)
    for i in range(n):
        idx, uid, c, a = split_dataname(dataset[i])
        idxset.add(idx)
        uidset.add(uid)
        cset.add(c)
        aset.add(a)

    return idxset, uidset, cset, aset

def print_split(datasplit, splitName):
    trainName = splitName + '_train'
    valName = splitName + '_val'
    print("********************************")
    print(splitName)
    print("train:")
    print("    ", "n = ", len(datasplit[trainName]))
    idxset, uidset, cset, aset = parse_data(datasplit[trainName])
    print("    ", 'idxset: %s, uidset: %s, cset: %s, aset: %s' % (len(idxset), len(uidset), len(cset), len(aset)))

    print(cset)

    print("val:")
    print("    ", "n = ", len(datasplit[valName]))
    idxset, uidset, cset, aset = parse_data(datasplit[valName])
    print("    ", 'idxset: %s, uidset: %s, cset: %s, aset: %s' % (len(idxset), len(uidset), len(cset), len(aset)))
    print("********************************")

    print(cset)

def get_group_from_name(name):
    splits = name.split('_')
    return splits[0] + '_' + splits[1] + '_' + splits[3]

def regroup_views(alldata):
    datagrouped = {}
    for data in alldata:
        group = get_group_from_name(data)
        if group not in datagrouped:
            datagrouped[group] = set()
        datagrouped[group].add(data)
    return datagrouped

def keep_only_one_view(alldata, view):
    datatokeep = []
    for data in alldata:
        _, _, c, _ = split_dataname(data)
        if c == view:
            datatokeep.append(data)
    return datatokeep

def remove_seq_with_missing_views(groupedData):
    maxcams = 0
    for keyg in groupedData:
        ncams = len(groupedData[keyg])
        if ncams > maxcams:
            maxcams = ncams

    filteredGroups = {}
    for keyg in groupedData:
        ncams = len(groupedData[keyg])
        if ncams == maxcams:
            filteredGroups[keyg] = groupedData[keyg]

    return filteredGroups

if __name__ == '__main__':
    opts = parse_args()
    with open (opts.file_path, 'rb') as f: data = pickle.load(f)

    print(data['split'].keys())
    print(data['annotations'][0].keys())
    print(data['annotations'][0]["frame_dir"])
    print(data['annotations'][0]["total_frames"])
    print(data['annotations'][0]["keypoint"].shape)
    print(data['annotations'][0]["keypoint_score"].shape)
    print(data['annotations'][0]["img_shape"])
    print(data['annotations'][0]["original_shape"])

    print_split(data['split'], 'xsub')

    if not opts.only_print:

        splitTrain = None
        splitVal = None

        if opts.only_cam is None:
            print("Group data by views")
            datagroupedTrain = regroup_views(data["split"]["xsub_train"])
            datagroupedVal = regroup_views(data["split"]["xsub_val"])

            if not opts.do_not_equalize_views:
                datagroupedTrain = remove_seq_with_missing_views(datagroupedTrain)
                datagroupedVal = remove_seq_with_missing_views(datagroupedVal)

            splitTrain = datagroupedTrain
            splitVal = datagroupedVal
            print(len(datagroupedTrain), " groups in new xsub train")
            print(len(datagroupedVal), "groups in new xsub val")

        else:
            print("Keep only view #", opts.only_cam)
            splitTrain = keep_only_one_view(data["split"]["xsub_train"], opts.only_cam)
            splitVal = keep_only_one_view(data["split"]["xsub_val"], opts.only_cam)
            print(len(data["split"]["xsub_train"]), " => ", len(splitTrain))
            print(len(data["split"]["xsub_val"]), " => ", len(splitVal))

        
        # splitTrain = data["split"]["xsub_train"][:20]
        # splitVal = data["split"]["xsub_val"][:10]


        newdict = {"split": {"xsub_train": splitTrain, "xsub_val": splitVal}, "annotations": data["annotations"]}
        with open(opts.out_path, 'wb') as f:
            pickle.dump(newdict, f)