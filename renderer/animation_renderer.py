from renderer.map_to_babel import few_babel
from abc import ABC, abstractmethod
import json
import os
from os.path import join as ospj
from os.path import dirname as ospd

class AnimationRenderer(ABC):
    def __init__(self, skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        self.skip_existing = skip_existing
        self.strict_label = strict_label
        self.n_classes = n_classes
        self.only_some_actions = only_some_actions

        self.labels_2_idx = {}
        with open(ospj(ospd(os.path.abspath(__file__)), "action_label_2_idx.json")) as infile:
            self.labels_2_idx = json.load(infile)

        self.missing_labels = []

        self.animation_loaded = ""
        
    def get_classidx_from_filename(self, filename):
        splits = filename.split("_")
        assert len(splits) == 3, str(len(splits))
        label = splits[2]

        if self.strict_label:
            assert label in self.labels_2_idx, label

        if not label in self.labels_2_idx:
            self.missing_labels.append(label)
            return -1

        labelidx = self.labels_2_idx[label]
        return labelidx

    def babel_to_stdname(self, filename, camera_name):

        splits = filename.split("_")
        assert len(splits) == 3, str(len(splits))

        labelidx = self.get_classidx_from_filename(filename)
        if labelidx == -1:
            return ""
        
        if self.only_some_actions:
            if labelidx not in few_babel:
                return ""
        
        labelStr = "A" + str(labelidx + 1).rjust(3, '0') # +1 to correspond to NTU notation

        return splits[0] + "_" + splits[1] + "_" + camera_name + "_" + labelStr  
        
    @abstractmethod
    def load_animation(self, animation_path):
        pass

    @abstractmethod
    def render_animation_in_camera(self, camera_name, animation_filename, animation_folder):
        pass
    
    @abstractmethod
    def clear(self):
        pass

    def render_animation(self, animation_folder, animation_filename, cams):
        self.clear()

        classidx = self.get_classidx_from_filename(os.path.splitext(animation_filename)[0])
        if classidx >= self.n_classes:
            print("Skipped animation rendering because class index is out of range : {}".format(classidx))
            return

        print("Render animation {}".format(animation_filename))

        # TODO : list all cameras in the scene
        for cam in cams:
            self.render_animation_in_camera(cam, animation_filename, animation_folder)

        self.clear()
