import sys
import os
from os.path import dirname as ospd
sys.path.append(ospd(ospd(os.path.abspath(__file__))))
import bpy

from renderer.animation_renderer import AnimationRenderer

def remove_default_cube():
# Check if the cube exists in the data
    if "Cube" in bpy.data.objects:
        cube_object = bpy.data.objects["Cube"]
        bpy.data.objects.remove(cube_object, do_unlink=True)

def remove_all_objects():
    for key in bpy.data.objects:
        print(key.name)
        if not "camera" in key.name.lower() and not "light" in key.name.lower():
            bpy.data.objects.remove(bpy.data.objects[key.name], do_unlink=True)

def remove_all_armatures():
    for key in bpy.data.armatures:
        bpy.data.armatures.remove(bpy.data.armatures[key.name], do_unlink=True)

def remove_all_meshes():
    for key in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[key.name], do_unlink=True)
    
def remove_all_materials():
    for key in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[key.name], do_unlink=True)

def remove_all():
    remove_all_objects()
    remove_all_meshes()
    remove_all_armatures()
    remove_all_materials()

def reset_animations():
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0


class AnimationRendererBlender(AnimationRenderer):
    def __init__(self, skip_existing = False, strict_label = False, n_classes = 120, only_some_actions = False):
        super(AnimationRendererBlender, self).__init__(skip_existing, strict_label, n_classes, only_some_actions)

    def load_animation(self, animation_path):
        if animation_path == self.animation_loaded:
            return
        
        bpy.ops.object.smplx_add_animation(filepath=animation_path)
        self.animation_loaded = animation_path

    def clear(self):
        remove_all()
        reset_animations()
        renderer.animation_loaded = ""

    def render_animation_in_camera(self, camera_name, animation_filename, animation_folder):
        scene = bpy.context.scene
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.image_settings.file_format = "AVI_JPEG"

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
        
        # load animation if not done yet
        self.load_animation(os.path.join(animation_folder, animation_filename))

        scene.render.filepath = os.path.join(out_folder, stdname + '.avi')

        my_camera = bpy.data.objects.get(camera_name)
        if my_camera:
            bpy.context.scene.camera = my_camera
            print(f"Active camera set to {my_camera.name}")
        else:
            print(f"Camera {camera_name} not found in the scene.")
            return

        # Render the current frame
        bpy.ops.render.render(write_still=False, animation=True)




if __name__ == '__main__':
    remove_default_cube()

    an_folder = "/home/yanik/repos/BABEL/action_recognition/data/babel_v1.0_smplx_split/val/"

    renderer = AnimationRendererBlender(skip_existing=True, only_some_actions=True)

    for root, _, files in os.walk(an_folder):
        for an_f in files:
            if an_f.endswith(("_transition.npz", "_misc. activities.npz", "_misc. action.npz", "_inward motion.npz")):
                continue

            if an_f.endswith((".npz")):

                renderer.render_animation(root, an_f, ['Camera1', 'Camera2', 'Camera3'])

    print("missing labels : ", renderer.missing_labels)
