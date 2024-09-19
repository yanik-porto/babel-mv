# from renderer.animation_renderer_blender import *
from renderer.animation_renderer_pyrender import *
from renderer.animation_renderer_joints_3D import *
from renderer.animation_renderer_joints_2D import *

def create_renderer(method, convention="LSP"):
    renderer = None

    if method == 'pyrender':
        renderer = AnimationRendererPyrender(convention)
    # elif method == 'blender':
        # renderer = AnimationRendererBlender()
    if method == 'joints3d':
        renderer = AnimationRendererJoints3D(convention)
    elif method == 'joints2d':
        renderer = AnimationRendererJoints2D(convention)

    return renderer