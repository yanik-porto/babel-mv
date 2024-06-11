[![Python Package using Conda](https://github.com/yanik-porto/babel-mv/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/yanik-porto/babel-mv/actions/workflows/python-package-conda.yml)

# babel-mv
babel dataset for multiview operations

<img src="doc/mygif_babel_vid.gif" width="320" height="240" />

## Data
12 classes : [Download](https://www.dropbox.com/scl/fi/jc97uk7fszw1k1mmoi3vo/babel_mv_0.2.pkl?rlkey=uns5suu5p5lpd8b2mf5w1gy10&st=0rco7rnh&dl=0)

## Generation

### Blender

Install smplx plugin on blender

Render sequences with :

`blender --python babel-mv/renderer/animation_renderer_blender.py`

<img src="doc/blender_cam1.gif" width="320" height="240" /> 
<img src="doc/blender_cam2.gif" width="320" height="240" />
<img src="doc/blender_cam3.gif" width="320" height="240" />

### Pyrender

`python render.py /home/yanik/Documents/datasets/BABEL_MV/kick_pyrender --convention='COCO' `

<img src="doc/pyrender_cam1.gif" width="320" height="240" /> 
<img src="doc/pyrender_cam2.gif" width="320" height="240" />
<img src="doc/pyrender_cam3.gif" width="320" height="240" />

### visualize

Visualize video with estimated humand pose and ground truth

`python tools/visualize.py <RENDER_FOLDER_PATH>`