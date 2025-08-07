import numpy as np
import trimesh

def add_camera_mesh(extrinsic, camerascale=1):
    # 8 points camera
    r = np.zeros((3,4,3))

    r[0][0] = np.array([-0.5, 0.5, 0]) * camerascale
    r[0][1] = np.array([0.5, 0.5, 0]) * camerascale
    r[0][2] = np.array([0.5, -0.5, 0]) * camerascale
    r[0][3] = np.array([-0.5, -0.5, 0]) * camerascale

    r[1][0] = np.array([-1, 1, 1]) * camerascale
    r[1][1] = np.array([1, 1, 1]) * camerascale
    r[1][2] = np.array([1, -1, 1]) * camerascale
    r[1][3] = np.array([-1, -1, 1]) * camerascale

    r[2][0] = np.array([-0.5, 0.5, -2]) * camerascale
    r[2][1] = np.array([0.5, 0.5, -2]) * camerascale
    r[2][2] = np.array([0.5, -0.5, -2]) * camerascale
    r[2][3] = np.array([-0.5, -0.5, -2]) * camerascale

    P = np.zeros((3, 40))
    for i in range(3):
        P[:,i * 8 + 0] = r[i][0] 
        P[:,i * 8 + 1] = r[i][1]
        P[:,i * 8 + 2] = r[i][1] 
        P[:,i * 8 + 3] = r[i][2]
        P[:,i * 8 + 4] = r[i][2] 
        P[:,i * 8 + 5] = r[i][3]
        P[:,i * 8 + 6] = r[i][3] 
        P[:,i * 8 + 7] = r[i][0]

    for i in range(2):
        P[:,24 + i * 8 + 0] = r[0][0] 
        P[:,24 + i * 8 + 1] = r[i + 1][0]
        P[:,24 + i * 8 + 2] = r[0][1] 
        P[:,24 + i * 8 + 3] = r[i + 1][1]
        P[:,24 + i * 8 + 4] = r[0][2] 
        P[:,24 + i * 8 + 5] = r[i + 1][2]
        P[:,24 + i * 8 + 6] = r[0][3] 
        P[:,24 + i * 8 + 7] = r[i + 1][3]

    # // transform from camera space to object space
    # // this step is critical for visualizing the cameras since our viewpoint is in the object space
    M = np.linalg.inv(extrinsic)
    for i in range(P.shape[1]):
        t = np.ones((4,))
        t[:3] = P[:,i]
        p = np.dot(M, t)
        P[:,i] = p[:3] / p[3]

    return P

def add_camera_mesh_like_blender(extrinsic, camerascale=1, doinverse=True):
    # 8 points camera
    r = np.zeros((3,4,3))

    r[0][0] = np.array([-0.5, 0.5, 0]) * camerascale
    r[0][1] = np.array([0.5, 0.5, 0]) * camerascale
    r[0][2] = np.array([0.5, -0.5, 0]) * camerascale
    r[0][3] = np.array([-0.5, -0.5, 0]) * camerascale

    r[1][0] = np.array([-1, 1.8, -1]) * camerascale
    r[1][1] = np.array([1, 1.8, -1]) * camerascale
    r[1][2] = np.array([1, -1, -1]) * camerascale
    r[1][3] = np.array([-1, -1, -1]) * camerascale

    r[2][0] = np.array([-0.5, 0.5, 2]) * camerascale
    r[2][1] = np.array([0.5, 0.5, 2]) * camerascale
    r[2][2] = np.array([0.5, -0.5, 2]) * camerascale
    r[2][3] = np.array([-0.5, -0.5, 2]) * camerascale

    P = np.zeros((3, 40))
    for i in range(3):
        P[:,i * 8 + 0] = r[i][0] 
        P[:,i * 8 + 1] = r[i][1]
        P[:,i * 8 + 2] = r[i][1] 
        P[:,i * 8 + 3] = r[i][2]
        P[:,i * 8 + 4] = r[i][2] 
        P[:,i * 8 + 5] = r[i][3]
        P[:,i * 8 + 6] = r[i][3] 
        P[:,i * 8 + 7] = r[i][0]

    for i in range(2):
        P[:,24 + i * 8 + 0] = r[0][0] 
        P[:,24 + i * 8 + 1] = r[i + 1][0]
        P[:,24 + i * 8 + 2] = r[0][1] 
        P[:,24 + i * 8 + 3] = r[i + 1][1]
        P[:,24 + i * 8 + 4] = r[0][2] 
        P[:,24 + i * 8 + 5] = r[i + 1][2]
        P[:,24 + i * 8 + 6] = r[0][3] 
        P[:,24 + i * 8 + 7] = r[i + 1][3]

    # // transform from camera space to object space
    # // this step is critical for visualizing the cameras since our viewpoint is in the object space
    M = np.linalg.inv(extrinsic) if doinverse else extrinsic
    for i in range(P.shape[1]):
        t = np.ones((4,))
        t[:3] = P[:,i]
        p = np.dot(M, t)
        P[:,i] = p[:3] / p[3]

    return P

def load_camera_para(file):
    """"
    load camera parameters
    """
    campose = []
    intra = []
    campose_ = []
    intra_ = []
    f = open(file,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if len(words) == 3:
            intra_.append([float(words[0]),float(words[1]),float(words[2])])
        elif len(words) == 4:
            campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
        else:
            pass

    index = 0
    intra_t = []
    for i in intra_:
        index+=1
        intra_t.append(i)
        if index == 3:
            index = 0
            intra.append(intra_t)
            intra_t = []

    index = 0
    campose_t = []
    for i in campose_:
        index+=1
        campose_t.append(i)
        if index == 3:
            index = 0
            campose_t.append([0.,0.,0.,1.])
            campose.append(campose_t)
            campose_t = []
    
    return np.array(campose), np.array(intra)

def create_cube_mesh(xyz):
    # Define the vertices of the cube
    vertices = np.array([
        [-1, -1, -1], # 0
        [ 1, -1, -1], # 1
        [ 1,  1, -1], # 2
        [-1,  1, -1], # 3
        [-1, -1,  1], # 4
        [ 1, -1,  1], # 5
        [ 1,  1,  1], # 6
        [-1,  1,  1]  # 7
    ], dtype=float)

    vertices *= 0.05

    # Calculate the current center of the cube
    current_center = np.mean(vertices, axis=0)

    desired_center = np.array(xyz)

    # Calculate the offset to move the current center to the desired center
    offset = desired_center - current_center

    vertices += offset
    # for i in range(len(vertices)):
    #     vertices[i] += offset

    # Define the faces of the cube
    faces = np.array([
        [0, 1, 2], [2, 3, 0], # Bottom
        [4, 5, 6], [6, 7, 4], # Top
        [0, 1, 5], [5, 4, 0], # Front
        [1, 2, 6], [6, 5, 1], # Right
        [2, 3, 7], [7, 6, 2], # Back
        [3, 0, 4], [4, 7, 3]  # Left
    ])

    # Create a trimesh object for the cube
    cube = trimesh.Trimesh(vertices=vertices, faces=faces)

    return cube