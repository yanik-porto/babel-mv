import numpy as np
import math
def vector_mesh_intersection(vector_start, vector_end, mesh_vertices, mesh_faces):
    """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    Checks if a vector defined by a start and end point intersects with a 3D mesh.
    
    Parameters:
    vector_start (numpy.ndarray): 3D starting point of the vector
    vector_end (numpy.ndarray): 3D ending point of the vector
    mesh_vertices (numpy.ndarray): 3D coordinates of the mesh vertices
    mesh_faces (numpy.ndarray): Indices of the vertices that form each triangle face
    
    Returns:
    bool: True if the vector intersects the mesh, False otherwise"""

    # Calculate the vector direction
    vector_direction = vector_end - vector_start
    
    # Loop through each triangle face in the mesh
    for face in mesh_faces:
        v0 = mesh_vertices[face[0]]
        v1 = mesh_vertices[face[1]]
        v2 = mesh_vertices[face[2]]

        # Calculate edges and normal of the triangle
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)

        # Calculate the determinant
        det = np.dot(-vector_direction, normal)

        # If the vector is parallel to the triangle, there is no intersection
        if np.abs(det) < 1e-8:
            continue

        inv_det = 1.0 / det
        p = vector_start - v0

        # Calculate the distance to the intersection point
        t = np.dot(normal, p) * inv_det
        if t < 0.0 or t > 1.0:
            continue
        
        u0 = np.cross(e2, -vector_direction)
        u = np.dot(u0, p) * inv_det
        v0 = np.cross(-vector_direction, e1)
        v = np.dot(v0, p) * inv_det

        # check if the interesection point is in the parallelogram formed by v0 and vectors v01 and v02
        if u < 0.0 or u > 1.0 or v < 0.0 or u + v > 1.0:
            continue

        # check if intersection point is outside the triangle 
        if u + v < 0.0 or u + v > 1.0:
            continue

        return True

    return False

def create_realistic_mask(joints, cam_trans):
    rshould = 6
    lshould = 5
    rhip = 12
    lhip = 11
    body_joints = (6, 5, 12, 11)

    for jp, joints_person in enumerate(joints):
        for jf, joints_frame in enumerate(joints_person):
            vertices = joints_frame
            faces = ((rshould, lshould, lhip), (lhip, rhip, rshould))

            mask = np.ones(joints_frame.shape, dtype=joints_frame.dtype)
            for j in range(len(joints_frame)):
                if j in body_joints:
                    continue
                intersect = vector_mesh_intersection(cam_trans, joints_frame[j], vertices, faces)
                if intersect:
                    print(f"{j}th joint is occluded")
                mask[j] = float(not intersect)

            joints[jp, jf] = joints_frame * mask

    return joints