import numpy as np
import unittest
import sys
import os
sys.path.insert(0, os.getcwd())
from tools.geometry import vector_mesh_intersection

class TestVectorMeshIntersection(unittest.TestCase):
    def setUp(self):
        # Create a simple triangular mesh for testing
        # A single triangle in the XY plane at z=0
        self.mesh_vertices = np.array([
            [0, 0, 0],  # v0
            [1, 0, 0],  # v1
            [0, 1, 0],  # v2
        ])
        self.mesh_faces = np.array([[0, 1, 2]])  # Single triangle face

    def test_simple_intersection(self):
        # Vector passing through the triangle
        vector_start = np.array([0.25, 0.25, 1])
        vector_end = np.array([0.25, 0.25, -1])
        
        result = vector_mesh_intersection(vector_start, vector_end, 
                                       self.mesh_vertices, self.mesh_faces)
        self.assertTrue(result, "Should detect intersection with triangle")

    def test_no_intersection(self):
        # Vector passing beside the triangle
        vector_start = np.array([-1, -1, 1])
        vector_end = np.array([-1, -1, -1])
        
        result = vector_mesh_intersection(vector_start, vector_end, 
                                       self.mesh_vertices, self.mesh_faces)
        self.assertFalse(result, "Should not detect intersection when vector misses triangle")

    def test_parallel_to_mesh(self):
        # Vector parallel to the triangle
        vector_start = np.array([0, 0, 1])
        vector_end = np.array([1, 0, 1])
        
        result = vector_mesh_intersection(vector_start, vector_end, 
                                       self.mesh_vertices, self.mesh_faces)
        self.assertFalse(result, "Should not detect intersection when vector is parallel")

    def test_endpoint_before_mesh(self):
        # Vector ending before reaching the triangle
        vector_start = np.array([0.25, 0.25, 2])
        vector_end = np.array([0.25, 0.25, 1])
        
        result = vector_mesh_intersection(vector_start, vector_end, 
                                       self.mesh_vertices, self.mesh_faces)
        self.assertFalse(result, "Should not detect intersection when vector ends before mesh")

    def test_startpoint_after_mesh(self):
        # Vector starting after the triangle
        vector_start = np.array([0.25, 0.25, -1])
        vector_end = np.array([0.25, 0.25, -2])
        
        result = vector_mesh_intersection(vector_start, vector_end, 
                                       self.mesh_vertices, self.mesh_faces)
        self.assertFalse(result, "Should not detect intersection when vector starts after mesh")

def run_tests():
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    run_tests()
