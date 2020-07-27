import math

import lac.vector as vector_ops
import lac.matrix as matrix_ops
from lac import Matrix, Vector

VEC1 = Vector([1, 2, -3])
VEC2 = Vector([-4, 5, -6])
VEC3 = Vector([1, 2, 3])
VEC4 = Vector([math.pi, 1 / math.pi, math.e])
VEC5 = Vector([0, 0, 0])

ALL_VECTORS = [VEC1, VEC2, VEC3, VEC4, VEC5]

ALL_SCALARS = [0, 1, -1, math.pi, math.e, 0.5, 1 / math.pi, -1 / math.e]

MAT1 = Matrix.from_rowvectors([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])

ALL_MATRICES = [MAT1]


def assert_vectors_almost_equal(v1, v2):
    assert vector_ops.almost_equal(v1, v2)


def assert_matrices_almost_equal(m1, m2):
    assert matrix_ops.almost_equal(m1, m2)
