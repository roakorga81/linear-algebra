import itertools
import math
import unittest

import lac.vector as vector_ops
from lac import Vector
from lac import PRECISION


VEC1 = Vector([1, 2, -3])
VEC2 = Vector([-4, 5, -6])
VEC3 = Vector([1, 2, 3])
VEC4 = Vector([math.pi, 1 / math.pi, math.e])
VEC5 = Vector([0, 0, 0])

ALL_VECTORS = [VEC1, VEC2, VEC3, VEC4, VEC5]

ALL_SCALARS = [0, 1, -1, math.pi, math.e, 0.5, 1 / math.pi, -1 / math.e]


def assert_vectors_almost_equal(v1, v2):
    assert vector_ops.almost_equal(v1, v2)


class TestVector(unittest.TestCase):
    def make_random(self):
        dim = 3
        vec = Vector.make_random(dim)
        self.assertEqual(vec.norm, 1, PRECISION)

    def test_make_zero(self):
        dim = 3
        vec = Vector.make_zero(dim)
        self.assertEqual(vec.norm, 0, PRECISION)
        self.assertEqual(vec.dim, dim)

    def test_make_unitary(self):
        vec = Vector.make_unitary(VEC4)
        self.assertAlmostEqual(vec.norm, 1, PRECISION)
        self.assertEqual(vec.dim, VEC4.dim)

    def test_dim(self):
        components = [3, 2, 1]
        vec = Vector(components)
        self.assertEqual(len(components), vec.dim)

    def test_add(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            assert_vectors_almost_equal(v1 + v2, vector_ops.add(v1, v2))

    def test_scalar_multiply(self):
        for v, k in itertools.product(ALL_VECTORS, ALL_SCALARS):
            assert_vectors_almost_equal(k * v, vector_ops.scale(v, k))

    def test_dot(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            self.assertAlmostEqual(v1 @ v2, vector_ops.dot(v1, v2), PRECISION)


class TestVectorOps(unittest.TestCase):
    def test_build_unit_vector(self):
        pass

    def test_project(self):
        pass

    def test_scale(self):
        for v, k in itertools.product(ALL_VECTORS, ALL_SCALARS):
            if v.norm != 0 and k > 0:
                unit = vector_ops.build_unit_vector(v)
                assert_vectors_almost_equal(vector_ops.scale(unit, v.norm), v)
                assert_vectors_almost_equal(
                    vector_ops.build_unit_vector(vector_ops.scale(v, k)), unit
                )
                assert_vectors_almost_equal(vector_ops.scale(v, (1 / v.norm)), unit)

    def test_add_commutative(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            vec1 = vector_ops.add(v1, v2)
            vec2 = vector_ops.add(v2, v1)
            assert_vectors_almost_equal(vec1, vec2)

    def test_add_associative(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            for v3 in ALL_VECTORS:
                vec1 = vector_ops.add(vector_ops.add(v1, v2), v3)
                vec2 = vector_ops.add(v1, vector_ops.add(v2, v3))
                assert_vectors_almost_equal(vec1, vec2)

    def test_subtract(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            assert_vectors_almost_equal(
                vector_ops.subtract(v1, v2), vector_ops.add(v1, -v2)
            )

    def test_dot_commutative(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            self.assertAlmostEqual(vector_ops.dot(v1, v2), vector_ops.dot(v2, v1))

    def test_dot_distributive(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            for v3 in ALL_VECTORS:
                f1 = vector_ops.dot(v1, v2 + v3)
                f2 = vector_ops.dot(v1, v2) + vector_ops.dot(v1, v3)
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_bilinear(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            for v3 in ALL_VECTORS:
                for k in ALL_SCALARS:
                    f1 = vector_ops.dot(v1, (k * v2) + v3)
                    f2 = k * (vector_ops.dot(v1, v2)) + vector_ops.dot(v1, v3)
                    self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_scalar_multiplitcation(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            for k1, k2 in itertools.product(ALL_SCALARS, ALL_SCALARS):
                f1 = vector_ops.dot(k1 * v1, k2 * v2)
                f2 = k1 * k2 * vector_ops.dot(v1, v2)
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_not_associative(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            for v3 in ALL_VECTORS:
                with self.assertRaises(AttributeError):
                    vector_ops.dot(vector_ops.dot(v1, v2), v3)

    def test_dot_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.dot(v1, v2), 0, PRECISION)

    def test_angle_between_parallel(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), 0, PRECISION)

    def test_angle_between_antiparallel(self):
        v1 = Vector([-1, 0, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), math.pi, PRECISION)

    def test_angle_between_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), math.pi/2, PRECISION)

    def test_dot_angle_equality(self):
        for v1, v2 in itertools.product(ALL_VECTORS, ALL_VECTORS):
            if v1.norm > 0 and v2.nrom > 0:
                f1 = vector_ops.dot(v1, v2)
                f2 = v1.norm * v2.norm * math.cos(vector_ops.angle_between(v1, v2))
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_cross_parallel(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([1, 0, 0])


    def test_cross_antiparallel(self):
        v1 = Vector([-1, 0, 0])
        v2 = Vector([1, 0, 0])

    def test_cross_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])

    def test_cross_angle_equality(self):
        pass
