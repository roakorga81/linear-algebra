import itertools
import math
import unittest
from t.Union[int, float]s import t.Union[int, float]

import lac.matrix as matrix_ops
import lac.vector as vector_ops
from lac import Matrix, Vector, PRECISION


MAT1 = Matrix.from_rowvectors([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])

ALL_MATRICES = [MAT1]

ALL_SCALARS = [0, 1, -1, math.pi, math.e, 0.5, 1 / math.pi, -1 / math.e]


def assert_matrices_almost_equal(m1, m2):
    assert matrix_ops.almost_equal(m1, m2)


class TestMatrix(unittest.TestCase):
    def test_from_columnvectors(self):
        mat = Matrix.from_columnvectors(
            [Vector([1, 0]), Vector([0, 1]), Vector([0, 0])]
        )
        self.assertTupleEqual(mat.shape, (2, 3))

    def test_from_rowvectors(self):
        mat = Matrix.from_rowvectors([Vector([1, 0]), Vector([0, 1]), Vector([0, 0])])
        self.assertTupleEqual(mat.shape, (3, 2))

    def test_make_random(self):
        mat = Matrix.make_random(2, 4)
        self.assertTupleEqual(mat.shape, (2, 4))

    def test_make_identity(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            eye_right = Matrix.make_identity(m, m)
            eye_left = Matrix.make_identity(n, n)
            assert_matrices_almost_equal(mat @ eye_right, mat)
            assert_matrices_almost_equal(eye_left @ mat, mat)

    def test_make_zero(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            zero_right = Matrix.make_zero(m, m)
            zero_left = Matrix.make_zero(n, m)
            assert_matrices_almost_equal(mat @ zero_right, zero_left)
            assert_matrices_almost_equal(zero_left.T @ mat, zero_right)

    def test_iteration_by_rows(self):
        for mat in ALL_MATRICES:
            for i, (row1, row2) in enumerate(zip(mat, mat.iterrows())):
                self.assertEqual(row1.dim, mat.num_columns)
                self.assertEqual(row2.dim, mat.num_columns)
                self.assertEqual(row1, row2)
            self.assertEqual(i + 1, mat.num_rows)

    def test_iteration_by_columns(self):
        for mat in ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                self.assertEqual(col.dim, mat.num_rows)
            self.assertEqual(i + 1, mat.num_columns)

    def test_slicing_row(self):
        for mat in ALL_MATRICES:
            for i, row in enumerate(mat.iterrows()):
                vector_ops.almost_equal(mat[i], row)

    def test_slicing_column(self):
        for mat in ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                vector_ops.almost_equal(mat[:, i], col)

    def test_slicing_matrix(self):
        for mat in ALL_MATRICES:
            for i, j in itertools.product(range(mat.num_rows), range(mat.num_columns)):
                self.assertIsInstance(mat[:i, :j], t.Union[int, float])


    def test_slicing_single_value(self):
        for mat in ALL_MATRICES:
            for i, j in itertools.product(range(mat.num_rows), range(mat.num_columns)):
                self.assertIsInstance(mat[i, j], t.Union[int, float])

    def test_matmul(self):
        for mat1, mat2 in itertools.product(ALL_MATRICES, ALL_MATRICES):
            if mat1.num_columns == mat2.num_rows:
                assert_matrices_almost_equal(
                    mat1 @ mat2, matrix_ops.matrix_multiply(mat1, mat2)
                )

    def test_add(self):
        for mat1, mat2 in itertools.product(ALL_MATRICES, ALL_MATRICES):
            if mat1.shape == mat2.shape:
                assert_matrices_almost_equal(mat1 + mat2, matrix_ops.add(mat1, mat2))

    def test_scalar_multiply(self):
        for k, mat in itertools.product(ALL_SCALARS, ALL_MATRICES):
            assert_matrices_almost_equal(k * mat, matrix_ops.scale(mat, k))

    def test_negation(self):
        for mat in ALL_MATRICES:
            assert_matrices_almost_equal(-mat, matrix_ops.scale(mat, -1))

    def test_subtraction(self):
        for mat1, mat2 in itertools.product(ALL_MATRICES, ALL_MATRICES):
            assert_matrices_almost_equal(mat1 - mat2, matrix_ops.subtract(mat1, mat2))


class TestMatrixOps(unittest.TestCase):
    pass
