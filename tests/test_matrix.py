import itertools
import unittest

import lac.matrix as matrix_ops
from lac import Matrix, Vector, PRECISION


MAT1 = Matrix.from_rowvectors([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])

ALL_MATRICES = [MAT1]


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
        pass

    def test_slicing_column(self):
        pass

    def test_slicing_matrix(self):
        pass

    def test_slicing_single_value(self):
        pass

    def test_matmul(self):
        pass


class TestMatrixOps(unittest.TestCase):
    pass
