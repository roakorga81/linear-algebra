import itertools
import math
import unittest

import lac.matrix as matrix_ops
import lac.vector as vector_ops
from lac import Matrix, Vector, PRECISION

import lac.testing_utils as utils


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
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            self.assertTupleEqual(mat.shape, (n, m))

    def test_make_identity(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            eye_right = Matrix.make_identity(m, m)
            eye_left = Matrix.make_identity(n, n)
            utils.assert_matrices_almost_equal(mat @ eye_right, mat)
            utils.assert_matrices_almost_equal(eye_left @ mat, mat)

    def test_make_zero(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            zero_right = Matrix.make_zero(m, m)
            zero_left = Matrix.make_zero(n, m)
            utils.assert_matrices_almost_equal(mat @ zero_right, zero_left)
            utils.assert_matrices_almost_equal(zero_left.T @ mat, zero_right)

    def test_iteration_by_rows(self):
        for mat in utils.ALL_MATRICES:
            for i, (row1, row2) in enumerate(zip(mat, mat.iterrows())):
                self.assertEqual(row1.dim, mat.num_columns)
                self.assertEqual(row2.dim, mat.num_columns)
                self.assertEqual(row1, row2)
            self.assertEqual(i + 1, mat.num_rows)

    def test_iteration_by_columns(self):
        for mat in utils.ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                self.assertEqual(col.dim, mat.num_rows)
            self.assertEqual(i + 1, mat.num_columns)

    def test_slicing_row(self):
        for mat in utils.ALL_MATRICES:
            for i, row in enumerate(mat.iterrows()):
                vector_ops.almost_equal(mat[i], row)

    def test_slicing_column(self):
        for mat in utils.ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                vector_ops.almost_equal(mat[:, i], col)

    def test_slicing_matrix_edges(self):
        for mat in utils.ALL_MATRICES:
            for i, j in itertools.product(
                range(1, mat.num_rows + 1), range(1, mat.num_columns + 1)
            ):
                mat_ = mat[:i, :j]
                self.assertIsInstance(mat_, Matrix)
                if i < mat.num_rows and j < mat.num_columns:
                    mat_ = mat[i:, j:]
                    self.assertIsInstance(mat_, Matrix)

    def test_slicing_single_value(self):
        for mat in utils.ALL_MATRICES:
            for i, j in itertools.product(range(mat.num_rows), range(mat.num_columns)):
                self.assertIsInstance(mat[i, j], (int, float))

    def test_transponse_involution(self):
        for mat in utils.ALL_MATRICES:
            utils.assert_matrices_almost_equal(mat.T.T, mat)

    def test_transpose_linearity(self):
        pass

    def test_transpose_multiplication(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            utils.assert_matrices_almost_equal((m1 @ m2).T, m2.T @ m1.T)

    def test_transpose_cyclic(self):
        pass

    def test_transpose_det(self):
        pass

    def test_transpose_inverse(self):
        pass

    def test_transponse_eigenvalues(self):
        pass

    def test_norm_scale(self):
        pass

    def test_norm_addition(self):
        pass

    def test_norm_possitive(self):
        pass

    def test_norm_zero(self):
        pass

    def test_determinant(self):
        for mat in utils.ALL_MATRICES:
            if mat.num_rows == mat.num_columns:
                _ = mat.determinant
            else:
                with self.assertRaises(RuntimeError):
                    _ = mat.determinant

    def test_inverse(self):
        pass

    def test_trace(self):
        pass

    def test_trace_identity(self):
        pass

    def test_matmul(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.num_columns == mat2.num_rows:
                utils.assert_matrices_almost_equal(
                    mat1 @ mat2, matrix_ops.matrix_multiply(mat1, mat2)
                )

    def test_add(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.shape == mat2.shape:
                utils.assert_matrices_almost_equal(
                    mat1 + mat2, matrix_ops.add(mat1, mat2)
                )

    def test_scalar_multiply(self):
        for k, mat in itertools.product(utils.ALL_SCALARS, utils.ALL_MATRICES):
            utils.assert_matrices_almost_equal(k * mat, matrix_ops.scale(mat, k))

    def test_negation(self):
        for mat in utils.ALL_MATRICES:
            utils.assert_matrices_almost_equal(-mat, matrix_ops.scale(mat, -1))

    def test_subtraction(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            utils.assert_matrices_almost_equal(
                mat1 - mat2, matrix_ops.subtract(mat1, mat2)
            )


class TestMatrixOps(unittest.TestCase):
    def test_scale(self):
        for m, k in itertools.product(utils.ALL_MATRICES, utils.ALL_SCALARS):
            mat = matrix_ops.scale(m, k)
            self.assertTupleEqual(m.shape, mat.shape)
            for i in range(mat.num_rows):
                for j in range(mat.num_columns):
                    self.assertAlmostEqual(k * m[i, j], mat[i, j])

    def test_add(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.shape == m2.shape:
                mat = matrix_ops.add(m1, m2)
                for i in range(mat.num_rows):
                    for j in range(mat.num_columns):
                        self.assertAlmostEqual(m1[i, j] + m2[i, j], mat[i, j])

    def test_subtract(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.shape == m2.shape:
                mat = matrix_ops.subtract(m1, m2)
                for i in range(mat.num_rows):
                    for j in range(mat.num_columns):
                        self.assertAlmostEqual(m1[i, j] - m2[i, j], mat[i, j])

    def test_vector_multiply_from_right(self):
        for mat in utils.ALL_MATRICES:
            for vec in utils.ALL_VECTORS:
                if mat.num_columns == vec.dim:
                    for b in (True, False):
                        v = matrix_ops.vector_multiply(mat, vec, from_left=b)
                        self.assertIsInstance(v, Vector)
                        if b:
                            self.assertEqual(v.dim, mat.num_columns)
                        else:
                            self.assertEqual(v.dim, mat.num_rows)
                else:
                    with self.assertRaises(ValueError):
                        _ = matrix_ops.vector_multiply(mat, v)

    def test_matrix_multiply(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.num_columns == m2.num_rows:
                m = matrix_ops.matrix_multiply(m1, m2)
                self.assertIsInstance(m, Matrix)
            else:
                with self.assertRaises(ValueError):
                    _ = matrix_ops.matrix_multiply(m1, m2)
