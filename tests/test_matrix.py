import unittest

from lac import Matrix


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.mat1 = Matrix.make_random(3, 3)
        self.mat2 = Matrix.make_random(3, 2)

    def test_from_columnvectors(self):
        pass

    def test_from_rowvectors(self):
        pass

    def test_make_random(self):
        pass

    def test_make_identity(self):
        pass

    def test_make_zero(self):
        pass

    def test_iteration(self):
        pass

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

