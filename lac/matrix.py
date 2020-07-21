import typing as t
from numbers import Number

import lac.vector as vector_ops
from lac import Vector, PRECISION


def _validate_vector_dimensions(vectors: t.Sequence[Vector]) -> None:
    ref = vectors[0].dim
    if not all(v.dim == ref for v in vectors):
        raise ValueError("vectors do not have the same number of dimensions")


class Matrix:
    @classmethod
    def from_columnvectors(cls, vectors: t.Sequence[Vector]):
        return cls(columns=vectors)

    @classmethod
    def from_rowvectors(cls, vectors: t.Sequence[Vector]):
        return cls(rows=vectors)

    @classmethod
    def make_random(cls, num_rows: int, num_columns: int):
        """A Matrix built out of random unit row vectors. """
        return cls.from_rowvectors(
            Vector.make_random(num_columns) for _ in range(num_rows)
        )

    @classmethod
    def make_identity(cls, num_rows: int, num_columns: int):
        rowvectors = []
        for i in range(num_rows):
            components = [0] * num_columns
            if i < num_columns:
                components[i] = 1
            rowvectors.append(Vector(components))
        return cls.from_rowvectors(rowvectors)

    @classmethod
    def make_zero(cls, num_rows: int, num_columns: int):
        return cls.from_rowvectors(
            Vector.make_zero(num_columns) for _ in range(num_rows)
        )

    def __init__(self, columns=None, rows=None):
        cond1 = columns is None and rows is None
        cond2 = columns is not None and rows is not None
        if cond1 or cond2:
            msg = "vou should pass either column vectors only or row vectors only"
            raise TypeError(msg)
        if columns is not None:
            self._columvectors = tuple(columns)
            _validate_vector_dimensions(self._columvectors)
            self._rowvectors = None
        if rows is not None:
            self._rowvectors = tuple(rows)
            _validate_vector_dimensions(self._rowvectors)
            self._columvectors = None

    @property
    def columnvectors(self) -> t.Tuple[Vector, ...]:
        if self._columvectors is None:
            self._columvectors = tuple(
                Vector(v[i] for v in self.rowvectors) for i in range(self.num_columns)
            )
        return self._columvectors

    @property
    def rowvectors(self) -> t.Tuple[Vector, ...]:
        if self._rowvectors is None:
            self._rowvectors = tuple(
                Vector(v[i] for v in self.columnvectors) for i in range(self.num_rows)
            )
        return self._rowvectors

    @property
    def num_columns(self) -> int:
        if self._columvectors is None:
            return self.rowvectors[0].dim
        return len(self.columnvectors)

    @property
    def num_rows(self) -> int:
        if self._rowvectors is None:
            return self.columnvectors[0].dim
        return len(self.rowvectors)

    @property
    def shape(self) -> t.Tuple[int, int]:
        return (self.num_rows, self.num_columns)

    @property
    def T(self):
        return transpose(self)

    def iterrows(self):
        for row in self.rowvectors:
            yield row

    def itercolumns(self):
        for col in self.columnvectors:
            yield col

    def __eq__(self, other):
        return almost_equal(self, other)

    def __matmul__(self, other):
        return matrix_multiply(self, other)

    def __iter__(self):
        return self.iterrows()

    def __len__(self):
        return self.num_rows

    def __getitem__(self, slice_):
        if isinstance(slice_, int):
            return self.rowvectors[slice_]
        elif isinstance(slice_, tuple):
            row, col = slice_
            if isinstance(row, int) and isinstance(col, int):
                return self.rowvectors[row][col]
            elif isinstance(row, slice) and isinstance(col, int):
                return Vector(self.columnvectors[col][row])
            elif isinstance(row, int) and isinstance(col, slice):
                return Vector(self.rowvectors[row][col])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __repr__(self):
        index = len("Vector(")
        vals = "\n  ".join(repr(v)[index:-1] for v in self.iterrows())
        return f"Matrix(\n  {vals[:-1]}],\n shape={self.shape}\n)"


def scale(m: Matrix, k: Number) -> Matrix:
    """Scale matrix m by k. """
    return Matrix.from_rowvectors((k * v for v in m.rowvectors))


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """Adds two matrices. """
    return Matrix.from_rowvectors(
        (v1 + v2 for v1, v2 in zip(m1.rowvectors, m2.rowvectors))
    )


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """Substracts the second matrix from the first one. """
    return add(m1, scale(m2, -1))


def vector_multiply(m: Matrix, v: Vector, from_left: bool = False) -> Vector:
    """Multiplies a matrix with a vector from the right or the left. """
    cond1 = m.num_rows != v.dim and from_left
    cond2 = m.num_columns != v.dim and not from_left
    if cond1 or cond2:
        raise ValueError(f"Shape mismatch: m({m.shape}), v({v.dim})")

    if from_left:
        out = Vector(vector_ops.dot(v, vi) for vi in m.columnvectors)
    else:
        out = Vector(vector_ops.dot(v, vi) for vi in m.rowvectors)
    return out


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Multiplies two matrices together.

    Args:
        m1 (Matrix): Matrx of shape (m,n)
        m2 (Matrix): Matrix of shape (n, k)

    Returns:
        (Matrix): The product of m1 and m2, has shape (m, k)

    Raises:
        ValueError: if the number of columns in m1 does not match the number of
            rows in m2
    """
    if m1.num_columns != m2.num_rows:
        msg = (
            "number of columns in m1 must match number of rows in m2, got {} and {} "
            "instead"
        )
        raise ValueError(msg.format(m1.num_columns, m2.num_rows))

    out = Matrix.from_rowvectors(
        vector_multiply(m2, row, from_left=True) for row in m1.iterrows()
    )
    return out


def transpose(m: Matrix) -> Matrix:
    return Matrix.from_rowvectors(m.itercolumns())


def trace(m: Matrix) -> Number:
    """Computes the sum of the diagnal entries of a matrix. """
    return sum(row[i] for i, row in enumerate(m.iterrows()) if i < m.num_columns)


def invert(m: Matrix) -> Matrix:
    raise NotImplementedError


def determinant(m: Matrix) -> Matrix:
    raise NotImplementedError


def almost_equal(m1: Matrix, m2: Matrix, ndigits: int = PRECISION) -> Matrix:
    return all(
        vector_ops.almost_equal(v1, v2, ndigits=ndigits) for v1, v2 in zip(m1, m2)
    )

