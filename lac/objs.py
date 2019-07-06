from lac import vec_ops
from lac import mat_ops

class Vector:
    def __init__(self, components):
        self._components = tuple(components)

    @property
    def components(self):
        return self._components

    @property
    def ndim(self):
        return len(self.components)

    def __getitem__(self, slice_):
        return self.components[slice_]
        
    def __neg__(self):
        return vec_ops.scale(self, -1)

    def __add__(self, v):
        if not isinstance(v, type(self)):
            msg = "Can only add vectors to other vectors, got {}"
            raise TypeError(msg.format(type(v)))
        return vec_ops.add(self, v)

    def __div__(self, alpha):
        return vec_ops.scale(self, 1 / alpha)

    def __mul__(self, alpha):
        return vec_ops.scale(self, alpha)

    def __rmul__(self, alpha):
        return self * alpha

    def __abs__(self):
        return vec_ops.norm(self)

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __repr__(self):
        comp_repr = (('{},'*(self.ndim))[:-1]).format(*self.components)
        return "Vector([" + comp_repr + "])"


class Matrix:
    @classmethod
    def from_column_vectors(cls, vectors):
        return cls(columns=vectors)

    @classmethod
    def from_row_vectors(cls, vectors):
        return cls(rows=vectors)

    def __init__(self, columns=None, rows=None):
        if columns is None and rows is None:
            msg = "You should pass either column vectors or row vectors"
            raise TypeError(msg)
        self._columvectors = columns
        self._rowvectors = rows

    @property
    def column_vectors(self):
        if self._columvectors is None:
            raise NotImplementedError
        return self._columvectors

    @property
    def row_vectors(self):
        if self._rowvectors is None:
            raise NotImplementedError
        return self._rowvectors