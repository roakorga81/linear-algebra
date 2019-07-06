
from lac.vectors import ops

class Vector:
    def __init__(self, components):
        self._components = tuple(components)

    @property
    def components(self):
        return self._components

    @property
    def ndim(self):
        return len(self.components)

    def __neg__(self):
        return ops.scale(self, -1)

    def __add__(self, v):
        if not isinstance(v, type(self)):
            msg = "Can only add vectors to other vectors, got {}"
            raise TypeError(msg.format(type(v)))
        return ops.add(self, v)

    def __div__(self, alpha):
        return ops.scale(self, 1 / alpha)

    def __mul__(self, alpha):
        return ops.scale(self, alpha)

    def __rmul__(self, alpha):
        return self * alpha

    def __abs__(self):
        return ops.norm(self)

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __repr__(self):
        comp_repr = (('{},'*(self.ndim))[:-1]).format(*self.components)
        return "Vector([" + comp_repr + "])"