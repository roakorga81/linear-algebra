"""Vector related implementations.

This module implements logic for Vector objects and vector operations.

"""
import math
from array import array

class Vector:
    typecode = 'd'

    def __init__(self, components):
        self._components = array(self.typecode, components)

    def __iter__(self):
        return iter(self._components)
    
    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return f'Vector({components})'

    @property
    def dim(self):
        return len(self._components)

    def __getitem__(self, slice_):
        return self.components[slice_]
        
    def __neg__(self):
        return -1 * 

    def __add__(self, v):
        if not isinstance(v, type(self)):
            msg = "Can only add vectors to other vectors, got {}"
            raise TypeError(msg.format(type(v)))
        return add(self, v)

    def __div__(self, alpha):
        return scale(self, 1 / alpha)

    def __mul__(self, alpha):
        return scale(self, alpha)

    def __rmul__(self, alpha):
        return self * alpha

    def __abs__(self):
        return norm(self)

    def __len__(self):
        return self.dim

    def __iter__(self):
        return iter(self.components)

    def __repr__(self):
        comp_repr = (('{},'*(self.ndim))[:-1]).format(*self.components)
        return "Vector([" + comp_repr + "])"
