"""Vector related implementations.

This module implements logic for Vector objects and vector operations.

"""

import math

###################################################################
########################## Vector class ###########################
###################################################################
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
        return scale(self, -1)

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
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __repr__(self):
        comp_repr = (('{},'*(self.ndim))[:-1]).format(*self.components)
        return "Vector([" + comp_repr + "])"
###################################################################
########################## Operations #############################
###################################################################
def scale(v, alpha):
    """Scales a vectore by alpha.

    Arguments:
        v (Vector): Vector to be scaled.
        alpha (float, int): scale factor
    
    Returns
        Vector: A vector whose components are the components of the original
            vector multiplied by alpha.
    
    Raises:
        TypeError: if alpha is not float or int.
    """
    if not isinstance(alpha, (float, int)):
        msg = "Vectors can only be scaled by scalars! got {}"
        raise TypeError(msg.format(type(alpha)))

    return Vector((alpha * c for c in v))

def add(v1, v2):
    """Adds two vectors of the same dimension.

    Arguments:
        v1 (Vector): First vector in the addition operation.
        v2 (Vector): Second vector in the addition operation.

    Returns
        Vector: A vector whose components are the sum of the components of
            the two provided vectors.

    Raise:
        ValueError: if vectors do not have the same dimesion.
    """
    if v1.ndim != v2.ndim:
        msg = "Vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.ndim, v2.ndim))

    return Vector(c1 + c2 for c1, c2 in zip(v1, v2))

def subtraction(v1, v2):
    """Subtracts two vectors of the same dimension.

    Arguments:
        v1 (Vector): First vector in the subtraction operation.
        v2 (Vector): Second vector in the subtraction operation.

    Returns:
        Vector: A vector whose components are the difference of the components
            of the two provided vectors.

    """
    return add(v1, scale(-1, v2))

def dot(v1, v2):
    """Computes the dot product of two vectors.

    Arguments:
        v1 (Vector): First vector for the dot product.
        v2 (Vector): Second vector for the dot product.

    Returns:
        float, int: the sum of the multiplication of each of the components
            of the two provided vectors.
    """
    if v1.ndim != v2.ndim:
        msg = "Vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.ndim, v2.ndim))
        
    return sum(c1*c2 for c1,c2 in zip(v1, v2))

def cross(v1, v2):
    """Computes the cross product between two 3-dimensional vectors.

    Note that this operation is only defined between 3-dimensional vectors.

    Arguments:
        v1 (Vector): First vector for the cross product.
        v2 (Vector): Second vector for the cross product.

    Returns:
        Vector: the result of the cross product between the two provided
            vectors.

    Raises:
        TypeError: if any of the vectors in not 3 dimentional.
    """
    for v in [v1, v2]:
        if v.ndim != 3:
            msg = "Expected 3-dimentional vector, got a {}-dimentional"
            raise TypeError(msg.format(v.ndim))

    c1 = v1[1] * v2[2] - v1[2] * v2[1]
    c2 = v1[2] * v2[0] - v1[2] * v2[0]
    c3 = v1[0] * v2[1] - v1[1] * v2[0]
    return Vector((c1, c2, c3))

def norm(v):
    """Computes the norm of the given vector.

    Arguments:
        v (Vector): vector to compute the norm for.

    Returns:
        float: the norm of the vector.
    """
    return math.sqrt(dot(v, v))

def build_unit_vector(v):
    """Builds a unit vector from the provided vector.

    Arguments:
        v (Vector): a vector from which to build a unit vector.

    Returns:
        Vector: a vector with the same direction as v but with euclidean norm
        equal to 1.
    """
    return v / norm(v)

def projection(v, d):
    """Projects a vector v into the vector d.

    Arguments:
        v (Vector): the vector you wish to project.
        d (Vector): the fector you wish to project onto.

    Returns:
        Vector: the projection of v onto d.
    """
    d = build_unit_vector(d)
    return dot(v, d)