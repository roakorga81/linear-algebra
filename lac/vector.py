import math
import random
import reprlib
import typing as t
from array import array

from lac import PRECISION


class Vector:
    typecode = "d"

    @classmethod
    def make_random(cls, dim):
        """Make a unitary vector of random components. """
        return cls(random.random() for _ in range(dim))

    @classmethod
    def make_zero(cls, dim: int):
        """Make a zero vector of any lenght. """
        return cls([0] * dim)

    @classmethod
    def make_unitary(cls, components: t.Iterable[t.Union[int, float]]):
        """Make a unitary vetor out of the components. """
        return build_unit_vector(cls(components))

    def __init__(self, components: t.Iterable[t.Union[int, float]]):
        self._components = array(self.typecode, components)
        self._norm: t.Optional[t.Union[int, float]] = None

    @property
    def norm(self) -> t.Union[int, float]:
        if self._norm is None:
            self._norm = math.sqrt(dot(self, self))
        return self._norm

    @property
    def components(self) -> array:
        return self._components

    @property
    def dim(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, slice_):
        if isinstance(slice_, int):
            return self.components[slice_]
        elif isinstance(slice_, slice):
            return Vector(self.components[slice_])
        else:
            raise RuntimeError("unsupported slicing")

    def __matmul__(self, other):
        return dot(self, other)

    def __add__(self, other):
        return add(self, other)

    def __rmul__(self, k):
        return scale(self, k)

    def __neg__(self):
        return scale(self, -1)

    def __sub__(self, other):
        return subtract(self, other)

    def __abs__(self):
        return self.norm

    def __len__(self):
        return self.dim

    def __eq__(self, other):
        return almost_equal(self, other)

    def __repr__(self):
        components = reprlib.repr([round(c, 3) for c in self.components])
        idx = components.find("[")
        components = components[idx:]
        return f"Vector({components})"


def scale(v: Vector, k: t.Union[int, float]) -> Vector:
    """Scales a vector by k.

    Raises:
        TypeError: if k is not a t.Union[int, float].
    """
    if not isinstance(k, (int, float)):
        msg = "Vectors can only be scaled by scalars! got {}"
        raise TypeError(msg.format(type(k)))

    return Vector([k * c for c in v])


def add(v1: Vector, v2: Vector) -> Vector:
    """Adds two vectors of the same dimension.

    Raise:
        ValueError: if vectors do not have the same dimesion.
    """
    if v1.dim != v2.dim:
        msg = "Vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.dim, v2.dim))

    return Vector([c1 + c2 for c1, c2 in zip(v1, v2)])


def subtract(v1: Vector, v2: Vector) -> Vector:
    """Subtracts the second vector from the first vector. """
    return add(v1, scale(v2, -1))


def dot(v1: Vector, v2: Vector) -> t.Union[int, float]:
    """Computes the dot product of two vectors.

    Raises:
        ValueError: if vectors do not have the same dimension

    """
    if v1.dim != v2.dim:
        msg = "vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.dim, v2.dim))

    return math.fsum(c1 * c2 for c1, c2 in zip(v1, v2))


def angle_between(v1: Vector, v2: Vector) -> t.Union[int, float]:
    """Computes the angle between two vectors. """
    frac = dot(v1, v2) / (v1.norm * v2.norm)
    frac = max(min(frac, 1), -1)
    return math.acos(frac)


def cross(v1: Vector, v2: Vector) -> Vector:
    """Computes the cross product between two 3-dimensional vectors.

    Raises:
        ValueError: if any of the vectors in not 3 dimentional.
    """
    for v in [v1, v2]:
        if v.dim != 3:
            msg = "Expected 3-dimentional vector, got a {}-dimentional"
            raise ValueError(msg.format(v.dim))

    c1 = v1[1] * v2[2] - v1[2] * v2[1]
    c2 = v1[0] * v2[2] - v1[2] * v2[0]
    c3 = v1[0] * v2[1] - v1[1] * v2[0]
    return Vector((c1, -c2, c3))


def build_unit_vector(v: Vector) -> Vector:
    """Builds a unit vector from the provided vector. """
    return scale(v, (1 / v.norm))


def project(v: Vector, d: Vector) -> t.Union[int, float]:
    """Projects a vector v into the vector d.

    Arguments:
        v (Vector): the vector you wish to project.
        d (Vector): the fector you wish to project onto.

    Returns:
        Vector: the projection of v onto d.
    """
    return dot(v, build_unit_vector(d))


def almost_equal(v1: Vector, v2: Vector, ndigits: int = PRECISION) -> bool:
    return (
        all(round(c1, ndigits) == round(c2, ndigits) for c1, c2 in zip(v1, v2))
        and v1.dim == v2.dim
    )
