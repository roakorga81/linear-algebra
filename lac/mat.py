from lac import vec

###################################################################
########################## Matrix class ###########################
###################################################################

def _validate_vector_dimensions(vectors):
    ref = vectors[0].ndim
    if not all(v.ndim == ref for v in vectors):
        raise ValueError("Vectors do not have the same number of dimensions")

class Matrix:
    @classmethod
    def from_columnvectors(cls, vectors):
        return cls(columns=vectors)

    @classmethod
    def from_rowvectors(cls, vectors):
        return cls(rows=vectors)

    def __init__(self, columns=None, rows=None):
        cond1 = columns is None and rows is None
        cond2  = columns is not None and rows is not None
        if cond1 or cond2:
            msg = "You should pass either column vectors only or row vectors only"
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
    def columnvectors(self):
        if self._columvectors is None:
            self._columvectors = tuple(
                vec.Vector(v[i] for v in self.rowvectors)
                for i in range(self.num_columns)
            )
        return self._columvectors

    @property
    def rowvectors(self):
        if self._rowvectors is None:
            self._rowvectors = tuple(
                vec.Vector(v[i] for v in self.columnvectors)
                for i in range(self.num_rows)
            )
        return self._rowvectors

    @property
    def num_columns(self):
        if self._columvectors is None:
            return self.rowvectors[0].ndim
        return len(self.columnvectors)

    @property
    def num_rows(self):
        if self._rowvectors is None:
            return self.columnvectors[0].ndim
        return len(self.rowvectors)

    @property
    def shape(self):
        return (self.num_rows, self.num_columns)

    def iterrows(self):
        for row in self.rowvectors:
            yield row

    def itercolumns(self):
        for col in self.columnvectors:
            yield col

    def __iter__(self):
        return self.iterrows()

    def __len__(self):
        return self.num_rows

    def __repr__(self):
        return f"Matrix(shape={self.shape})"


###################################################################
########################### Operations ############################
###################################################################

def scale(m, alpha):
    """Escale each component of a matrix by a factor alpha.

    Arguments:
        m (Matrix): The matrix to be scaled
        alpha (float or int): Scale factor

    Returns:
        A Matrix whose components are the ones of m scaled by alpha.

    """
    return Matrix.from_rowvectors((alpha * v for v in m.rowvectors))

def add(m1, m2):
    """Adds two matrices.

    Args:
        m1, m2 (Matrix): The matrices to be added.

    Returns:
        A Matrix whose elements are the sum of the components of each of the
        provided Matrices.

    """
    return Matrix.from_rowvectors(
        (v1 + v2 for v1, v2 in zip(m1.rowvectors, m2.rowvectors)))

def subtract(m1, m2):
    """Substracts the second matrix from the first one.

    Args:
        m1, m2 (Matrix): Matrices to perform m1 - m2.

    Returns:
        A Matrix whose elements are the numbers shuch that m2 + out = m1.
    """
    return add(m1, scale(m2, -1))

def vector_multiply(m, v, from_left=False):
    """Multiplies a matrix with a vector from the right or the left.

    Args:
        m (Matrix): The matrix
        v (Vector): The vector
        from_left (bool): Whether the vector multiplies from the left. Defaults
            to `False`.
    
    Returns:
        A Matrix whose shape is determined by the same of `v` and the
        `from_left` parameter.
    """
    cond1 = m.num_rows != v.ndim and from_left
    cond2 = m.num_columns != v.ndim and not from_left
    if cond1 or cond2:
        raise ValueError(f"Shape mismatch: m({m.shape}), v({v.ndim})")

    if from_left:
        out = vec.Vector(vec.dot(v, vi) for vi in m.columnvectors)
    else:
        out = vec.Vector(vec.dot(v, vi) for vi in m.rowvectors)
    return out

# def matrix_multiply(m1, m2):
#     raise NotImplementedError

# def transpose(m):
#     raise NotImplementedError

# def inverse(m):
#     raise NotImplementedError

# def determinan(m):
#     raise NotImplementedError

# def trace(m):
#     raise NotImplementedError