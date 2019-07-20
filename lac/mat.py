from lac import vec

###################################################################
########################## Matrix class ###########################
###################################################################

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
            
        self._columvectors = tuple(columns)
        self._rowvectors = tuple(rows)

    @property
    def columnvectors(self):
        if self._columvectors is None:
            self._columvectors = tuple(
                v[i] for v in self.rowvectors
                for i in range(self.num_columns)
            )
        return self._columvectors

    @property
    def rowvectors(self):
        if self._rowvectors is None:
            self._rowvectors = tuple(
                v[i] for v in self.columnvectors
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

    Arguments:
        m1, m2 (Matrix): The matrices to be added.

    Returns:
        A Matrix whose elements are the sum of the components of each of the
        provided Matrices.

    """
    return Matrix.from_rowvectors(
        (v1 + v2 for v1, v2 in zip(m1.rowvectors, m2.rowvectors)))

def subtract(m1, m2):
    raise NotImplementedError

def vector_multiply(m, v):
    raise NotImplementedError

def matrix_multiply(m1, m2):
    raise NotImplementedError

def transpose(m):
    raise NotImplementedError

def inverse(m):
    raise NotImplementedError

def determinan(m):
    raise NotImplementedError

def trace(m):
    raise NotImplementedError