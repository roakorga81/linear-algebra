from lac import vec

class Matrix:
    @classmethod
    def from_column_vectors(cls, vectors):
        return cls(columns=vectors)

    @classmethod
    def from_row_vectors(cls, vectors):
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