"""Structures for linear programming variables.

Allows multi-dimensional arrays of objects to be processed in parallel using element-wise
operations, with full support for pulp.LpVariable objects, unlike numpy or pandas classes.
Arithmetic operations avaible across and between classes, and constraints can be added in
conjunction when associated with LpProblem (outer class for pulp.LpProblem).

    Typical usage example:

    #TODO

"""


class LpProblem:
    pass


class LpArray:
    def __init__(self, data, index, prob):
        pass

    @classmethod
    def variable():
        pass


class LpMatrix:
    def to_tensor(self):
        pass


class LpTensor:
    pass
