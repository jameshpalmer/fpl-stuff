"""Structures for linear programming variables.

Allows multi-dimensional arrays of objects to be processed in parallel using element-wise
operations, with full support for pulp.LpVariable objects, unlike numpy or pandas classes.
Arithmetic operations avaible across and between classes, and constraints can be added in
conjunction with LpProblem (outer class for pulp.LpProblem).

    Typical usage example:

    #TODO
"""


class LpProblem:
    pass


class LpArray:
    pass


class LpMatrix:
    pass


class LpTensor:
    pass
