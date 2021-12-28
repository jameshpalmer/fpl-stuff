"""Structures for linear programming variables.

Allows multi-dimensional arrays of objects to be processed in parallel using element-wise
operations, with full support for pulp.LpVariable objects, unlike numpy or pandas classes.
Arithmetic operations avaible across and between classes, and constraints can be added in
conjunction when associated with LpProblem (outer class for pulp.LpProblem).

    Typical usage example:

    #TODO

"""

import pulp
import numpy as np
from numpy.typing import ArrayLike


class LpProblem:
    pass


class LpArray:
    def __init__(self, data: ArrayLike = None, index: ArrayLike = None, prob: LpProblem = None):
        pass

    @classmethod
    def variable(cls, name: str, index: ArrayLike = None, lower: float = None, upper: float = None, cat: type = None):
        """Create LpArray containing pulp.LpVariable objects.

        Args:
            name (str): Name for pulp.LpVariable
            index (ArrayLike, optional): [description]. Defaults to empty
            lower (float, optional) : Lower bound for variables to be created. Defaults to None
            upper (float, optional): Upper bound for variables to be created. Defaults to None
            cat (type, optional): Category of variables: bool, int, or float. Defaults to float
        """


class LpMatrix:
    def to_tensor(self) -> 'LpTensor':
        pass


class LpTensor:
    def get_diag(self) -> LpMatrix:
        pass


if __name__ == '__main__':
    lp = pulp.LpVariable('lp', )
    LpArray.variable
