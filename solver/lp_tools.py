"""Structures for linear programming variables.

Allows multi-dimensional arrays of objects to be processed in parallel using element-wise operations, with full support
for pulp.LpVariable objects, unlike numpy or pandas classes. Arithmetic operations avaible across and between classes,
and constraints can be added when associated with LpProblem (outer class for pulp.LpProblem).

    Typical usage example:

    #TODO

"""

import pulp
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


class LpProblem:
    pass


class LpArray:
    """1-Dimensional array-like structure with support for pulp.LpVariable objects, and support for applying element-
    wise PuLP constraints when associated with LpProblem.
    """

    def __init__(self, data: ArrayLike = None, index: ArrayLike = None, prob: LpProblem = None):
        """This class models an array with the following parameters:

        Args:
            data (ArrayLike, optional): Values contained in LpArray. Defaults to None.
            index (ArrayLike, optional): Paired indices of values. Defaults to None.
            prob (LpProblem, optional): Associated LpProblem object for constraint application. Defaults to None.
        """
        if data is None:
            data = []

        if index is None:
            index = []

        try:
            if len(data) != len(index):
                raise Exception("data and index have different lengths")
        except TypeError:
            raise TypeError("data and/or index are not array-like")

        self.values = np.array(data)
        self.index = np.array(index)
        self.prob = prob

    @classmethod
    def from_dict(cls, data: dict = None, prob: LpProblem = None, sort_index: bool = False):
        """Initialise LpArray with data from dict, with the following parameters:

        Args:
            data (dict, optional): dict (length n) object containing {index[0]: values[0], index[1]: values[1], ..., \
                 index[1]: values[n]}. Defaults to None.
            prob (LpProblem, optional): [description]. Defaults to None.
        """
        if sort_index:
            data = dict(sorted(data.items()))  #  Sort dict by keys
        return cls(list(data.values()), list(data.keys()), prob)    # Initialise class instance from dict

    @classmethod
    def variable(cls, name: str, index: ArrayLike = None, lower: float = None, upper: float = None, cat: type = None,
                 prob: LpProblem = None):
        """Initialise LpArray containing pulp.LpVariable objects, with the following parameters:

        Args:
            name (str): Name for pulp.LpVariable
            index (ArrayLike, optional): [description]. Defaults to empty
            lower (float, optional) : Lower bound for variables to be created. Defaults to None
            upper (float, optional): Upper bound for variables to be created. Defaults to None
            cat (type, optional): Category of variables: bool, int, or float. Defaults to float
        """
        # Generate and process dict of pulp.LpVariable objects
        return cls.from_dict(pulp.LpVariable.dict(name, index, lower, upper, (
            'Binary', 'Integer', 'Continuous')[(bool, int, float).index(cat)]), prob)

    def __str__(self):
        return str(pd.Series(self.values, self.index))


class LpMatrix:
    def to_tensor(self) -> 'LpTensor':
        pass


class LpTensor:
    def get_diag(self) -> LpMatrix:
        pass


if __name__ == '__main__':
    lp = pulp.LpVariable
    a = LpArray.variable('Bench', range(100), cat=int)
    print(a)
