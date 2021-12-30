"""Structures for linear programming variables, simplifying the implementation of vector-based LP problems.

Classes allow multi-dimensional arrays of objects to be processed in parallel using element-wise operations, with full
support for `pulp.LpVariable` objects, unlike `numpy` or `pandas` classes. Arithmetic operations avaible across and
between classes, and constraints can be added when associated with LpProblem (outer class for `pulp.LpProblem`).

    Typical usage example:

    # TODO

"""
import pulp
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from typing import Sequence, Any


class LpProblem:
    pass


class LpArray:
    """1-Dimensional `pd.Series`-like structure with support for `pulp.LpVariable` objects, and support for \
    applying element-wise PuLP constraints when associated with `LpProblem`.
    """

    def __init__(self, data: Sequence = None, index: Sequence[float] = None, prob: LpProblem = None):
        """This class models an array with the following parameters:

        Args:
            data (Sequence, optional): Values contained in `LpArray`. Defaults to `None`.
            index (Sequence[float], optional): Indices (usually `int`) paired to corrresponding values. Defaults to \
                 `None`.
            prob (LpProblem, optional): Associated `LpProblem` object for constraint application. Defaults to `None`.
        """
        if type(data) == LpArray:
            self.values = data.values

            if index is None:
                self.index = data.index

            if prob is None:
                self.prob = data.prob

        # Default for values and index is empty
        if data is None:
            data = np.array([])

        if index is None:
            index = np.array([])

        try:
            if len(data) != len(index):
                raise Exception("data and index have different lengths")
        except TypeError:
            raise TypeError("data and/or index are not sequential")

        self.values = np.array(data)
        self.index = np.array(index)
        self.prob = prob

    @classmethod
    def from_dict(cls, data: dict = None, prob: LpProblem = None, sort_index: bool = False) -> 'LpArray':
        """Initialise `LpArray` with data from `dict`, with the following parameters:

        Args:
            data (dict, optional): `dict` (length n) object containing `{index[0]: values[0], index[1]: values[1], \
                  ..., index[1]: values[n]}`. Defaults to `None`.
            prob (LpProblem, optional): LpProblem associated with LpArray instance. Defaults to None.
            sort_index (bool, optional): If `True`, return `LpArray.from_dict(dict(sorted(dict.values())), ...)`

        Returns:
            LpArray: with values `dict.values()` and index `dict.keys()`
        """
        if sort_index:
            data = dict(sorted(data.items()))  #  Sort dict by keys
        return cls(list(data.values()), list(data.keys()), prob)    # Initialise class instance from dict

    @classmethod
    def variable(cls, name: str, index: Sequence[float] = None, lower: float = None, upper: float = None,
                 cat: type[bool | int | float] = None, prob: LpProblem = None) -> 'LpArray[pulp.LpVariable]':
        """Initialise `LpArray` containing `pulp.LpVariable` objects, with the following parameters:

        Args:
            name (str): Name for `pulp.LpVariable`
            index (Sequence[float], optional): Index of returned `LpArray`. Defaults to `None`
            lower (float, optional) : Lower bound for variables to be created. Defaults to `None`
            upper (float, optional): Upper bound for variables to be created. Defaults to `None`
            cat (type, optional): Category of variables: `bool`, `int`, or `float`. Defaults to `float`
            prob (LpProblem, optional): LpProblem associated with variables

        Returns:
            LpArray: with values that are pulp.LpVariable instances, named f'{name}_{i}' for all i in index
        """
        # Generate and process dict of pulp.LpVariable objects
        return cls.from_dict(pulp.LpVariable.dict(name, index, lower, upper, (
            'Binary', 'Integer', 'Continuous')[(bool, int, float).index(cat)]), prob)

    def __str__(self) -> str:
        """Convert LpArray to string for easy readability.

        Returns:
            str: pandas.Series-style overview of array
        """
        if self.values.size == 0:   # If empty LpArray
            return 'LpArray([])'

        return str(pd.Series([str(i) for i in self.values], self.index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: float | Sequence[bool]) -> Any:
        """Get item or subset of items from `self.values`, By index or binary inclusion sequence.

        Args:
            item (float | Sequence[bool]): Index corrresponding to wanted value, or sequence of binary values, where \
                nth element corresponds to whether to include nth index/value pair in output `LpArray`


        Returns:
            Any: Value corresponding to passed index, or `LpArray` corresponding to passed binary inclusion sequence
        """
        try:
            index = self.index.tolist().index(item)  # Get item as index
            return self.values[index]   # Return corresponding value

        except ValueError:  # Item is an invalid index
            try:
                return self.filter(item)    # Get item as filter

            except ValueError:  # Item is an invalid filter
                try:
                    # Get list of locations of referenced indices
                    indices = [self.index.tolist().index(i) for i in item]
                    # Return list of values corresponding to referenced indices
                    return [self.values[i] for i in indices]
                except ValueError as e:
                    raise ValueError(f"Invalid LpArray index/filter: {item} ({e})")

            except TypeError:   # Item has no len and is not in index
                raise ValueError(f"Invalid LpArray index/filter: {item}")

    def filter(self, item: Sequence[bool]) -> 'LpArray':
        """Filter `LpArray` with a binary sequence.

        Args:
            item (Sequence[bool]): Squence of `bool` values, indicating whether to include nth value in nth entry

         Returns:
             LpArray: Filtered LpArray
         """
        if len(item) != len(self):  # Filter has the wrong length
            raise ValueError(f"Invalid LpArray filter: {item} is not the same length.")

        if not all([(i in (0, 1)) for i in item]):  # Filter is not binary
            raise ValueError(f"Invalid LpArray filter: {item} is not a binary sequence.")

        return LpArray(data=[self.values[index] for index, i in enumerate(item) if i == 1], index=[
            self.index[index] for index, i in enumerate(item) if i == 1], prob=self.prob)


class LpMatrix:
    def to_tensor(self) -> 'LpTensor':
        pass


class LpTensor:
    def get_diag(self) -> LpMatrix:
        pass


if __name__ == '__main__':
    print(isinstance(1, bool))
    lp = pulp.LpVariable
    a = LpArray.variable('Bench', range(100), cat=int)
    random = np.random.randint(2, size=100)
    print(a[[0, 1] * 50])
