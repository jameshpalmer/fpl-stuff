"""Array-based structures for linear programming variables, simplifying the implementation of vector-based LP problems.

Classes allow multi-dimensional arrays of objects to be processed in parallel using element-wise operations, with full
support for `pulp.LpVariable` objects, unlike `numpy` or `pandas` classes. Arithmetic operations avaible across and
between classes, and constraints can be added when associated with LpProblem (outer class for `pulp.LpProblem`).

    Typical usage example:

    # TODO

"""
import pulp
import numpy as np
import pandas as pd
from typing import Union, Literal, Callable, Sequence, Any
from operator import add, sub, mul


class LpProblem:
    pass


class LpArray:
    """1-Dimensional ``-like structure with support for `pulp.LpVariable` objects, and support for \
    applying element-wise PuLP constraints when associated with `LpProblem`.
    """

    def __init__(self, data: Sequence = None, index: Sequence[float] = None, prob: LpProblem = None):
        """This class models an array with the following parameters:

        Args:
            data (Sequence, optional): Values contained in `LpArray`. Defaults to `None`
            index (Sequence[float], optional): Indices (usually `int`) paired to corrresponding values. Defaults to \
                 `None`
            prob (LpProblem, optional): Associated `LpProblem` object for constraint application. Defaults to `None`
        """
        # Default for values and index is empty
        if data is None:
            data = np.array([])

        if index is None:
            index = np.arange(len(data))

        try:
            if len(data) != len(index):
                raise ValueError("data and index have different lengths")
        except TypeError:
            raise TypeError("data and/or index are not sequential")

        if type(data) == LpArray:
            self.values = data.values

            if index is None:
                self.index = data.index

            if prob is None:
                self.prob = data.prob

        self.values, self.index, self.prob = np.array(data), np.array(index), prob

    @classmethod
    def from_dict(cls, data: dict = None, prob: LpProblem = None, sort_index: bool = False) -> 'LpArray':
        """Initialise `LpArray` with data from `dict`, with the following parameters:

        Args:
            data (dict, optional): `dict` (length n) object containing `{index[0]: values[0], index[1]: values[1], \
                  ..., index[n]: values[n]}`. Defaults to `None`
            prob (LpProblem, optional): LpProblem associated with LpArray instance. Defaults to `None`
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
            lower (float, optional): Lower bound for variables to be created. Defaults to `None`
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

        return '\n'.join(str(pd.Series([str(i) for i in self.values], self.index)).split(
            '\n')[:-1]) + f"\nLength: {len(self)}, dtype: {type(self.values[0]).__name__}"

    def __len__(self) -> int:
        """Returns the length of the index."""
        return len(self.index)

    def __iter__(self):
        """Iterate through `self.values`"""
        for value in self.values:
            yield value

    def __getitem__(self, item: float | Sequence[bool]) -> Any:
        """Returns item or subset of items from `self.values`, By index or binary inclusion sequence.

        Args:
            item (float | Sequence[bool]): Index corrresponding to wanted value, or sequence of binary values, where \
                nth element corresponds to whether to include nth index/value pair in output `LpArray`

        Raises:
            ValueError: Invalid index or filter

        Returns:
            Any: Value corresponding to passed index, or `LpArray` corresponding to passed binary inclusion sequence
        """
        match item:
            case float() | int() | np.int64():  # For 0d index references
                index = self.index.tolist().index(item)  # Get item as index
                return self.values[index]   # Return corresponding value

            case Sequence:  # 1d index references
                try:
                    return self.filter(item)    # Try item as binary filter
                except ValueError:
                    return self.get_subset(item)    # Try item as sequence of indices

    def filter(self, item: Sequence[bool], inplace: bool = False) -> 'LpArray':
        """Filter `LpArray` using a binary sequence of the same length.

        Args:
            item (Sequence[bool]): Squence of `bool` values, indicating whether to include nth value in nth entry
            inplace (bool): True => filter existing object. False => return new filtered object

        Raises:
            ValueError: Attempt to filter with non-binary or differently-sized data

         Returns:
             LpArray: Filtered LpArray
         """
        if len(item) != len(self):  # Filter has the wrong length
            raise ValueError(
                f"Invalid LpArray filter: {item} does not have the same length as LpArray \
                    ({len(item)} vs. {len(self)})")

        if not all([(i in (0, 1)) for i in item]):  # Filter is not binary
            raise ValueError(f"Invalid LpArray filter: {item} is not a binary sequence")

        # Return LpArray with only "1" indices still in place
        if inplace:
            self.data = np.array([self[index] for index, i in zip(self.index, item) if i == 1])
            self.index = np.array([self.index[index] for index, i in enumerate(item) if i == 1])
        else:
            return LpArray(data=[self[index] for index, i in zip(self.index, item) if i == 1], index=[
                self.index[index] for index, i in enumerate(item) if i == 1], prob=self.prob)

    def get_subset(self, item: Sequence[float], by: Literal['index', 'location'] = 'index') -> 'LpArray':
        """Gets subset of `LpArray` based on indices (default) or location of items

        Args:
            item (Sequence[float]): Sequence of indices/locations to be selected in returned LpArray
            by (Literal['index', 'location']): If elements are to be selected by index or location

        Returns:
            LpArray: Containing only the subset of wanted elements
        """
        if by == 'index':
            return self.filter([int(i in item) for i in self.index])
        elif by == 'location':
            return self.filter([int(i in item) for i in range(len(self))])

    def operator(self, operation: Callable, other: Union['LpArray', pd.Series, np.ndarray, list, float],
                 drop: bool = True) -> 'LpArray':
        """Generic method for numerical operations, such as addition, subtraction, multiplication.

        Args:
            operation (Callable): Relevant operation (from operator library)
            other (Union['LpArray', pd.Series, np.ndarray, list, float]): Value or 1d data to be operated on
            drop (bool, optional): If `True`, remove non-shared indices, if `False` retain as original value. \
                Defaults to `False`

        Raises:
            ValueError: Attempt to operate on `LpArray` with `list` or `np.array` of different size

        Returns:
            LpArray: Array with relevant index and new values attributes
        """
        try:
            prob = self.prob or other.prob  # If other is type LpArray, prob can be acquired from other
        except AttributeError:
            prob = self.prob    # Otherwise prob is inherited from self

        match other:
            case LpArray() | pd.Series():
                if (other.index == self.index).all():
                    # Add values of self and other
                    return LpArray(operation(self.values, other.values), self.index, prob)
                elif drop:
                    intersect = np.intersect1d(self.index, other.index)  # Get common indices
                    # Apply operations to common indices
                    return LpArray(operation(self.values[intersect], other.values[intersect]), intersect, prob)

                intersect = np.intersect1d(self.index, other.index)  # Get common indices
                # Get non-commin indices
                diff_self = np.setdiff1d(self.index, other.index)
                diff_other = np.setdiff1d(other.index, self.index)

                # Concatenate arrays of indices
                index = np.concatenate([np.atleast_1d(a) for a in (intersect, diff_self, diff_other)])
                # Get values associated with index
                values = np.concatenate([np.atleast_1d(a) for a in (
                    operation(self[intersect], other[intersect]), self[diff_self], other[diff_other])])
                return LpArray(values, index, prob)

            case np.ndarray() | list():
                if len(other) != len(self):  # other must be same size, for convenience
                    raise ValueError(
                        f"cannot {operation.__name__} 'LpArray' to '{type(other).__name__}' of different size")
                return LpArray(operation(self.values, other), self.index, prob)

            case float() | int() | np.int64():
                return LpArray(operation(self.values, other), self.index, prob)

            case _:  # Invalid data type for operation
                raise TypeError(f"cannot {operation.__name__} types 'LpArray' and '{type(other).__name__}'")

    # Apply generic operation method to specific operations
    def __add__(self, other, drop=True):
        return self.operator(add, other, drop)
    __radd__ = __add__

    def __sub__(self, other, drop=True):
        return self.operator(sub, other, drop)

    def __rsub__(self, other, drop=True):
        return -self.operator(sub, other, drop)

    def __mul__(self, other, drop=True):
        return self.operator(mul, other, drop)
    __rmul__ = __mul__

    def __neg__(self):
        return LpArray(-self.values, self.index, self.prob)

    @property
    def shape(self):
        """Returns the shape of the values/index."""
        return self.values.shape


class LpMatrix:
    def to_tensor(self) -> 'LpTensor':
        pass


class LpTensor:
    def get_diag(self) -> LpMatrix:
        pass


if __name__ == '__main__':
    a = LpArray.variable('Bench', range(100), cat=int)
    a.index += 10
    b = LpArray.variable('Lineup', range(100), cat=int)
    x = a + b - 1
    for i in x:
        print(i)
