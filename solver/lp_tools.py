"""Array-based structures for linear programming variables, simplifying the implementation of vector-based LP problems.

Classes allow multi-dimensional arrays of objects to be processed in parallel using element-wise operations, with full
support for `pulp.LpVariable` objects, unlike `numpy` or `pandas` classes. Arithmetic operations avaible across and
between classes, and constraints can be added when associated with pulp.LpProblem (outer class for `pulp.LpProblem`).

Syntax is very similar to relevant Pandas data structures.

    Typical usage example:

    # TODO

"""
import pulp
import numpy as np
import pandas as pd
from typing import Generator, Union, Literal, Callable, Sequence, Any
from operator import add, sub, mul
from collections.abc import Generator
from collections import defaultdict


from pyparsing import col


class LpProblem:
    """Outer class for pulp.LpProblem, with syntactical support for constraint addition with `LpArray`, \
    `LpMatrix`, and `LpTensor` objects.
    """

    def __init__(self, name="NoName", sense=pulp.const.LpMaximize):
        self.model = pulp.LpProblem(name, sense)

    def __iadd__(self, other: Generator | pulp.pulp.LpConstraint |
                 pulp.pulp.LpAffineExpression | pulp.pulp.LpVariable = None) -> None:
        """pulp.LpProblem-style constraint & objective addition

        Args:
            other (Generator | pulp.pulp.LpConstraint | pulp.pulp.LpAffineExpression | pulp.pulp.LpVariable,\
                optional): LP object or generator of LP objects. Defaults to None.

        Raises:
            TypeError: Type not appropriate for constraint or objective.
        """
        match other:
            case Generator() as constraints:
                for const in constraints:
                    self.model += const

            case pulp.pulp.LpConstraint() | pulp.pulp.LpAffineExpression() | pulp.pulp.LpVariable() as const:
                self.model += const

            case _:
                raise TypeError(f"Type {type(other).__name__} cannot be applied to LP problem")

    def solve(self, time_limit=10000, optimizer=pulp.GUROBI, message=True) -> None:
        """Solve LpModel"""
        return self.model.solve(optimizer(timeLimit=time_limit, msg=message))

    def __str__(self) -> str:
        return str(self.model)


class LpIndexer:
    def __init__(self, object: Union['LpArray', 'LpMatrix', 'LpTensor'],
                 index_type: Literal['index', 'location'] = 'index'):
        """Utility to assist indexwise referenceing in LpArray, LpMatrix and LpTensor object types

        Args:
            object (Union['LpArray', 'LpMatrix', 'LpTensor']): Instance of `LpArray`, `LpMatrix`, or `LpTensor`.
            index_type (Literal['index', 'location'], optional): Locate by index or location. Defaults to 'index'.
        """
        self.object = object

        if index_type == 'location':
            self.object.index = np.array(range(len(object)))
        if object.dim > 2:
            self.levels = object.levels
        elif object.dim > 1:
            self.columns = object.columns

    def __getitem__(self, item:  float | Sequence[float | bool]):
        match self.object:
            case LpArray():
                subset = self.object[item]
                if len(subset) == 1:
                    return subset.values[0]
                elif len(subset) > 1:
                    return subset.values

            case LpMatrix():    # TODO
                pass

            case LpTensor():
                pass

    def __setitem__(self, item, value):
        match self.object:
            case LpArray():
                self.object[item] = value


class LpArray:
    """1-Dimensional `pandas.Series`-like structure with support for `pulp.LpVariable` objects, and support for \
    applying element-wise PuLP constraints when associated with `pulp.LpProblem`.
    """

    def __init__(self, data: Sequence | 'LpArray' = None, index: Sequence[float] = None):
        """This class models an array with the following parameters:

        Args:
            data (Sequence, optional): Values contained in `LpArray`. Defaults to `None`
            index (Sequence[float], optional): Indices (usually `int`) paired to corrresponding values. Defaults to \
                 `None`
        """
        # Default for values and index are empty
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

        self.values, self.index = np.array(data), np.array(index)
        self.dim = 1

    @classmethod
    def from_dict(cls, data: dict = None, sort_index: bool = False) -> 'LpArray':
        """Initialise `LpArray` with data from `dict`, with the following parameters:

        Args:
            data (dict, optional): `dict` (length n) object containing `{index[0]: values[0], index[1]: values[1], \
                  ..., index[n]: values[n]}`. Defaults to `None`
            sort_index (bool, optional): If `True`, return `LpArray.from_dict(dict(sorted(dict.values())), ...)`

        Returns:
            LpArray: with values `dict.values()` and index `dict.keys()`
        """
        if sort_index:
            data = dict(sorted(data.items()))  #  Sort dict by keys
        return cls(list(data.values()), list(data.keys()))    # Initialise class instance from dict

    @classmethod
    def variable(cls, name: str = "NoName", index: Sequence[float] = None, lower: float = None, upper: float = None,
                 cat: type[bool | int | float] = None) -> 'LpArray[pulp.LpVariable]':
        """Initialise `LpArray` containing `pulp.LpVariable` objects, with the following parameters:

        Args:
            name (str): Name for `pulp.LpVariable`. Defaults to `"NoName"`
            index (Sequence[float], optional): Index of returned `LpArray`. Defaults to `None`
            lower (float, optional): Lower bound for variables to be created. Defaults to `None`
            upper (float, optional): Upper bound for variables to be created. Defaults to `None`
            cat (type[bool | int | float], optional): Category of variables: `bool`, `int`, or `float`.\
                Defaults to `float`

        Returns:
            LpArray[pulp.LpVariable]: Values named "{name}_{i}" for all i in index
        """
        # Generate and process dict of pulp.LpVariable objects
        return cls.from_dict(pulp.LpVariable.dict(name, index, lower, upper, (
            'Binary', 'Integer', 'Continuous')[(bool, int, float).index(cat)]))

    def __str__(self) -> str:
        """Convert LpArray to string for easy readability.

        Returns:
            str: pandas.Series-style overview of array
        """
        if self.values.size == 0:   # If empty LpArray
            return 'LpArray([])'

        return '\n'.join(str(pd.Series([str(i) for i in self.values], self.index)).split(
            '\n')[:-1]) + f"\nLength: {len(self)}, dtype: {type(self.iloc[0]).__name__}\n"

    def __len__(self) -> int:
        """Returns the length of the index."""
        return len(self.index)

    def __iter__(self) -> Generator:
        """Iterate through `self.values`"""
        for value in self.values:
            yield value

    def __getitem__(self, item: float | Sequence[float | bool], by: Literal['index', 'location'] = 'index') -> Any:
        """Returns subset of self, by index *OR* binary inclusion sequence.\
            Works only if item is not repeated in `self.index`.

        Args:
            item (float | Sequence[bool]): Index corrresponding to wanted value, or sequence of binary values, where \
                nth element corresponds to whether to include nth index/value pair in output `LpArray`
            by (Literal['index', 'location']): Treat item as index or location reference. Defaults to `'index'`

        Raises:
            ValueError: Invalid index or filter

        Returns:
            Any: Value corresponding to passed index, or `LpArray` corresponding to passed binary inclusion sequence
        """
        try:
            return self.filter(item)    # Try item as binary filter
        except (ValueError, TypeError):  # Non-binary or zero length
            return self.get_subset(item, by)    # Try item as sequence of indices

    def __setitem__(self, key: float | Sequence[float], value: Any,
                    by: Literal['index', 'location'] = 'index') -> None:
        """Set key (or sequence of keys) in `LpArray` to given value (set of values)

        Args:
            key (float | Sequence[float]): Key of item to be set
            value (Any): Value (or sequence of values) of set item
            by (Literal['index', 'location'], optional): Whether key corresponds to index or location.\
                Defaults to 'index'.
        """
        match key:
            case float() | int() | np.int64():  # Single item to be set
                if by == 'index':
                    try:
                        key = self.index.tolist().index(key)    # Get location of index value
                    except ValueError:  # Key not in index
                        self.index, self.values = np.append(self.index, key), np.append(self.values, value)
                        return
                self.values[key] = value

            case [*keys]:   # Multiple items to be set
                if by == 'index':
                    keys = [self.index.tolist().index(k) for k in keys]    # Get locations of index values
                try:    # Assume case: value is a sequence
                    if len(keys) != len(value):
                        raise ValueError(f"{keys} and {value} are not the same length ({len(keys)} and {len(value)})")
                except TypeError:   # Case: value is not a sequence
                    for k in keys:
                        self.__setitem__(k, value, by='location')
                    return
                for k, val in zip(keys, value):
                    self.__setitem__(k, val, by='location')

            case _:  # Unexpected key type
                raise TypeError(f"Type {type(key).__name__} is not a valid {by} selection")

    def filter(self, item: Sequence[bool], inplace: bool = False) -> Union[None, 'LpArray']:
        """Filter `LpArray` using a binary sequence of the same length.

        Args:
            item (Sequence[bool]): Squence of `bool` values, indicating whether to include nth value in nth entry
            inplace (bool): True => filter existing object. False => return new filtered object. Defaults to `False`

        Raises:
            ValueError: Attempt to filter with non-binary or differently-sized data

         Returns:
             LpArray | None: Filtered LpArray if not inplace
         """
        if len(item) != len(self):  # Filter has the wrong length
            raise ValueError(
                f"Invalid LpArray filter: {item} does not have the same length as LpArray \
                    ({len(item)} vs. {len(self)})")

        if not all([(i in (0, 1)) for i in item]):  # Filter is not binary
            raise ValueError(f"Invalid LpArray filter: {item} is not a binary sequence")

        # Return LpArray with only "1" indices still in place, removing "0" indices and corresponding values
        if inplace:
            self.values, self.index = self.values[item], self.index[item]
        else:
            return LpArray(data=self.values[item], index=self.index[item])

    def get_subset(self, item: Sequence[float], by: Literal['index', 'location'] = 'index',
                   sorted: bool = False) -> 'LpArray':
        """Gets subset of `LpArray` based on indices (default) or location of items

        Args:
            item (Sequence[float]): Sequence of indices/locations to be selected in returned LpArray
            by (Literal['index', 'location']): If elements are to be selected by index or location.\
                Defailts to `'index'`
            sorted (bool): If True, repeating/unsorted items are condensed/sorted (e.g. (1, 1, 0) -> (0, 1)).\
                Defaults to `False`

        Returns:
            LpArray: Containing only the subset of wanted elements
        """
        if type(item) in (float, int, np.int64):
            item = [item]
        if sorted:
            if by == 'index':
                # Filter self by corresponding index
                return self.filter([int(i in item) for i in self.index], inplace=True)
            elif by == 'location':
                # Filter self by corresponding location
                return self.filter([int(i in item) for i in range(len(self))], inplace=True)
            raise(ValueError(f"Argument 'by' must be one of ('index', 'location'), not {by}"))

        if by == 'index':
            try:

                item = [np.where(self.index == i)[0][0] for i in item]  # Change item from index to location reference
            except IndexError:  # Item not in index
                return LpArray()
        return LpArray(self.values[item], self.index[item])

    @property
    def loc(self):
        return LpIndexer(self, 'index')

    @property
    def iloc(self):
        return LpIndexer(self, 'location')

    def drop(self, *item: float | Sequence[float], by: Literal['index', 'location'] = 'index',
             inplace: bool = False) -> Union[None, 'LpArray']:
        """Remove element from LpArray by its index or location

        Args:
            item (float | Sequence[float]): Index/location or sequence of indices/locations to be dropped
            by (Literal['index', 'location'], optional): If elements are to be selected by index or location.\
                Defaults to `'index'`
            inplace (bool): True => drop from existing object. False => return copy with elements dropped.\
                Defaults to `False`

        Raises:
            TypeError: If item type is not a float or sequence
        """
        match item:
            case float() | int() | np.int64():  # Drop single element
                if by == 'index':
                    item = self.index.tolist().index(item)

            case [*items]:  # Drop sequence of elements
                if by == 'index':
                    item = [self.index.tolist().index(i) for i in items]

            case _:  # Unrecognized type of element index/location
                raise TypeError(f"Type {type(item).__name__} cannot be dropped")

        if not inplace:
            return LpArray(np.delete(self.values, item), np.delete(self.index, item))
        self.index, self.values = np.delete(self.index, item), np.delete(self.values, item)

    def remove(self, value: Any, inplace: bool = False) -> Union[None, 'LpArray']:
        """Remove element from `LpArray` by value. Warning: does not support LpArrays containing `pulp.LpVariable` or\
            `pulp.LpAffineExpression` objects

        Args:
            value (Any): Value of element to be removed
        """
        match value:
            case float() | int() | np.int64():  # Remove single index
                item = self.values.tolist().index(value)

            case [*values]:  # Remove series of indices
                item = [self.values.tolist().index(val) for val in values]

        if not inplace:
            return self.drop(item, by='location')
        self.drop(item, by='location', inplace=True)

    def lag(self, lag_value: int = 1, inplace: bool = False) -> Union[None, 'LpArray']:
        """Translate (in increasing directions) LpArray index by given value"""
        if not inplace:
            return LpArray(self.values, self.index + lag_value)
        self.index += lag_value

    def sort_index(self, inplace=False) -> None:
        """Sort index of LpArray"""
        if not inplace:
            return LpArray(np.array([val for _, val in sorted(zip(self.index, self.values))]),
                           sorted(self.index))
        self.values = np.array([val for _, val in sorted(zip(self.index, self.values))])
        self.index = sorted(self.index)

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
        match other:
            case LpArray() | pd.Series():
                if (len(self) == len(other)) and (other.index == self.index).all():  # Self and other have same index
                    # Add values of self and other
                    return LpArray(operation(self.values, other.values), self.index)
                elif drop:
                    intersect = np.intersect1d(self.index, other.index)  # Get common indices
                    # Apply operations to common indices
                    return LpArray(operation(self[intersect], other[intersect]), intersect)

                intersect = np.intersect1d(self.index, other.index)  # Get common indices
                # Get non-common indices
                diff_self = np.setdiff1d(self.index, other.index)
                diff_other = np.setdiff1d(other.index, self.index)

                # Concatenate arrays of indices
                index = np.concatenate([np.atleast_1d(a) for a in (diff_self, intersect, diff_other)])
                # Get values associated with index
                values = np.concatenate([np.atleast_1d(a) for a in (self[diff_self], operation(
                    self[intersect], other[intersect]), other[diff_other])])

                if np.all(index[:-1] <= index[1:]):  # Case if index is sorted
                    return LpArray(values, index)

                # Put all index/value pairs into dict
                values_dict = {index: value for index, value in zip(
                    intersect, operation(self[intersect], other[intersect]))} | {
                    index: value for index, value in zip(diff_self, self[diff_self])} | {
                        index: value for index, value in zip(diff_other, other[diff_other])}
                return_data = (list(zip(*sorted(values_dict.items()))))  # Unzip sorted dict
                return LpArray(return_data[1], return_data[0])    # LpArray containing all data

            case np.ndarray() | list():
                if len(other) != len(self):  # other must be same size, for convenience
                    raise ValueError(
                        f"cannot {operation.__name__} 'LpArray' to '{type(other).__name__}' of different size")
                return LpArray(operation(self.values, other), self.index)

            case float() | int() | np.int64():
                return LpArray(operation(self.values, other), self.index)

            case _:  # Invalid data type for operation
                raise TypeError(f"cannot {operation.__name__} types 'LpArray' and '{type(other).__name__}'")

    # Apply generic operation method to specific operations
    def __add__(self, other, drop=True):
        return self.operator(add, other, drop)
    __radd__ = __add__

    def __or__(self, other, drop=False):
        return self.__add__(other, drop=drop)

    def __sub__(self, other, drop=True):
        return self.operator(sub, other, drop)

    def __rsub__(self, other, drop=True):
        return -self.operator(sub, other, drop)

    def __mul__(self, other, drop=True):
        return self.operator(mul, other, drop)
    __rmul__ = __mul__

    def __neg__(self):
        return LpArray(-self.values, self.index)

    def sum(self):
        return sum(self)

    @ property
    def shape(self) -> tuple:
        """Returns the shape of the values/index."""
        return self.values.shape

    def add_constraint(self, other: Union[float, 'LpArray', pd.Series, Sequence],
                       sense: type[pulp.const.LpConstraintEQ | pulp.const.LpConstraintLE | pulp.const.LpConstraintGE]
                       ) -> Generator[pulp.LpConstraint]:
        """General method for elementwise constraint addition. Returns a `generator` of pulp constraints to be\
            applied to an `LpProblem` object.

        Args:
            other (Union[float, 'LpArray', pd.Series, Sequence]): Object for RHS of constraints (either one term of a\
                sequence of terms with same length as LpArray)
            sense (type[pulp.const.LpConstraintEQ | pulp.const.LpConstraintLE | pulp.const.LpConstraintGE]):\
                Specify constraint type from (==, >=, <=).

        Returns:
            Generator[pulp.LpConstraint]: Iterable generator of elementwise constraints
        """
        match other:
            case float() | int() | np.int64() | pulp.LpAffineExpression() | pulp.pulp.LpVariable():
                return (pulp.LpConstraint(value, sense, rhs=other) for value in self.values)

            case LpArray() | pd.Series():  # Apply constraints between two LpArrays
                if len(other) == len(self):  # Elementwise constraints
                    return (pulp.LpConstraint(value, sense, rhs=rhs) for value, rhs in zip(self.values, other.values))
                elif len(other) == 1:   # Same constraint on all elements
                    return self.add_constraint(other.loc[0], sense)
                raise ValueError(
                    f"Invalid attempt to apply LP constraint: lengths do not match ({len(self)} vs {len(other)})")

            case [*others]:  # Iterable sequence
                if len(other) == len(self):  # Elementwise constraints
                    return (pulp.LpConstraint(value, sense, rhs=rhs) for value, rhs in zip(self.values, other))
                elif len(other) == 1:   # Same constraint on all elements
                    return self.add_constraint(other[0], sense)
                raise ValueError(
                    f"Invalid attempt to apply LP constraint: lengths do not match ({len(self)} vs {len(other)})")

            case _:  # Unknown type
                raise TypeError(f"LP Constriant cannot be applied to {other}")

    def __eq__(self, other):
        return self.add_constraint(other, pulp.const.LpConstraintEQ)

    def __le__(self, other):
        return self.add_constraint(other, pulp.const.LpConstraintLE)

    def __ge__(self, other):
        return self.add_constraint(other, pulp.const.LpConstraintGE)


class LpMatrix:
    def __init__(self, data: Sequence[Sequence | LpArray] | np.ndarray | 'LpMatrix' = None,
                 index: Sequence[float] = None, columns: Sequence[float | int | np.int64 | str] = None):
        """2-Dimensional `pandas.DataFrame`-like structure with support for `pulp.LpVariable` objects, and support for \
        applying element-wise PuLP constraints when associated with `pulp.LpProblem`.

        Args:
            data (Sequence[Sequence  |  LpArray] | np.ndarray | LpMatrix, optional): 1d or 2d data.\
                Defaults to `None`.
            index (Sequence[float], optional): Indices (usually `int`) paired to corrresponding rows.\
                Defaults to `None`.
            columns (Sequence[Any], optional): Names for corrresponding columns. Defaults to `None`.
        """
        if data is None:
            data = []
        self.values = np.array(data)

        if index is None:
            index = range(len(self.values))
        self.index = np.array(index)

        if columns is None:
            if self.values.ndim == 1:
                columns = []
            elif self.values.ndim == 2:
                columns = range(len(self.values[0]))
            else:
                raise ValueError(f"{self.values.ndim} is not a valid # of dimensions for LpMatrix data")
        self.columns = np.array(columns)

        try:
            if len(self.values) != len(self.index):
                raise ValueError(f"data and index length are not compatible")
        except TypeError:
            raise TypeError("data and/or index are not sequential")
        try:
            if self.values.ndim == 2 and len(self.values[0]) != len(self.columns):
                raise ValueError("data and column size are not compatible")
        except TypeError:
            raise TypeError("columns are not sequential")

    @classmethod
    def from_dict(cls, data) -> 'LpMatrix':
        """Generate LpMatrix instance from dict with following structure:\
            `{col_1: {index_1: i_11, ..., index_n: i_n1}, ...col_m: {index_1: i_1m, ..., index_n: i_nm}`"""
        return cls(np.array([list(d.values()) for d in data.values()]).T,
                   list(list(data.values())[0].keys()), list(data.keys()))

    @classmethod
    def variable(cls, name: str = "NoName", index: Sequence[float] = None,
                 columns: Sequence[Any] = None, lower: float = None, upper: float = None,
                 cat: type[bool | int | float] = None) -> 'LpMatrix[pulp.LpVariable]':
        """Initialise `LpMatrix` containing `pulp.LpVariable` objects, with the following parameters:

        Args:
            name (str, optional): Name for `pulp.LpVariable`. Defaults to `"NoName"`.
            index (Sequence[float], optional): Index of returned `LpMatrix`. Defaults to `None`.
            columns (Sequence[Any], optional): Column names of returned `LpMatrix`. Defaults to `None`.
            lower (float, optional): Lower bound for variables to be created. Defaults to `None`.
            upper (float, optional): Upper bound for variables to be created. Defaults to `None`.
            cat (type[bool | int | float], optional): Category of variables: `bool`, `int`, or `float`.\
                Defaults to `None`.

        Returns:
            LpMatrix[pulp.LpVariable]: Values named "{name}_{i}_{col}" for i, col in index, columns
        """
        # Create dict of dicts of pulp.LpVariable objects
        dct = {i: pulp.LpVariable.dict(f"{name}_{i}", columns, lower, upper, (
            'Binary', 'Integer', 'Continuous')[(bool, int, float).index(cat)]) for i in index}

        # Transpose layers of dict
        d = defaultdict(dict)
        for key1, inner in dct.items():
            for key2, value in inner.items():
                d[key2][key1] = value
        return cls.from_dict(d)

    def __str__(self) -> str:
        return str(pd.DataFrame(self.values, self.index, self.columns))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Generator:
        for column in self.columns:
            yield column

    def __getitem__(self, item: float | str | Sequence[float | bool] | tuple, by: Literal['index', 'location'] = 'index'):
        try:
            return self.filter(item)    # Try item as binary filter
        except (ValueError, TypeError):  # Non-binary or zero length
            return self.get_subset(item, by)    # Try item as sequence of indices

    def filter(self, item: Sequence[bool], inplace: bool = False) -> Union[None, 'LpMatrix']:
        """Filter `LpMatrix` using a binary sequence of the same length.

        Args:
            item (Sequence[bool]): Squence of `bool` values, indicating whether to include nth row
            inplace (bool): True => filter existing object. False => return new filtered object. Defaults to `False`

        Raises:
            ValueError: Attempt to filter with non-binary or differently-sized data

         Returns:
             LpMatrix | None: Filtered LpMatrix if not inplace
        """
        if len(item) != len(self):  # Filter has the wrong length
            raise ValueError(
                f"Invalid LpMatrix filter: {item} does not have the same length as LpMatrix \
                    ({len(item)} vs. {len(self)})")

        if not all([(i in (0, 1)) for i in item]):  # Filter is not binary
            raise ValueError(f"Invalid LpMatrix filter: {item} is not a binary sequence")

        if inplace:
            self.values, self.index = self.values[item], self.index[item]

        return LpMatrix(self.values[item], self.index[item], self.columns)

    def get_subset(self, item: float | str | Sequence[float | bool] | tuple,
                   by: Literal['index', 'location'] = 'index') -> Union[None, 'LpMatrix']:
        match item:
            case float() | int() | np.int64() | str() as col:   # Single column wanted
                if by == 'index':
                    try:
                        col = self.columns.tolist().index(col)
                    except ValueError:
                        raise ValueError(f"{col} is not in columns")
                return LpArray(self.values[:, col], self.index)

            case ([*indices], [*columns]):  # Non-trivial subset of index and columns
                if by == 'index':
                    try:
                        indices, columns = [self.index.tolist().index(i) for i in indices], [
                            self.columns.tolist().index(col) for col in columns]
                    except ValueError:
                        raise ValueError("Not all given indices and columns are in LpMatrix")

                return LpMatrix(self.values[[[i] for i in indices], columns], self.index[indices], self.columns[columns])

            case ([*indices], col):  # Non-trivial subset of index across one column
                return self[col].get_subset(indices, by=by)

            case (i, [*columns]):  # Non-trivial subset of columns across one index
                return self.T[i].get_subset(columns, by=by)

            case (i, col):  # Single entry in LpMatrix
                return self[col].get_subset(i, by=by)

            case _:
                raise TypeError(f"{item} is not a valid index")

    def drop(self) -> Union[None, 'LpMatrix']:
        pass

    def remove(self) -> Union[None, 'LpMatrix']:
        pass

    def operation(self) -> 'LpMatrix':
        pass

    def transpose(self, inplace=False) -> 'LpMatrix':
        if inplace:
            self.index, self.columns = self.columns, self.index
            self.values = self.values.T
        return LpMatrix(self.values.T, self.columns, self.index)

    @property
    def T(self):
        return self.transpose()

    def to_tensor(self) -> 'LpTensor':
        pass


class LpTensor:
    def get_diag(self) -> LpMatrix:
        pass


if __name__ == '__main__':
    prob = LpProblem()
    m = LpMatrix([[1, 2, 3], [4, 5, 6]], columns=[1, 2, 3])
    d = LpMatrix.from_dict({2: {1: 1, 2: 2}, 3: {1: 3, 2: 4}})
    print(d)
    v = LpMatrix.variable('lineup', range(10), range(5), 0, 1, bool)
    v.index += 1
    print(v)
    print(v[(1, 2), (1, 2)])
