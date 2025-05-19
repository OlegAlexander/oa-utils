from __future__ import annotations
import functools
import copy
from typing import Callable, Iterable, Iterator, TypeVar, Any, Generic

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

class Pipeline(Generic[T], Iterable[T]):
    """Fluent wrapper around a list.
    
    >>> (Pipeline(range(10))
    ... .filter(lambda x: x % 2 == 0)
    ... .map(lambda x: x * x)
    ... .sum())                      
    120
    """

    def __init__(self, iterable: Iterable[T]) -> None:
        """
        >>> Pipeline([x for x in range(5)]).to_list()
        [0, 1, 2, 3, 4]
        """
        self._data = list(iterable)

    def to_list(self) -> list[T]:
        """
        >>> Pipeline([1, 2, 3]).to_list()
        [1, 2, 3]
        """
        return self._data

    def map(self, fn: Callable[[T], U]) -> Pipeline[U]:
        """
        >>> Pipeline([1, 2, 3]).map(lambda x: x * 2).to_list()
        [2, 4, 6]
        """
        return Pipeline(map(fn, self._data))

    def filter(self, pred: Callable[[T], bool]) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).to_list()
        [2, 4]
        """
        return Pipeline(filter(pred, self._data))

    def zip(self, other: Iterable[U]) -> Pipeline[tuple[T, U]]:
        """
        >>> Pipeline([1, 2]).zip([10, 20]).to_list()
        [(1, 10), (2, 20)]
        """
        return Pipeline(zip(self._data, other, strict=True))

    def zip_with(self, fn: Callable[[T, U], V], other: Iterable[U]) -> Pipeline[V]:
        """
        >>> Pipeline([1, 2]).zip_with(lambda a, b: a + b, [10, 20]).to_list()
        [11, 22]
        """
        return Pipeline(fn(a, b) for a, b in zip(self._data, other, strict=True))

    def sorted(self, key: Callable[[T], Any] | None = None, reverse: bool = False) -> Pipeline[T]:
        """
        >>> Pipeline([3, 1, 2]).sorted().to_list()
        [1, 2, 3]
        >>> Pipeline([3, 1, 2]).sorted(reverse=True).to_list()
        [3, 2, 1]
        """
        return Pipeline(sorted(self._data, key=key, reverse=reverse))  # type: ignore

    def unique(self) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 2, 3]).unique().to_list()
        [1, 2, 3]
        """
        return Pipeline(dict.fromkeys(self._data))

    def deepcopy(self) -> Pipeline[T]:
        """
        >>> orig_p = Pipeline([[1, 2], [3, 4]])
        >>> copy_p = orig_p.deepcopy()
        >>> copy_p.to_list() == orig_p.to_list()
        True
        >>> copy_p.to_list() is orig_p.to_list()
        False
        """
        return Pipeline(copy.deepcopy(self._data))
    
    def slice(self, start: int = 0, end: int | None = None, step: int = 1) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3, 4, 5]).slice(1, 4).to_list()
        [2, 3, 4]
        """
        if end is None:
            end = len(self._data)
        return Pipeline(self._data[start:end:step])

    def enumerate(self, start: int = 0) -> Pipeline[tuple[int, T]]:
        """
        >>> Pipeline(['a', 'b']).enumerate().to_list()
        [(0, 'a'), (1, 'b')]
        """
        return Pipeline(enumerate(self._data, start))

    def batched(self, n: int) -> Pipeline[Pipeline[T]]:
        """
        >>> p = Pipeline(range(1, 6)).batched(2)
        >>> [batch.to_list() for batch in p]
        [[1, 2], [3, 4], [5]]
        """
        return Pipeline(
            Pipeline(self._data[i : i + n]) for i in range(0, len(self._data), n)
        )

    def take(self, n: int) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3, 4]).take(2).to_list()
        [1, 2]
        """
        return Pipeline(self._data[:n])
    
    def drop(self, n: int) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3, 4]).drop(2).to_list()
        [3, 4]
        """
        return Pipeline(self._data[n:])
    
    def for_each(self, fn: Callable[[T], None]) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3]).for_each(print).to_list()
        1
        2
        3
        [1, 2, 3]
        """
        for item in self._data:
            fn(item)
        return self

    def print(self, label: str | None = None) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3]).print("Numbers:")
        Numbers: Pipeline([1, 2, 3])
        Pipeline([1, 2, 3])
        """
        if label is not None:
            print(label, self)
        else:
            print(self)
        return self

    # === Terminal methods ===

    def reduce(self, fn: Callable[[V, T], V], initial: V) -> V:
        """
        >>> Pipeline([1, 2, 3]).reduce(lambda acc, x: acc + x, 0)
        6
        """
        return functools.reduce(fn, self._data, initial)

    def len(self) -> int:
        """
        >>> Pipeline([1, 2, 3]).len()
        3
        """
        return len(self._data)
    
    def min(self) -> T:
        """
        >>> Pipeline([3, 1, 2]).min()
        1
        """
        return min(self._data)  # type: ignore
    
    def max(self) -> T:
        """
        >>> Pipeline([3, 1, 2]).max()
        3
        """
        return max(self._data)  # type: ignore
    
    def sum(self) -> T:
        """
        >>> Pipeline([1, 2, 3]).sum()
        6
        """
        return sum(self._data)  # type: ignore
    
    def any(self) -> bool:
        """
        >>> Pipeline([False, False, True]).any()
        True
        >>> Pipeline([False, False, False]).any()
        False
        """
        return any(self._data)
    
    def all(self) -> bool:
        """
        >>> Pipeline([True, True, True]).all()
        True
        >>> Pipeline([True, False, True]).all()
        False
        """
        return all(self._data)
    
    def contains(self, item: T) -> bool:
        """
        >>> Pipeline([1, 2, 3]).contains(2)
        True
        >>> Pipeline([1, 2, 3]).contains(4)
        False
        """
        return item in self._data

    # === Wrapped list methods ===

    def append(self, item: T) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2]).append(3).to_list()
        [1, 2, 3]
        """
        return Pipeline(self._data + [item])

    def extend(self, items: Iterable[T]) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2]).extend([3, 4]).to_list()
        [1, 2, 3, 4]
        """
        return Pipeline(self._data + list(items))
    
    def insert(self, index: int, item: T) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 4]).insert(2, 3).to_list()
        [1, 2, 3, 4]
        """
        new_data = copy.deepcopy(self._data)
        new_data.insert(index, item)
        return Pipeline(new_data)
    
    def remove(self, item: T) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3]).remove(2).to_list()
        [1, 3]
        """
        new_data = copy.deepcopy(self._data)
        new_data.remove(item)
        return Pipeline(new_data)
    
    def index(self, item: T) -> int:
        """
        >>> Pipeline(['a', 'b', 'c']).index('b')
        1
        """
        return self._data.index(item)
    
    def count(self, item: T) -> int:
        """
        >>> Pipeline([1, 2, 2, 3]).count(2)
        2
        """
        return self._data.count(item)
    
    def reverse(self) -> Pipeline[T]:
        """
        >>> Pipeline([1, 2, 3]).reverse().to_list()
        [3, 2, 1]
        """
        return Pipeline(reversed(self._data))

    # === Dunder methods ===

    def __eq__(self, other: object) -> bool:
        """
        >>> Pipeline([1, 2, 3]) == Pipeline([1, 2, 3])
        True
        >>> Pipeline([1, 2]) == Pipeline([2, 1])
        False
        """
        if not isinstance(other, Pipeline):
            return False
        return self._data == other._data 

    def __str__(self) -> str:
        """
        >>> str(Pipeline([1, 2, 3]))
        'Pipeline([1, 2, 3])'
        """
        return f"Pipeline({self._data})"

    def __repr__(self) -> str:
        """
        >>> Pipeline([1, 2, 3])
        Pipeline([1, 2, 3])
        """
        return str(self)

    def __iter__(self) -> Iterator[T]:
        """
        >>> list(Pipeline([1, 2, 3]))
        [1, 2, 3]
        """
        return iter(self._data)

if __name__ == "__main__":
    # Interpreter usage: 
    # from importlib import reload; import oa_utils.pipeline; reload(oa_utils.pipeline); from oa_utils.pipeline import Pipeline
    import doctest
    doctest.testmod()
    