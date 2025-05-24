# C:/Python310/python.exe -m pytest
from oa_utils.pipeline import Pipeline
import more_itertools
from typing import Literal, Any
from typing_extensions import assert_type

def test_example_usage() -> None:
    p = Pipeline(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * x).sum()
    assert p == 120
    assert_type(p, int)

def test_map() -> None:
    p = Pipeline([1, 2, 3]).map(lambda x: x * 2)
    assert p == (2, 4, 6)
    assert_type(p, Pipeline[int])

def test_filter() -> None:
    p = Pipeline([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
    assert p == (2, 4)
    assert_type(p, Pipeline[int])

def test_zip() -> None:
    p = Pipeline([1, 2]).zip([10, 20])
    assert p == ((1, 10), (2, 20))
    assert_type(p, Pipeline[tuple[int, int]])

def test_zip_longest_1() -> None:
    p = Pipeline([1, 2]).zip_longest([10, 20, 30], fillvalue=None)
    assert p == ((1, 10), (2, 20), (None, 30))
    assert_type(p, Pipeline[tuple[int | None, int | None]])

def test_zip_longest_2() -> None:
    p = Pipeline([1, 2, 3]).zip_longest([10, 20], fillvalue=0)
    assert p == ((1, 10), (2, 20), (3, 0))
    assert_type(p, Pipeline[tuple[int, int]])

def test_zip_with() -> None:
    p = Pipeline([1, 2]).zip_with(lambda a, b: a + b, [10, 20])
    assert p == (11, 22)
    assert_type(p, Pipeline[int])

def test_starmap() -> None:
    p = Pipeline([(1, 2), (3, 4)]).starmap(lambda a, b: a + b)
    assert p == (3, 7)
    assert_type(p, Pipeline[int])

def test_cartesian_product() -> None:
    p = Pipeline([1, 2]).cartesian_product([10, 20])
    assert p == ((1, 10), (1, 20), (2, 10), (2, 20))
    assert_type(p, Pipeline[tuple[int, int]])

def test_sort_no_reverse() -> None:
    p = Pipeline([3, 1, 2]).sort()
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_sort_reverse() -> None:
    p = Pipeline([3, 1, 2]).sort(reverse=True)
    assert p == (3, 2, 1)
    assert_type(p, Pipeline[int])

def test_unique() -> None:
    p = Pipeline([1, 2, 2, 3]).unique()
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_slice() -> None:
    p = Pipeline([1, 2, 3, 4, 5]).slice(1, 4)
    assert p == (2, 3, 4)
    assert_type(p, Pipeline[int])

def test_take() -> None:
    p = Pipeline([1, 2, 3, 4]).take(2)
    assert p == (1, 2)
    assert_type(p, Pipeline[int])

def test_drop() -> None:
    p = Pipeline([1, 2, 3, 4]).drop(2)
    assert p == (3, 4)
    assert_type(p, Pipeline[int])

def test_enumerate() -> None:
    p = Pipeline(['a', 'b']).enumerate()
    assert p == ((0, 'a'), (1, 'b'))
    assert_type(p, Pipeline[tuple[int, str]])

def test_batch() -> None:
    p = Pipeline(range(1, 6)).batch(2)
    assert p == ((1, 2), (3, 4), (5,))
    assert_type(p, Pipeline[Pipeline[int]])

def test_batch_fill() -> None:
    p = Pipeline(range(1, 6)).batch_fill(2, fillvalue=0)
    assert p == ((1, 2), (3, 4), (5, 0))
    assert_type(p, Pipeline[Pipeline[int]])

def test_flatten() -> None:
    p = Pipeline([[1, 2], [3, 4]]).flatten()
    assert p == (1, 2, 3, 4)
    assert_type(p, Pipeline[int])

def test_for_each() -> None:
    # Not testing the printed output, but ensuring it returns the pipeline.
    p = Pipeline([1, 2, 3]).for_each(print)
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_for_self() -> None:
    # Not testing the output either, just that the pipeline returns itself.
    p = Pipeline([1, 2, 3]).for_self(lambda pipe: print(pipe.len()))
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_apply() -> None:
    p = Pipeline([[1, 2, 3], [4, 5, 6]]).apply(more_itertools.transpose)
    assert p == ((1, 4), (2, 5), (3, 6))
    assert_type(p, Pipeline[Any])

def test_print() -> None:
    # We can’t check printed text easily, but can check it returns the pipeline.
    p = Pipeline([1, 2, 3]).print("Numbers: ", end="\n\n")
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_print_label_only() -> None:
    p = Pipeline([1, 2, 3]).print("Numbers:", label_only=True)
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_pprint() -> None:
    # Again, just confirming the method returns the pipeline.
    p = Pipeline([1, 2, 3]).pprint("Numbers:", end="\n")
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_append() -> None:
    p = Pipeline([1, 2]).append(3)
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_prepend() -> None:
    p = Pipeline([2, 3]).prepend(1)
    assert p == (1, 2, 3)
    assert_type(p, Pipeline[int])

def test_extend() -> None:
    p = Pipeline([1, 2]).extend([3, 4])
    assert p == (1, 2, 3, 4)
    assert_type(p, Pipeline[int])

def test_insert() -> None:
    p = Pipeline([1, 2, 4]).insert(2, 3)
    assert p == (1, 2, 3, 4)
    assert_type(p, Pipeline[int])

def test_reverse() -> None:
    p = Pipeline([1, 2, 3]).reverse()
    assert p == (3, 2, 1)
    assert_type(p, Pipeline[int])

def test_group_by() -> None:
    p1 = Pipeline([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 2 == 0)
    assert p1 == ((False, (1, 3, 5)), (True, (2, 4, 6)))
    assert_type(p1, Pipeline[tuple[bool, Pipeline[int]]])
    
    people = [{'name': 'Roger', 'age': 25},
              {'name': 'Alice', 'age': 25},
              {'name': 'Bob', 'age': 11}]
    p2 = Pipeline(people).group_by(lambda person: person['age'])
    assert p2 == ((25, ({'name': 'Roger', 'age': 25}, {'name': 'Alice', 'age': 25})), (11, ({'name': 'Bob', 'age': 11},)))
    assert_type(p2, Pipeline[tuple[object, Pipeline[dict[str, object]]]])
    
    p3 = Pipeline(['Roger', 'Alice', 'Adam', 'Bob']).group_by(lambda name: name[0])
    assert p3 == (('R', ('Roger',)), ('A', ('Alice', 'Adam')), ('B', ('Bob',)))
    assert_type(p3, Pipeline[tuple[str, Pipeline[str]]])
    
def test_to_list() -> None:
    p = Pipeline([1, 2, 3]).to_list()
    assert p == [1, 2, 3]
    assert_type(p, list[int])

def test_to_set() -> None:
    p = Pipeline([1, 2, 3, 3]).to_set()
    assert p == {1, 2, 3}
    assert_type(p, set[int])

def test_to_dict() -> None:
    p = Pipeline([("a", 1), ("b", 2)]).to_dict()
    assert p == {"a": 1, "b": 2}
    assert_type(p, dict[str, int])

def test_first() -> None:
    p = Pipeline([1, 2, 3]).first()
    assert p == 1
    assert_type(p, int)

def test_last() -> None:
    p = Pipeline([1, 2, 3]).last()
    assert p == 3
    assert_type(p, int)

def test_reduce() -> None:
    p = Pipeline([104, 101, 108, 108, 111]).reduce(lambda acc, x: acc + chr(x), "")
    assert p == "hello"
    assert_type(p, str)

def test_reduce_non_empty() -> None:
    p = Pipeline([1, 2, 3]).reduce_non_empty(lambda acc, x: acc + x)
    assert p == 6
    assert_type(p, int)

def test_len() -> None:
    p = Pipeline([1, 2, 3]).len()
    assert p == 3
    assert_type(p, int)

def test_min() -> None:
    p = Pipeline([3, 1, 2]).min()
    assert p == 1
    assert_type(p, int)

def test_max() -> None:
    p = Pipeline([3, 1, 2]).max()
    assert p == 3
    assert_type(p, int)

def test_sum() -> None:
    p = Pipeline([1, 2, 3]).sum()
    assert p == 6
    assert_type(p, int)

def test_avg() -> None:
    p = Pipeline([1, 2, 3]).avg()
    assert p == 2.0
    assert_type(p, float)

def test_any_true() -> None:
    p = Pipeline([False, False, True]).any()
    assert p is True
    assert_type(p, Literal[True])

def test_any_false() -> None:
    p = Pipeline([False, False, False]).any()
    assert p is False
    assert_type(p, Literal[False])

def test_all_true() -> None:
    p = Pipeline([True, True, True]).all()
    assert p is True
    assert_type(p, Literal[True])

def test_all_false() -> None:
    p = Pipeline([True, False, True]).all()
    assert p is False
    assert_type(p, Literal[False])

def test_contains_true() -> None:
    p = Pipeline([1, 2, 3]).contains(2)
    assert p is True
    assert_type(p, Literal[True])

def test_contains_false() -> None:
    p = Pipeline([1, 2, 3]).contains(4)
    assert p is False
    assert_type(p, Literal[False])
