# -*- coding: utf-8 -*-
"""vec_map / vec_map_pair 单元测试与属性测试（Req 3.12）。"""
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

import normalizer
from normalizer import vec_map, vec_map_pair


def test_vec_map_does_not_merge_int_str_float_bool():
    col = pd.Series([1, "1", 1.0, True], dtype=object)
    seen = []

    def fn(u):
        seen.append(u)
        return u

    out = vec_map(col, fn)
    assert seen == ["1", "1.0", "True"]
    assert out.tolist() == ["1", "1", "1.0", "True"]


def test_vec_map_results_equal_scalar_on_original():
    col = pd.Series([1, "1", 1.0, True, None, float("nan"), " a b "], dtype=object)
    out = vec_map(col, normalizer.norm_text)
    expected = [normalizer.norm_text(x) for x in col]
    assert out.tolist() == expected


def test_vec_map_calls_fn_once_per_unique_and_keeps_index_order():
    col = pd.Series(["a", "b", "a", "c", "b"], index=[10, 7, 3, 99, 0])
    calls = []

    def fn(u):
        calls.append(u)
        return u.upper()

    out = vec_map(col, fn)
    assert calls == ["a", "b", "c"]
    assert out.index.tolist() == [10, 7, 3, 99, 0]
    assert out.tolist() == ["A", "B", "A", "C", "B"]


def test_vec_map_tuple_outputs_preserved():
    col = pd.Series(["a|b", "c", "a|b"])
    out = vec_map(col, lambda u: tuple(u.split("|")))
    assert out.tolist() == [("a", "b"), ("c",), ("a", "b")]
    assert out.ndim == 1


def test_vec_map_equal_length_tuples_stay_1d():
    col = pd.Series(["x", "y"])
    out = vec_map(col, lambda u: (u, u))
    assert len(out) == 2
    assert out.tolist() == [("x", "x"), ("y", "y")]


def test_vec_map_empty():
    out = vec_map(pd.Series([], dtype=object), lambda u: u)
    assert len(out) == 0


def test_vec_map_nan_none_as_missing():
    col = pd.Series([None, float("nan"), "None", "nan", "ok"], dtype=object)
    out = vec_map(col, normalizer.norm_text)
    assert out.tolist() == [None, None, None, None, "ok"]


def test_vec_map_pair_calls_once_per_unique_pair_with_original_objects():
    first = pd.Series([1, "1", 1, 1.0, True, 1], dtype=object)
    second = pd.Series(["a", "a", "a", "a", "a", "b"], dtype=object)
    calls = []

    def fn(p, v):
        calls.append((p, v))
        return (repr(p), v)

    o1, o2 = vec_map_pair(first, second, fn, 2)
    # 键为 str(x)：1 与 "1" 同键（设计如此），1.0、True 各自独立
    # 组：(1,a) (1.0,a) (True,a) (1,b)，每组第一行原始对象传给 fn
    assert len(calls) == 4
    assert [type(c[0]) for c in calls] == [int, float, bool, int]
    assert o1.tolist() == ["1", "1", "1", "1.0", "True", "1"]
    assert o2.tolist() == ["a", "a", "a", "a", "a", "b"]


def test_vec_map_pair_tuple_elements_and_multi_output():
    first = pd.Series(["p", "p", "q"], index=[5, 6, 7])
    second = pd.Series(["x|y", "x|y", "z"], index=[5, 6, 7])
    calls = []

    def fn(p, v):
        calls.append((p, v))
        frags = tuple(v.split("|"))
        return frags, len(frags), p

    frag, n, pp = vec_map_pair(first, second, fn, 3)
    assert len(calls) == 2
    assert frag.tolist() == [("x", "y"), ("x", "y"), ("z",)]
    assert n.tolist() == [2, 2, 1]
    assert pp.tolist() == ["p", "p", "q"]
    for s in (frag, n, pp):
        assert s.index.tolist() == [5, 6, 7]


def test_vec_map_pair_first_row_original_object_used():
    first = pd.Series([None, None], dtype=object)
    second = pd.Series([1, 1], dtype=object)
    got = []

    def fn(p, v):
        got.append((p, v))
        return (v,)

    (o,) = vec_map_pair(first, second, fn, 1)
    assert got == [(None, 1)]
    assert o.tolist() == [1, 1]


def test_vec_map_pair_empty():
    outs = vec_map_pair(
        pd.Series([], dtype=object), pd.Series([], dtype=object), lambda p, v: (p, v), 2
    )
    assert len(outs) == 2
    assert all(len(s) == 0 for s in outs)


def test_vec_map_pair_order_preserved():
    first = pd.Series(["b", "a", "b", "a", "c"])
    second = pd.Series(["1", "2", "1", "3", "1"])
    (o,) = vec_map_pair(first, second, lambda p, v: (p + v,), 1)
    assert o.tolist() == ["b1", "a2", "b1", "a3", "c1"]


@given(st.lists(st.one_of(st.integers(-3, 3), st.text(max_size=4), st.none(), st.booleans(),
                          st.floats(allow_nan=True, allow_infinity=False)), max_size=40))
def test_property_vec_map_equals_scalar(values):
    """**Validates: Requirements 3.12**"""
    col = pd.Series(values, dtype=object)
    out = vec_map(col, lambda u: normalizer.norm_text(u))
    expected = [normalizer.norm_text(x) for x in values]
    assert out.tolist() == expected
    assert len(out) == len(values)
