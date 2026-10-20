# -*- coding: utf-8 -*-
"""分数字段数值化属性测试（任务 5.4）。

Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
**Validates: Requirements 3.9**
"""

import math
import re
import unicodedata

from hypothesis import given, settings, strategies as st

from normalizer import norm_number

_NUMBER_TEXT_RE = re.compile(r"[0-9]+(\.[0-9]+)?")

# ---- 本文件内的局部策略 ----

# 非零有限数值单元格：整数与浮点（含负数，需求只对“文本”限制无符号）
_nonzero_ints = st.integers(min_value=-10**12, max_value=10**12).filter(lambda v: v != 0)
_nonzero_floats = st.floats(allow_nan=False, allow_infinity=False).filter(lambda v: v != 0)
_nonzero_cells = st.one_of(_nonzero_ints, _nonzero_floats)

# 正整数 / 正的“整数.小数”文本，数值上非零，且长度不超过 float 精确范围
_int_part = st.integers(min_value=0, max_value=10**12).map(str)
_frac_part = st.text(alphabet="0123456789", min_size=1, max_size=6)
_decimal_text = st.one_of(
    _int_part,
    st.builds(lambda i, f: f"{i}.{f}", _int_part, _frac_part),
).filter(lambda s: float(s) != 0)

# 补前导零的文本（允许，如 "007"）
_zero_padded_text = st.builds(
    lambda z, s: "0" * z + s, st.integers(min_value=1, max_value=3), _decimal_text
)

# 数值为 0 的文本："0"、"0.0"、"000.000"
_zero_text = st.builds(
    lambda i, f: i + ("." + f if f else ""),
    st.text(alphabet="0", min_size=1, max_size=4),
    st.one_of(st.just(""), st.text(alphabet="0", min_size=1, max_size=4)),
)


def _is_parsable(s):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    return _NUMBER_TEXT_RE.fullmatch(t) is not None


# 不能整串解析为数字的文本：既有随机文本，也有需求点名的反例
_examples_unparsable = st.sampled_from(
    ["暂无", "—", "1,234", "+5", "-1", "-0.5", "1e5", "1E3", "1.", ".5", "1.2.3",
     "0x10", "12a", "a12", "NaN", "nan", "inf", "-inf", "Infinity", "", " ", "None",
     "null", "无", "未知", "不详", "-", "--", "/", "1 2.5x", "１２,３"]
)
_random_unparsable = st.text(max_size=12).filter(lambda s: not _is_parsable(s))
_unparsable_text = st.one_of(_examples_unparsable, _random_unparsable)


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(v=_nonzero_cells)
def test_nonzero_numeric_cell_is_kept(v):
    """非零有限数值单元格：norm_number(v) == v。"""
    assert norm_number(v) == v


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(s=st.one_of(_decimal_text, _zero_padded_text))
def test_parsable_text_equals_float(s):
    """由阿拉伯数字、可带一个小数点及其后数字组成的文本：转为对应数值。"""
    assert norm_number(s) == float(s)
    # 前后空白（含全角空格）不影响结果
    assert norm_number(f" {s}\u3000") == float(s)


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(v=st.integers(min_value=1, max_value=10**12))
def test_positive_number_roundtrips_through_str(v):
    """正数值 v：norm_number(v) 与 norm_number(str(v)) 都等于 v（str(v) 为无符号十进制）。"""
    assert norm_number(v) == v
    assert norm_number(str(v)) == v
    f = float(v)
    assert norm_number(f) == f
    assert norm_number(f"{f:.1f}") == f
    assert norm_number(str(f)) == f  # 如 "612.0" 与 "612" 都为 612


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(z=st.sampled_from([0, 0.0, -0.0]) | _zero_text)
def test_zero_is_missing(z):
    """0（数值或文本）一律为缺失值。"""
    assert norm_number(z) is None


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(x=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_non_finite_is_missing(x):
    """NaN 与无穷大为缺失值。"""
    assert norm_number(x) is None
    assert norm_number(str(x)) is None


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=200, deadline=None)
@given(s=_unparsable_text)
def test_unparsable_text_is_missing(s):
    """不能整串解析为数字的文本：结果为缺失值。"""
    assert norm_number(s) is None


# Feature: admission-plan-matching-refactor, Property 6: 分数字段数值化
@settings(max_examples=100, deadline=None)
@given(x=st.one_of(_nonzero_cells, _decimal_text, _zero_text, _unparsable_text))
def test_result_is_none_or_nonzero_finite_float(x):
    """输出只会是缺失值或非零有限 float。"""
    r = norm_number(x)
    assert r is None or (isinstance(r, float) and math.isfinite(r) and r != 0)
