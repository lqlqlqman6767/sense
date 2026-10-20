# -*- coding: utf-8 -*-
"""Property 4：省份标准化（A 编码与 B 名称）。"""

import string

from hypothesis import given, strategies as st

import config
from normalizer import norm_code, norm_province_id, norm_province_name, norm_text

_PREFIXES = sorted(config.PROVINCE_PREFIX_MAP)
_NAMES = sorted(config.PROVINCE_NAMES)
_SUFFIX_OR_EMPTY = ("",) + tuple(config.PROVINCE_SUFFIXES)
_ASCII_DIGITS = frozenset("0123456789")


# ---------- 局部策略 ----------

def _digits(n):
    return st.text(alphabet=string.digits, min_size=n, max_size=n)


# 任意字符串，并有意混入 6 位数字、带 .0、带空白、全角数字、省份名等“接近有效”的值
_prefix = st.sampled_from(_PREFIXES)
_name = st.sampled_from(_NAMES)
_suffix = st.sampled_from(config.PROVINCE_SUFFIXES)
_any_text = st.one_of(
    st.text(max_size=12),
    st.text(alphabet=string.digits + ".  \u3000", max_size=10),
    _digits(6),
    st.builds(lambda p, d: p + d + ".0", _prefix, _digits(4)),
    st.builds(lambda p, d: p + d, _prefix, _digits(4)),
    st.builds(lambda n, s: n + s, _name, st.sampled_from(_SUFFIX_OR_EMPTY)),
    st.builds(lambda n, s1, s2: n + s1 + s2, _name, _suffix, _suffix),
    st.builds(lambda a, b: a + b, _name, st.text(max_size=4)),
    st.builds(lambda s: " " + s + " ", _name),
    st.builds(lambda n, s: n + s, _name, st.sampled_from(config.PROVINCE_SUFFIXES))
    .map(lambda s: "\uff08" + s + "\uff09"),
)

_wrong_length_digits = st.one_of(
    _digits(2), _digits(4), _digits(7),
    _digits(1), _digits(3), _digits(5), _digits(8),
)
# 含 ASCII 字母的值：NFKC 不会把字母变成数字，必然不是 6 位纯数字
_with_non_digit = st.builds(
    lambda p, d, ch, pos: (lambda s: s[:pos] + ch + s[pos:])(p + d),
    _prefix,
    _digits(4),
    st.sampled_from(string.ascii_letters),
    st.integers(min_value=0, max_value=6),
)


# ---------- 参考实现（只依赖 config 与已有的 norm_text/norm_code） ----------

def _ref_province_id(s):
    c = norm_code(s)
    if c is not None and len(c) == 6 and all(ch in _ASCII_DIGITS for ch in c):
        return config.PROVINCE_PREFIX_MAP.get(c[:2])
    return None


def _ref_province_name(s):
    t = norm_text(s)
    if t is None:
        return None
    for suffix in config.PROVINCE_SUFFIXES:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t if t in config.PROVINCE_NAMES else None


# Feature: admission-plan-matching-refactor, Property 4: 省份标准化（A 编码与 B 名称）
# **Validates: Requirements 3.3**
@given(
    s_id=_any_text,
    s_name=_any_text,
    prefix=_prefix,
    tail=_digits(4),
    bad_len=_wrong_length_digits,
    bad_char=_with_non_digit,
    name=_name,
    suffix=st.sampled_from(_SUFFIX_OR_EMPTY),
    suffix1=_suffix,
    suffix2=_suffix,
)
def test_property_4_province_normalization(
    s_id, s_name, prefix, tail, bad_len, bad_char, name, suffix, suffix1, suffix2
):
    # --- norm_province_id：任意字符串与参考实现一致 ---
    assert norm_province_id(s_id) == _ref_province_id(s_id)

    # 31 个国标前缀 + 任意 4 位数字，带不带 ".0" 都映射到对应简称
    expected = config.PROVINCE_PREFIX_MAP[prefix]
    assert norm_province_id(prefix + tail) == expected
    assert norm_province_id(prefix + tail + ".0") == expected

    # 2 位、4 位、7 位数字及含非数字字符的值映射为 None
    assert norm_province_id(bad_len) is None
    assert norm_province_id(bad_char) is None

    # --- norm_province_name：任意字符串与参考实现一致 ---
    assert norm_province_name(s_name) == _ref_province_name(s_name)

    # 简称 + 任一后缀（或无后缀）还原为简称
    assert norm_province_name(name + suffix) == name

    # 简称后接两个非空后缀：只删一次，结果不是简称，返回 None
    assert norm_province_name(name + suffix1 + suffix2) is None
