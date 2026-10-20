# -*- coding: utf-8 -*-
"""专业组标准化属性测试（任务 5.3）。"""

from hypothesis import given, strategies as st

from normalizer import norm_major_group

_FULLWIDTH_DIGITS = "０１２３４５６７８９"


def _to_fullwidth(digits):
    return "".join(_FULLWIDTH_DIGITS[int(c)] for c in digits)


@st.composite
def wrapped_major_group(draw):
    """生成 (n, 包装后的文本)：第前缀、组/专业组后缀、外围括号、前导零、全角数字、空白。"""
    n = draw(st.integers(min_value=0, max_value=10**6))
    zeros = "0" * draw(st.integers(min_value=0, max_value=5))
    digits = zeros + str(n)
    if draw(st.booleans()):
        digits = _to_fullwidth(digits)

    prefix = draw(st.sampled_from(["", "第"]))
    suffix = draw(st.sampled_from(["", "组", "专业组"]))
    core = prefix + digits + suffix

    # 括号可包在整体外面，也可只包住数字（夹在“第”“组”之间）
    if draw(st.booleans()):
        inner_depth = draw(st.integers(min_value=0, max_value=2))
        digits_wrapped = draw(st.sampled_from([("(", ")"), ("（", "）")]))
        core = (
            prefix
            + digits_wrapped[0] * inner_depth
            + digits
            + digits_wrapped[1] * inner_depth
            + suffix
        )
    outer_depth = draw(st.integers(min_value=0, max_value=3))
    left, right = draw(st.sampled_from([("(", ")"), ("（", "）")]))
    text = left * outer_depth + core + right * outer_depth

    pad = draw(st.sampled_from(["", " ", "\u3000", "\t"]))
    return n, pad + text + pad


# Feature: admission-plan-matching-refactor, Property 5: 专业组标准化
# **Validates: Requirements 3.7**
@given(wrapped_major_group())
def test_norm_major_group_equals_str_n(case):
    n, text = case
    assert norm_major_group(text) == str(n)
