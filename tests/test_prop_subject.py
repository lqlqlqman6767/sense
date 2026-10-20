# -*- coding: utf-8 -*-
"""选科映射与按省一致性属性测试。"""

from hypothesis import example, given

import config
from normalizer import norm_subject_a, norm_subject_b, norm_text
from tests.strategies import province_subject

# 参考映射：键同样先经 norm_text（与 Req 3.4 一致，独立于被测函数的缓存表）
_REF_MAP = {norm_text(k): v for k, v in config.SUBJECT_MAP.items()}


def _ref_subject_b(x, p):
    """按 Req 3.4 文字直接实现的参考：先查表，再按省一致性规则。"""
    t = norm_text(x)
    s = _REF_MAP.get(t) if t is not None else None
    if s is None:
        return None
    if p in config.COMPREHENSIVE_PROVINCES:
        return s if s == "综合" else None
    return None if s == "综合" else s


# Feature: admission-plan-matching-refactor, Property 31: 选科映射与按省一致性
# Validates: Requirements 3.4, 3.13
# Req 15.37 表（9 行）
@example(("江苏", "物理"))
@example(("河北", "历史"))
@example(("河北", "综合"))
@example(("河北", "物化"))
@example(("河北", None))  # B“选科”缺失，“选科组合”列不参与匹配
@example(("新疆", "理科"))
@example(("新疆", "文科"))
@example(("新疆", "文科"))  # 表中第 8 行（A 为物理）与第 7 行 B 取值相同
@example(("新疆", "综合"))
# Req 15.38 表（4 行）
@example(("山东", "综合"))
@example(("山东", "不限"))
@example(("浙江", "物理"))
@example(("山东", None))
@given(province_subject)
def test_prop_subject_mapping_and_province_consistency(ps):
    p, x = ps

    a = norm_subject_a(x)
    b = norm_subject_b(x, p)

    # A 只查表，与省份无关
    t = norm_text(x)
    assert a == (_REF_MAP.get(t) if t is not None else None)
    assert a in (None, "物理类", "历史类", "综合")
    assert norm_subject_a(x) == a

    # B：与参考实现一致
    assert b == _ref_subject_b(x, p)

    # B 的结构性质
    if p in config.COMPREHENSIVE_PROVINCES:
        assert b in (None, "综合")
        assert b == ("综合" if a == "综合" else None)
    else:
        assert b in (None, "物理类", "历史类")
        assert b == (None if a == "综合" else a)

    # 映射目标值再映射后等于自身（Req 3.13）
    if a is not None:
        assert norm_subject_a(a) == a

    # 同一省份下映射两次结果不变
    assert norm_subject_b(b, p) == b


# Req 15.37、15.38 的 13 行：(省份, B“选科”, B 标准选科)；明确断言期望值，不依赖参考实现
_TABLE_ROWS = [
    ("江苏", "物理", "物理类"),
    ("河北", "历史", "历史类"),
    ("河北", "综合", None),
    ("河北", "物化", None),
    ("河北", None, None),
    ("新疆", "理科", "物理类"),
    ("新疆", "文科", "历史类"),
    ("新疆", "文科", "历史类"),
    ("新疆", "综合", None),
    ("山东", "综合", "综合"),
    ("山东", "不限", "综合"),
    ("浙江", "物理", None),
    ("山东", None, None),
]


def test_prop_subject_requirement_table_rows():
    assert len(_TABLE_ROWS) == 13
    for p, x, expected in _TABLE_ROWS:
        assert norm_subject_b(x, p) == expected, (p, x)
