# -*- coding: utf-8 -*-
"""norm_subject_b 示例测试（Req 3.4）。"""

import pytest

from normalizer import norm_subject_a, norm_subject_b


@pytest.mark.parametrize("province, raw, expected", [
    ("江苏", "物理", "物理类"),
    ("新疆", "文科", "历史类"),
    ("新疆", "综合", None),
    ("山东", "不限", "综合"),
    ("浙江", "物理", None),
    ("河北", "综合", None),
    ("江苏", "物化", None),
    ("浙江", "物化", None),
    (None, "物理", "物理类"),
    (None, "综合", None),
    ("山东", None, None),
    ("山东", "历史类", None),
])
def test_norm_subject_b_examples(province, raw, expected):
    assert norm_subject_b(raw, province) == expected


def test_norm_subject_a_has_no_province_rule():
    assert norm_subject_a("综合") == "综合"
    assert norm_subject_a("物理") == "物理类"
