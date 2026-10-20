# -*- coding: utf-8 -*-
"""批次清洗链示例表（design.md / Req 3.5、3.17–3.22 的示例）。"""

import pytest

import normalizer as nz
from loader import ABatchTable

CLEAN_CASES = [
    ("高职(专科)批-蒙授", "专科批", ("蒙授",)),
    ("本科院校(物理)", "本科院校", ("物理",)),
    ("本科批A段-国家专项计划", "本科批A段", ("国家专项计划",)),
    ("本科-提前-C", "本科提前C", ()),
    ("本科批(物理)-蒙授", "本科批", ("物理", "蒙授")),
    ("本科批)", "本科批", ()),
    ("本科批()", "本科批", ()),
    ("特殊类型招生", "特殊类型招生", ()),
    ("蒙授-国家专项", None, ("蒙授", "国家专项")),
    ("（专项南单对口本二）", None, ("专项南单对口本二",)),
    ("（专项南单对口本二", None, ("专项南单对口本二",)),
    ("（单列类（选考外语））", None, ("单列类(选考外语)",)),
    ("本科(A(B)", "本科", ("A(B)",)),
    ("本科普通批", "本科批", ()),
    ("高职高专普通批", "专科批", ()),
    (None, None, ()),
    ("nan", None, ()),
]


@pytest.mark.parametrize("raw,body,frags", CLEAN_CASES)
def test_clean_batch_examples(raw, body, frags):
    assert nz.clean_batch(raw, "河北") == (body, frags)


COMPONENT_CASES = [
    ("本科批", ("本科", "否", None, None), "本科批"),
    ("本科院校", ("本科", "否", None, None), "本科批"),
    ("本科批地方专项计划", ("本科", "否", None, None), "本科批"),
    ("专科批次", ("专科", "否", None, None), "专科批"),
    ("本科提前批", ("本科", "是", None, None), "本科提前批"),
    ("本科批次B段", ("本科", "否", "B", None), "本科批B段"),
    ("本科提前C", ("本科", "是", "C", None), "本科提前批C段"),
    ("本科二批", ("本科", "否", None, "二"), "本科二批"),
    ("专科批次10段", ("专科", "否", "10", None), "专科批10段"),
    ("提前批", (None, "是", None, None), "提前批"),
    ("普通批", (None, "否", None, None), "普通批"),
    ("一段", (None, "否", "一", None), "普通批一段"),
    ("特殊类型招生", (None, None, None, None), None),
    ("本科批A段B段", ("本科", "否", None, None), "本科批"),
    ("本科AB段", ("本科", "否", None, None), "本科批"),
    ("本科一批二批", ("本科", "否", None, None), "本科批"),
    ("本科C提前D", ("本科", "是", None, None), "本科提前批"),
    ("专科批十一段", ("专科", "否", None, None), "专科批"),
    ("本科专科批", (None, None, None, None), None),
]


@pytest.mark.parametrize("body,components,name", COMPONENT_CASES)
def test_extract_and_compose_examples(body, components, name):
    assert nz.extract_components(body) == components
    assert nz.compose_batch_name(components) == name


# Req 3.18 示例表逐名展开：(省份, 批次名称, 层次, 提前标记, 段, 批序, 规范批次名)
# 全链路：clean_batch → extract_components → compose_batch_name
TABLE_3_18 = [
    ("河北", "本科批", "本科", "否", None, None, "本科批"),
    ("河北", "本科", "本科", "否", None, None, "本科批"),
    ("河北", "本科院校", "本科", "否", None, None, "本科批"),
    ("河北", "本科院校(物理)", "本科", "否", None, None, "本科批"),
    ("河北", "本科院校（物理）", "本科", "否", None, None, "本科批"),
    ("河北", "本科批地方专项计划", "本科", "否", None, None, "本科批"),
    ("河北", "专科批", "专科", "否", None, None, "专科批"),
    ("河北", "高职(专科)批次", "专科", "否", None, None, "专科批"),
    ("河北", "高职院校专项计划", "专科", "否", None, None, "专科批"),
    ("河北", "本科提前批", "本科", "是", None, None, "本科提前批"),
    ("河北", "高职(专科)提前批次", "专科", "是", None, None, "专科提前批"),
    ("四川", "本科批次B段", "本科", "否", "B", None, "本科批B段"),
    ("四川", "本科提前批C段", "本科", "是", "C", None, "本科提前批C段"),
    ("四川", "本科-提前-C", "本科", "是", "C", None, "本科提前批C段"),
    ("河北", "本科二批", "本科", "否", None, "二", "本科二批"),
    ("浙江", "高职(专科)批次⑩段", "专科", "否", "10", None, "专科批10段"),
    ("浙江", "专科批次⑩段", "专科", "否", "10", None, "专科批10段"),
    ("河北", "提前批", None, "是", None, None, "提前批"),
    ("河北", "普通批", None, "否", None, None, "普通批"),
    ("浙江", "一段", None, "否", "一", None, "普通批一段"),
    ("山东", "一段", None, "否", "一", None, "普通批一段"),
    ("山东", "二段", None, "否", "二", None, "普通批二段"),
    ("河北", "（专项南单对口本二）", None, None, None, None, None),
    ("河北", "特殊类型招生", None, None, None, None, None),
    ("河北", "本科批A段B段", "本科", "否", None, None, "本科批"),
    ("河北", "本科AB段", "本科", "否", None, None, "本科批"),
    ("河北", "本科一批二批", "本科", "否", None, None, "本科批"),
    ("河北", "本科C提前D", "本科", "是", None, None, "本科提前批"),
    ("河北", "专科批十一段", "专科", "否", None, None, "专科批"),
    ("河北", "本科专科批", None, None, None, None, None),
]


@pytest.mark.parametrize("prov,raw,level,early,seg,order,name", TABLE_3_18)
def test_req_3_18_table_full_chain(prov, raw, level, early, seg, order, name):
    body, _ = nz.clean_batch(raw, prov)
    comps = nz.extract_components(body)
    assert comps == (level, early, seg, order)
    assert nz.compose_batch_name(comps) == name
    assert nz.canonical_batch_name(raw, prov) == name


def test_circled_digit_segment_via_clean_chain():
    body, _ = nz.clean_batch("高职(专科)批次⑩段", "浙江")
    assert nz.extract_components(body) == ("专科", "否", "10", None)
    assert nz.canonical_batch_name("高职(专科)批次⑩段", "浙江") == "专科批10段"


def test_early_chain_example():
    assert nz.canonical_batch_name("本科-提前-C", None) == "本科提前批C段"


def test_alias_province_precedence():
    alias = {("四川", "X"): "甲", ("*", "X"): "乙"}
    assert nz.apply_alias("X", "四川", alias) == "甲"
    assert nz.apply_alias("X", "重庆", alias) == "乙"
    assert nz.apply_alias("X", None, alias) == "乙"
    assert nz.apply_alias("Y", "四川", alias) == "Y"
    assert nz.apply_alias(None, "四川", alias) is None


def test_alias_keys_are_cleaned(monkeypatch):
    monkeypatch.setattr(nz.config, "BATCH_ALIAS", {("*", "高职（高专）普通批"): "专科批",
                                                    ("四川", "ＸＹ"): "甲"})
    table = nz.get_alias_table()
    assert table[("四川", "XY")] == "甲"
    assert nz.clean_batch("XY", "四川") == ("甲", ())
    assert nz.clean_batch("XY", "河北") == ("XY", ())


def _table():
    return ABatchTable(
        exact={("130000", "1"): "本科批", ("510000", "1038"): "本科批A段-国家专项计划"},
        by_type={"3": "本科一批", "1": "本科批"},
        ambiguous={("110100", "7"): ("本科批", "专科批")},
    )


def test_map_a_batch_name():
    t = _table()
    assert nz.map_a_batch_name("130000.0", 1, t) == "本科批"
    assert nz.map_a_batch_name(650000, 3, t) == "本科一批"
    assert nz.map_a_batch_name(510000, "1038", t) == "本科批A段-国家专项计划"
    assert nz.map_a_batch_name(110100, 10001, t) is None
    assert nz.map_a_batch_name(110100, 7, t) is None
    assert nz.map_a_batch_name(130000, None, t) is None
    assert nz.map_a_batch_name(130000, "nan", t) is None


def test_legal_sets_and_batch_check():
    import pandas as pd

    t = ABatchTable(exact={("510000", "1"): "本科批B段-国家专项计划"},
                    by_type={}, ambiguous={("510000", "9"): ("专科批", "高职(专科)提前批")})
    plist = pd.DataFrame({"province_id": ["130000", "999999"], "batch_name": ["本科普通批", "本科批"]})
    legal = nz.build_province_legal_sets(t, plist)
    assert legal["四川"] == frozenset({"本科批B段", "专科批", "专科提前批"})
    assert legal["河北"] == frozenset({"本科批"})
    assert set(legal) == {"四川", "河北"}

    std = nz.canonical_batch_name("本科批次B段", "四川")
    assert nz.batch_check("本科批次B段", std, "四川", legal) == "合法"
    assert nz.batch_check("特殊类型招生", None, "四川", legal) == "未识别"
    assert nz.batch_check(None, None, "四川", legal) == "空"
    assert nz.batch_check("本科批", "本科批", "四川", legal) == "不在本省清单"
    assert nz.batch_check("本科批", "本科批", None, legal) == "不在本省清单"
    assert nz.batch_check("本科批", "本科批", "上海", legal) == "不在本省清单"
    assert nz.build_province_legal_sets(t, None).keys() == {"四川"}


def test_merge_batch_remark():
    assert nz.merge_batch_remark("中外合作", ("物理",)) == ("中外合作", "物理")
    assert nz.merge_batch_remark(None, ()) == ()
    assert nz.merge_batch_remark("nan", ("物理",)) == ("物理",)
    assert nz.join_batch_remark(("a", "b")) == "a|b"
