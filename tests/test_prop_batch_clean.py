# -*- coding: utf-8 -*-
"""批次清洗五步与成分抽取属性测试（任务 5.17）。"""

import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import config
import loader
import normalizer
from tests.factories import make_b_table
from tests.reference import (
    ref_b_layer,
    ref_clean_batch,
    ref_clean_batch_steps_1_4,
    ref_extract_components,
)
from tests.strategies import charset_text

# ---------------------------------------------------------------------------
# 生成器：把括号（半角/全角）、“-”、批次关键字、拉丁字母、带圈数字、中文数字、段/批词、
# 归一表词条、别名键、缺失值记号与空白拼成批次文本。
# ---------------------------------------------------------------------------
_BATCH_PIECES = (
    ["(", ")", "（", "）", "-", "－", "--", "()", "（）", "((", "))"]
    + list(config.BATCH_KEYWORDS)
    + list(config.BATCH_NAME_NORMALIZE)
    + ["本科批", "专科批", "提前批", "本科普通批", "普通本科批", "专科普通批", "高职高专普通批",
       "A", "b", "C", "Ｂ", "ａ", "AB", "10", "1", "０９", "⑩", "①", "⑳",
       "一", "二", "三", "四", "十", "一段", "二批", "三批", "段", "批",
       "物理", "蒙授", "国家专项计划", "无", "nan", "None", " ", "\u3000", "\t"]
)
batch_text = st.one_of(
    st.lists(st.sampled_from(_BATCH_PIECES), min_size=0, max_size=10).map("".join),
    st.lists(
        st.one_of(st.sampled_from(_BATCH_PIECES), st.text(max_size=3)),
        min_size=0, max_size=8,
    ).map("".join),
    charset_text,
)
batch_input = st.one_of(batch_text, st.none(), st.just(float("nan")))
province_input = st.sampled_from([None, "四川", "河北", "浙江"])

# 成分抽取直接作用于批次主体：不含空白，组合同样的片段
body_text = st.lists(
    st.sampled_from([p for p in _BATCH_PIECES if p.strip()]), min_size=0, max_size=8
).map("".join)

# ---------------------------------------------------------------------------
# 参考实现自检：Req 3.5、3.18 的示例
# ---------------------------------------------------------------------------
REQ_3_5_EXAMPLES = [
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
]

# (批次名称, 层次, 提前, 段, 批序, 规范批次名)
REQ_3_18_EXAMPLES = [
    ("本科批", "本科", "否", None, None, "本科批"),
    ("本科", "本科", "否", None, None, "本科批"),
    ("本科院校", "本科", "否", None, None, "本科批"),
    ("本科院校(物理)", "本科", "否", None, None, "本科批"),
    ("本科批地方专项计划", "本科", "否", None, None, "本科批"),
    ("专科批", "专科", "否", None, None, "专科批"),
    ("高职(专科)批次", "专科", "否", None, None, "专科批"),
    ("高职院校专项计划", "专科", "否", None, None, "专科批"),
    ("本科提前批", "本科", "是", None, None, "本科提前批"),
    ("高职(专科)提前批次", "专科", "是", None, None, "专科提前批"),
    ("本科批次B段", "本科", "否", "B", None, "本科批B段"),
    ("本科提前批C段", "本科", "是", "C", None, "本科提前批C段"),
    ("本科-提前-C", "本科", "是", "C", None, "本科提前批C段"),
    ("本科二批", "本科", "否", None, "二", "本科二批"),
    ("高职(专科)批次⑩段", "专科", "否", "10", None, "专科批10段"),
    ("提前批", None, "是", None, None, "提前批"),
    ("普通批", None, "否", None, None, "普通批"),
    ("一段", None, "否", "一", None, "普通批一段"),
    ("（专项南单对口本二）", None, None, None, None, None),
    ("特殊类型招生", None, None, None, None, None),
]


@pytest.mark.parametrize("raw,body,frags", REQ_3_5_EXAMPLES)
def test_reference_clean_matches_req_3_5_examples(raw, body, frags):
    assert ref_clean_batch(raw, None) == (body, frags)


@pytest.mark.parametrize("raw,layer,early,segment,order,canonical", REQ_3_18_EXAMPLES)
def test_reference_components_match_req_3_18_examples(raw, layer, early, segment, order, canonical):
    body, _ = ref_clean_batch(raw, None)
    comps = ref_extract_components(body)
    assert comps == (layer, early, segment, order)
    assert normalizer.compose_batch_name(comps) == canonical


# ---------------------------------------------------------------------------
# Property 7 第 1 条：clean_batch 等于五步参考实现
# ---------------------------------------------------------------------------
# Feature: admission-plan-matching-refactor, Property 7: 批次清洗与成分抽取
# **Validates: Requirements 3.5**
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(batch_input, province_input)
@example("高职(专科)批-蒙授", None)
@example("本科院校(物理)", None)
@example("本科批A段-国家专项计划", None)
@example("本科-提前-C", None)
@example("本科批(物理)-蒙授", None)
@example("本科批)", None)
@example("本科批()", None)
@example("特殊类型招生", None)
@example("蒙授-国家专项", None)
@example("（专项南单对口本二）", None)
@example("（专项南单对口本二", None)
@example("（单列类（选考外语））", None)
@example("本科(A(B)", None)
@example("本科普通批", "四川")
@example("n-a-n", None)
@example("本科(物理)(", None)
@example("((本科)", None)
def test_clean_batch_equals_reference(x, province):
    body, fragments = normalizer.clean_batch(x, province)
    ref_body, ref_fragments = ref_clean_batch(x, province)
    assert (body, fragments) == (ref_body, ref_fragments)

    # 片段来源：步骤③括号内容在前，其后只能是不含批次关键字且非单个拉丁字母的“-”段；
    # 片段与主体都不为空，主体不为缺失值记号
    _b, bracket_frags, dash_frags = ref_clean_batch_steps_1_4(x)
    assert list(fragments) == bracket_frags + dash_frags
    keywords = [normalizer.norm_text(k) for k in config.BATCH_KEYWORDS]
    for frag in dash_frags:
        assert not any(k in frag for k in keywords)
        assert not (len(frag) == 1 and frag.isascii() and frag.isalpha())
    for frag in fragments:
        assert frag != "" and frag.lower() not in {m.lower() for m in config.MISSING_TOKENS}
    if body is not None:
        assert body != "" and body.lower() not in {m.lower() for m in config.MISSING_TOKENS}
    if normalizer.norm_text(x) is None:
        assert (body, fragments) == (None, ())


# 带省份键的别名表：省份键优先于 "*"，省份为空只查 "*"
_CUSTOM_ALIAS = {
    ("四川", "本科批A"): "甲",
    ("*", "本科批A"): "乙",
    ("*", "（备注）本科批B"): "丙",       # 键先按步骤①–④清洗：括号内容被移出，主体为“本科批B”
    ("河北", "高职高专批"): "丁",          # 键先经归一替换：主体为“专科专科批”
}
_ALIAS_TEXTS = ["本科批A", "本科批B", "高职高专批", "专科专科批", "本科批A-蒙授", "(物理)本科批A"]


# Feature: admission-plan-matching-refactor, Property 7: 批次清洗与成分抽取
# **Validates: Requirements 3.5**
@settings(max_examples=150)
@given(st.one_of(st.sampled_from(_ALIAS_TEXTS), batch_text), province_input)
@example("本科批A", "四川")
@example("本科批A", "重庆")
@example("本科批A", None)
@example("本科批B", "四川")
@example("高职高专批", "河北")
def test_clean_batch_alias_matches_reference(x, province):
    alias = normalizer.build_alias_table(_CUSTOM_ALIAS)
    assert normalizer.clean_batch(x, province, alias=alias) == ref_clean_batch(
        x, province, raw_alias=_CUSTOM_ALIAS)


# ---------------------------------------------------------------------------
# Property 7 第 2 条：extract_components 等于参考成分抽取
# ---------------------------------------------------------------------------
# Feature: admission-plan-matching-refactor, Property 7: 批次清洗与成分抽取
# **Validates: Requirements 3.18**
@settings(max_examples=300)
@given(st.one_of(body_text, st.none(), st.just("")))
@example("本科批")
@example("本科提前批C段")
@example("专科批10段")
@example("一段")
@example("二十段")          # “十”前一字是中文数字，不产生取值
@example("A段B段")          # 两个不同的段取值 → 空
@example("AB段")            # “B”前一字是拉丁字母 → 不产生取值
@example("本科A")           # 无“段”：单个拉丁字母
@example("本科AB")
@example("本科批一批二批")   # 批序两个取值 → 空
@example("本科专科批")       # 层次两者都含 → 空
@example("特殊类型招生")
@example(None)
def test_extract_components_equals_reference(body):
    comps = normalizer.extract_components(body)
    assert comps == ref_extract_components(body)
    layer, early, segment, order = comps
    assert layer in ("本科", "专科", None)
    if body is None or body == "":
        assert comps == (None, None, None, None)
    if early is None:
        assert layer is None and segment is None and order is None


# 清洗链端到端：clean_batch 的主体再抽取成分
# Feature: admission-plan-matching-refactor, Property 7: 批次清洗与成分抽取
# **Validates: Requirements 3.5, 3.18**
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(batch_input, province_input)
def test_clean_then_extract_equals_reference(x, province):
    body, _ = normalizer.clean_batch(x, province)
    ref_body, _ = ref_clean_batch(x, province)
    assert normalizer.extract_components(body) == ref_extract_components(ref_body)


# ---------------------------------------------------------------------------
# Property 7 第 3 条：B 标准层次 = “层次”列 → “层级”列 → 批次层次成分 的第一个非空值
# ---------------------------------------------------------------------------
_LAYER_POOL = [None, "", "本科", "专科", "高职", "高职高专", "本科/专科", "未知", "nan", "理科",
               " 本科 ", "（本科）", "高职(专科)", "专科(高职)", "本科专科", "Ｕ本科"]
layer_text = st.one_of(st.sampled_from(_LAYER_POOL), charset_text)
triple = st.tuples(layer_text, layer_text, batch_text)


# Feature: admission-plan-matching-refactor, Property 7: 批次清洗与成分抽取
# **Validates: Requirements 3.6**
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(triple, min_size=1, max_size=5))
@example([(None, None, "本科批")])
@example([(None, "本科", None)])
@example([("高职", None, None)])
@example([("本科", None, "专科批")])
@example([(None, None, None)])
@example([("本科专科", "专科", "本科批")])
@example([(None, "未知", "高职(专科)批次⑩段")])
def test_b_standard_layer_order(triples):
    rows = [
        {"province": "四川", "layer_primary": l1, "layer_secondary": l2, "batch": b}
        for l1, l2, b in triples
    ]
    b_kept = loader.assign_b_ids(make_b_table(rows))
    b_std, _report = normalizer.normalize_b(b_kept, {}, verbose=False)
    assert len(b_std) == len(triples)

    for i, (l1, l2, b) in enumerate(triples):
        expected_layer = ref_b_layer(l1, l2, b, "四川")
        got = b_std["layer"].iloc[i]
        assert (None if pd.isna(got) else got) == expected_layer

        # 批次部分一并核对：主体、片段、批次层次成分、其余 3 个成分
        ref_body, ref_frags = ref_clean_batch(b, "四川")
        comps = ref_extract_components(ref_body)
        for col, want in (("batch_body", ref_body), ("batch_layer", comps[0]),
                          ("early", comps[1]), ("segment", comps[2]),
                          ("batch_order", comps[3])):
            v = b_std[col].iloc[i]
            assert (None if pd.isna(v) else v) == want, col
        assert tuple(b_std["batch_remark"].iloc[i]) == ref_frags
