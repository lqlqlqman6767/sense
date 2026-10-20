# -*- coding: utf-8 -*-
"""Normalizer 测试。"""

import re
import unicodedata

from hypothesis import given

import config
import normalizer as nz
from normalizer import norm_text
from tests.strategies import messy_text

_MISSING = {t.lower() for t in config.MISSING_TOKENS}


# Feature: admission-plan-matching-refactor, Property 3: 文本清洗符合定义
# **Validates: Requirements 3.1**
@given(messy_text)
def test_norm_text_matches_definition(s):
    r = norm_text(s)
    cleaned = re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    if cleaned.lower() in _MISSING:
        assert r is None
    else:
        assert r == cleaned
    if r is not None:
        assert not any(c.isspace() for c in r)
        assert "（" not in r and "）" not in r
        assert r.lower() not in _MISSING


# ---------------------------------------------------------------
# 任务 5.1：其余标量函数的示例检查（来自 design.md / requirements 3.3–3.9）
# ---------------------------------------------------------------
import pytest

from normalizer import (
    layer_from_column,
    norm_major,
    norm_major_group,
    norm_number,
    norm_province_id,
    norm_province_name,
    norm_subject_a,
)


@pytest.mark.parametrize("x,expected", [
    ("110100", "北京"), (500100.0, "重庆"), ("650000", "新疆"), (" 110100.0 ", "北京"),
    ("11", None), ("1101", None), ("1101000", None), ("990000", None), (None, None), ("", None),
])
def test_norm_province_id_examples(x, expected):
    assert norm_province_id(x) == expected


@pytest.mark.parametrize("x,expected", [
    ("天津市", "天津"), ("广西壮族自治区", "广西"), ("新疆维吾尔自治区", "新疆"),
    ("宁夏回族自治区", "宁夏"), ("内蒙古自治区", "内蒙古"), ("黑龙江省", "黑龙江"),
    ("北京", "北京"), ("香港特别行政区", None), ("北京市市", None), ("市", None), (None, None),
])
def test_norm_province_name_examples(x, expected):
    assert norm_province_name(x) == expected


@pytest.mark.parametrize("x,expected", [
    ("物理", "物理类"), ("理科", "物理类"), ("文科", "历史类"), ("历史类", "历史类"),
    ("综合", "综合"), ("不限", "综合"), (" 物 理 ", "物理类"), ("物化", None), (None, None), ("", None),
])
def test_norm_subject_a_examples(x, expected):
    assert norm_subject_a(x) == expected


def test_norm_major_only_normalizes_text():
    assert norm_major("法学（中外合作）") == norm_major("法学 (中外合作)") == "法学(中外合作)"
    assert norm_major("法学(中外合作)") != norm_major("法学")
    assert norm_major("无") is None


@pytest.mark.parametrize("x,expected", [
    ("第01组", "1"), ("(01)", "1"), ("（01）", "1"), ("01专业组", "1"), ("专业组(001)", "1"),
    ("((01))", "1"), ("第0组", "0"), ("第00组", "0"), ("A01", "A1"), ("123Q1", "123Q1"),
    ("001组", "1"), ("(1)(2)", "(1)(2)"), ("组", None), ("()", None), (None, None),
])
def test_norm_major_group_examples(x, expected):
    assert norm_major_group(x) == expected


@pytest.mark.parametrize("x,expected", [
    ("612", 612.0), ("612.0", 612.0), ("612.5", 612.5), (" 612 ", 612.0), (612, 612.0), (612.5, 612.5),
    ("+5", None), ("-1", None), ("1,234", None), ("暂无", None), ("—", None), ("1.", None),
    (".5", None), ("0", None), ("0.0", None), (0, None), (0.0, None), (float("nan"), None),
    (None, None), ("", None),
])
def test_norm_number_examples(x, expected):
    assert norm_number(x) == expected


@pytest.mark.parametrize("x,expected", [
    ("本科", "本科"), ("专科", "专科"), ("高职", "专科"), ("高职(专科)", "专科"), ("高职（专科）", "专科"),
    ("高职高专", "专科"), ("本科专科", None), ("研究生", None), (None, None), ("", None),
])
def test_layer_from_column_examples(x, expected):
    assert layer_from_column(x) == expected


# ---------------------------------------------------------------
# 任务 5.18：Normalizer 单元测试（Req 3.7、3.12、3.14、3.15、3.17、3.21）
# 表级用例只用合成数据（factories），不读取 ABC/data。
# ---------------------------------------------------------------
import numpy as np
import pandas as pd

import loader
import normalizer as nz
from normalizer import norm_code, norm_text, vec_map
from tests.factories import make_a_batch_table, make_a_table, make_b_table

_CHECK_ORDER = ["空", "未识别", "合法", "不在本省清单"]


@pytest.fixture(scope="module")
def batch_tbl():
    """含一个 ambiguous 组合 (130000, 9)：同时对应“本科批”“专科批”，且 9 在全表也不唯一。"""
    return make_a_batch_table([
        ("130000", "1", "本科批"),
        ("130000", "2", "专科批"),
        ("130000", "3", "特殊类型招生"),        # 名称存在，但规范批次名为空 -> 未识别
        ("130000", "5", "本科批B段"),
        ("130000", "9", "本科批"),
        ("130000", "9", "专科批"),
    ])


@pytest.fixture(scope="module")
def legal_sets(batch_tbl):
    return nz.build_province_legal_sets(batch_tbl, None)


def _scope(rows):
    a = make_a_table(rows)
    a.index = pd.Index(np.arange(len(a)) + 7)         # 模拟 split_scope 之后的位置索引
    return a


def _b(rows):
    return loader.assign_b_ids(make_b_table(rows))


# ---- Req 3.12：vec_map 不把 1 / "1" / 1.0 / True 错并 ----

def test_vec_map_keeps_one_string_float_bool_apart():
    col = pd.Series([1, "1", 1.0, True], dtype=object)
    # norm_text：1 与 "1" 的 str 相同，结果相同是正确的；1.0、True 不能被并入 "1"
    assert vec_map(col, norm_text).tolist() == ["1", "1", "1.0", "True"]
    assert vec_map(col, norm_text).tolist() == [norm_text(x) for x in col]
    # norm_code 会把 "1.0" 去成 "1"，但 True 必须仍是 "True"
    assert vec_map(col, norm_code).tolist() == [norm_code(x) for x in col] == ["1", "1", "1", "True"]


# ---- Req 3.7：专业组删除顺序（先删全部“第”，再删“专业组”） ----

@pytest.mark.parametrize("x,expected", [
    ("专第业组1", "1"),          # 先删“第”得“专业组1”，再删“专业组”得“1”；颠倒顺序会剩“专业1”
    ("专第业第组01", "1"),
    ("第专业组", None),          # 删完为空 -> 缺失
    ("专业第组(02)", "2"),
])
def test_norm_major_group_deletion_order(x, expected):
    assert norm_major_group(x) == expected


# ---- Req 3.14：normalize_a / normalize_b 不改原列、行数、行顺序 ----

def test_normalize_a_keeps_columns_row_count_and_order(batch_tbl, legal_sets):
    names = ["丙", " 甲（x） ", "乙", "甲(x)", None, "丙"]            # 乱序、重复、含空白与缺失
    rows = [{"province": p, "batch": b, "major_name": n, "school_id": s, "subject": sub}
            for p, b, n, s, sub in zip(
                ["130000", 130000.0, "abc", "130000", "130000", "130000"],
                ["1", 2.0, "9", None, "10001", True],
                names,
                ["1001", 1001.0, " 1002 ", None, "1001.0", "ＡＢ１"],
                ["物理", None, float("nan"), "理科", "物化", "综合"])]
    a = _scope(rows)
    a["extra"] = ["x", "y", "z", "w", "v", "u"]                      # 额外原始列也必须原样
    before = a.copy(deep=True)
    cols, index = list(a.columns), a.index.copy()

    std, rep = nz.normalize_a(a, batch_tbl, legal_sets, verbose=False)

    pd.testing.assert_frame_equal(a, before)
    assert list(a.columns) == cols and a.index.equals(index)
    assert len(std) == len(a) == rep.rows
    # 标准列与输入行一一、按序对应
    assert std["major_name"].tolist() == [nz.norm_major(n) for n in names]
    assert std["a_row"].tolist() == [r + 2 for r in range(7, 13)]
    # 再标准化一次结果完全相同（说明没有原地改动输入）
    std2, _ = nz.normalize_a(a, batch_tbl, legal_sets, verbose=False)
    pd.testing.assert_frame_equal(std, std2)


def test_normalize_b_keeps_columns_row_count_and_order(legal_sets):
    names = ["丙", " 甲（x） ", "乙", "甲(x)", None, "丙"]
    rows = [{"major_name": n, "province": p, "batch": b, "subject": s, "最低分": m}
            for n, p, b, s, m in zip(
                names,
                ["河北省", "浙江", "天津市", "香港特别行政区", None, "新疆维吾尔自治区"],
                ["本科批", None, "特殊类型招生", "", 1.0, "本科批(物理)-蒙授"],
                ["物理", "物理", "综合", "文科", "综合", "不限"],
                ["612", 0, "暂无", 600.5, None, 1])]
    b = _b(rows)
    before = b.copy(deep=True)
    cols, index = list(b.columns), b.index.copy()

    std, rep = nz.normalize_b(b, legal_sets, verbose=False)

    pd.testing.assert_frame_equal(b, before)
    assert list(b.columns) == cols and b.index.equals(index)
    assert len(std) == len(b) == rep.rows
    assert std["b_id"].tolist() == b["b_id"].tolist()
    assert std["major_name"].tolist() == [nz.norm_major(n) for n in names]
    std2, _ = nz.normalize_b(b, legal_sets, verbose=False)
    pd.testing.assert_frame_equal(std, std2)


# ---- Req 3.15 / 3.21：A 的批次核查四类计数与 batch_type 查不到 ----

def test_a_batch_table_has_ambiguous_combination(batch_tbl):
    """夹具前提：(130000, 9) 是 ambiguous，且 9 不在 by_type。"""
    assert ("130000", "9") in batch_tbl.ambiguous
    assert "9" not in batch_tbl.by_type


def test_normalize_a_ambiguous_combination_counted_as_unmapped(batch_tbl, legal_sets):
    std, rep = nz.normalize_a(_scope([
        {"province": "130000", "batch": "9"},        # ambiguous -> 查不到
        {"province": "330000", "batch": "9"},        # 其它省：exact 无、by_type 无 -> 查不到
        {"province": "130000", "batch": "1"},        # 正常
        {"province": "130000", "batch": None},       # batch_type 缺失：不算“查不到”
    ]), batch_tbl, legal_sets, verbose=False)
    assert rep.a_batch_unmapped == 2
    assert rep.a_batch_unmapped_examples == [("130000", "9"), ("330000", "9")]
    assert std["batch_name"].tolist() == [None, None, "本科批", None]
    assert rep.batch_check_counts == {"空": 3, "未识别": 0, "合法": 1, "不在本省清单": 0}


def test_normalize_a_four_way_counts_cover_every_category(batch_tbl, legal_sets):
    rows = [
        {"province": "130000", "batch": "1"},        # 合法
        {"province": "130000", "batch": "2"},        # 合法
        {"province": "130000", "batch": "3"},        # 未识别（“特殊类型招生”）
        {"province": "130000", "batch": "10001"},    # 空（查不到）
        {"province": "130000", "batch": None},       # 空
        {"province": "abc", "batch": "5"},           # by_type 唯一 -> 本科批B段；省份无法转换 -> 不在本省清单
    ]
    std, rep = nz.normalize_a(_scope(rows), batch_tbl, legal_sets, verbose=False)
    assert list(rep.batch_check_counts) == _CHECK_ORDER
    assert rep.batch_check_counts == {"空": 2, "未识别": 1, "合法": 2, "不在本省清单": 1}
    assert sum(rep.batch_check_counts.values()) == rep.rows == len(rows)
    assert std["batch_check"].tolist() == ["合法", "合法", "未识别", "空", "空", "不在本省清单"]
    assert rep.unrecognized_examples == [("河北", "特殊类型招生")]
    assert rep.not_in_list_examples == [(None, "本科批B段")]
    assert rep.a_batch_unmapped == 1 and rep.a_batch_unmapped_examples == [("130000", "10001")]


# ---- Req 3.15：B 的选科置空（表外值 / 按省规则）及示例 ----

def test_normalize_b_subject_dropped_by_table_and_by_province_rule(legal_sets):
    std, rep = nz.normalize_b(_b([
        {"province": "浙江", "subject": "物理"},         # 按省规则置空
        {"province": "河北", "subject": "综合"},         # 按省规则置空
        {"province": "河北", "subject": "物化"},         # 表外值置空
        {"province": "山东", "subject": "综合"},         # 保留
        {"province": "河北", "subject": "理科"},         # 保留（物理类）
        {"province": "河北", "subject": None},           # 原值缺失，不计
        {"province": "河北", "subject": ""},             # 不计
        {"province": "河北", "subject": "无"},           # 缺失值记号，不计
    ]), legal_sets, verbose=False)
    assert rep.subject_dropped == 3
    assert rep.subject_dropped_examples == [("浙江", "物理"), ("河北", "综合"), ("河北", "物化")]
    assert std["subject"].tolist() == [None, None, None, "综合", "物理类", None, None, None]


# ---- Req 3.15：四类计数之和、示例至多 20 个不同值（按行顺序） ----

def test_normalize_b_examples_capped_at_20_distinct_in_row_order(legal_sets):
    n = 25
    rows = []
    rows += [{"province": f"外{i}", "batch": None} for i in range(n)] + [{"province": "外0", "batch": None}]
    rows += [{"province": "河北", "batch": f"特殊类型招生{i}"} for i in range(n)] \
        + [{"province": "河北", "batch": "特殊类型招生0"}]
    rows += [{"province": "河北", "batch": f"本科批{i + 1}段"} for i in range(n)] \
        + [{"province": "河北", "batch": "本科批1段"}]
    rows += [{"province": "河北", "batch": "本科批", "subject": f"化生{i}"} for i in range(n)] \
        + [{"province": "河北", "batch": "本科批", "subject": "化生0"}]
    _std, rep = nz.normalize_b(_b(rows), legal_sets, verbose=False)

    assert rep.rows == len(rows) == 4 * (n + 1)
    assert sum(rep.batch_check_counts.values()) == rep.rows
    assert rep.batch_check_counts == {"空": n + 1, "未识别": n + 1, "合法": n + 1, "不在本省清单": n + 1}

    assert rep.province_failed == n + 1
    assert rep.province_failed_examples == [f"外{i}" for i in range(20)]
    assert rep.unrecognized_examples == [("河北", f"特殊类型招生{i}") for i in range(20)]
    assert rep.not_in_list_examples == [("河北", f"本科批{i + 1}段") for i in range(20)]
    assert rep.subject_dropped == n + 1
    assert rep.subject_dropped_examples == [("河北", f"化生{i}") for i in range(20)]


def test_normalize_a_not_in_list_and_unrecognized_examples_capped_at_20(batch_tbl, legal_sets):
    """A 侧示例同样至多 20 个不同值：用 by_type 不唯一以外的方式批量造不同名称。"""
    n = 25
    big = make_a_batch_table(
        [("130000", str(100 + i), f"特殊类型招生{i}") for i in range(n)]
        + [("330000", str(200 + i), f"本科批{i + 1}段") for i in range(n)]
        + [("130000", "1", "本科批")])
    legal = nz.build_province_legal_sets(big, None)
    # 把浙江的合法集合改成只含“本科批”，让“本科批{i}段”落到“不在本省清单”
    legal = {**legal, "浙江": frozenset({"本科批"})}
    rows = [{"province": "130000", "batch": str(100 + i)} for i in range(n)] \
        + [{"province": "330000", "batch": str(200 + i)} for i in range(n)]
    _std, rep = nz.normalize_a(_scope(rows), big, legal, verbose=False)
    assert sum(rep.batch_check_counts.values()) == rep.rows == 2 * n
    assert rep.batch_check_counts["未识别"] == n and rep.batch_check_counts["不在本省清单"] == n
    assert rep.unrecognized_examples == [("河北", f"特殊类型招生{i}") for i in range(20)]
    assert rep.not_in_list_examples == [("浙江", f"本科批{i + 1}段") for i in range(20)]


# ---- Req 3.15：计数为 0 也要打印 ----

def test_default_report_prints_zero_counts_for_both_tables():
    for table in ("A", "B"):
        out = "\n".join(nz.NormalizeReport(table=table).format_lines())
        assert f"表={table} | 参与标准化行数=0" in out
        assert "批次核查: 空=0 未识别=0 合法=0 不在本省清单=0 | 合计=0" in out
        assert "行数=0" in out
    a_out = "\n".join(nz.NormalizeReport(table="A").format_lines())
    b_out = "\n".join(nz.NormalizeReport(table="B").format_lines())
    assert "batch_type 查不到批次名称行数=0" in a_out and "选科置空" not in a_out
    assert "选科置空（原值非缺失、标准选科为缺失）行数=0" in b_out and "batch_type" not in b_out


def test_tables_with_all_legal_rows_print_zero_for_other_categories_and_no_examples(
        batch_tbl, legal_sets, capsys):
    nz.normalize_a(_scope([{"province": "130000", "batch": "1"}]), batch_tbl, legal_sets)
    a_out = capsys.readouterr().out
    nz.normalize_b(_b([{"province": "河北", "batch": "本科批", "subject": "物理"}]), legal_sets)
    b_out = capsys.readouterr().out
    for out in (a_out, b_out):
        assert "空=0 未识别=0 合法=1 不在本省清单=0 | 合计=1" in out
        assert "省份无法转换（原始非缺失、标准省份为空）行数=0" in out
        assert out.count("（无）") == 4            # 各项示例都为空，仍然打印


def test_split_major_name_strips_parentheses_into_fragments():
    base, fragments = nz.split_major_name(
        "物理学（拔尖人才创新班）（只招已参加该校三位一体综合评价的考生）")
    assert base == "物理学"
    assert fragments == ("拔尖人才创新班", "只招已参加该校三位一体综合评价的考生")

    assert nz.split_major_name("数学与应用数学（师范类，地方转型计划）") == (
        "数学与应用数学", ("师范类，地方转型计划",))
    assert nz.split_major_name("法学") == ("法学", ())
    assert nz.split_major_name(None) == (None, ())


def test_major_name_parts_extracts_remark_tags_from_parentheses():
    base, fragments, mask = nz.major_name_parts("法学（中外合作办学）")
    assert base == "法学" and fragments == ("中外合作办学",)
    assert "中外合作" in nz.mask_to_tags(mask)

    _base, _frags, empty_mask = nz.major_name_parts("法学")
    assert empty_mask == 0
