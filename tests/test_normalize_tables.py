# -*- coding: utf-8 -*-
"""表级接口测试（任务 5.16）：scope_keys / split_scope / normalize_b / normalize_a / NormalizeReport。

全部使用代码构造的合成 DataFrame，不读取 ABC/data。
"""
import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

import config
import loader
import normalizer as nz
from factories import make_a_batch_table, make_a_table, make_b_table

# ------------------------------------------------------------
# 夹具
# ------------------------------------------------------------

A_COL = config.A_COLUMNS
B_COL = config.B_COLUMNS


@pytest.fixture(scope="module")
def a_batch():
    return make_a_batch_table([
        ("130000", "1", "本科批"),
        ("130000", "2", "专科批"),
        ("330000", "1", "一段"),
        ("510000", "1", "本科批B段-蒙授"),
        ("650000", "3", "本科一批"),
    ])


@pytest.fixture(scope="module")
def legal(a_batch):
    return nz.build_province_legal_sets(a_batch, None)


def _b_kept(rows):
    return loader.assign_b_ids(make_b_table(rows))


B_MIXED_ROWS = [
    {"province": "河北省", "batch": "本科批(物理)-蒙授", "subject": "物理", "layer_secondary": "本科",
     "remark_major": "中外合作办学", "remark_batch": "联合培养", "remark_plan": "非定向",
     "major_group": "第01组", "major_code": "030101.0", "最低分": "612", "major_name": "法学（中外合作）"},
    {"province": "浙江", "batch": "一段", "subject": "物理", "最低分": 0, "最高分": "暂无"},
    {"province": "香港特别行政区", "batch": "本科批", "subject": "物理"},
    {"province": "新疆维吾尔自治区", "batch": "特殊类型招生", "subject": "文科", "remark_plan": "定向"},
    {"province": "四川", "batch": None, "layer_primary": "高职", "layer_secondary": "本科"},
    {"province": "河北", "batch": "专科批", "subject": "综合", "layer_primary": "本科"},
    {"province": "四川省", "batch": "本科批次B段", "subject": "不限", "remark_major": "国家专项计划"},
    {"province": "天津市", "batch": "", "subject": "", "平均分": 612.5, "录取人数": 3.0},
]


# ------------------------------------------------------------
# scope_keys / split_scope
# ------------------------------------------------------------

def _scope_fixture():
    b = _b_kept([
        {"school_id": "1001", "province": "河北"},
        {"school_id": "1001", "province": "河北省"},       # 重复键
        {"school_id": "2002", "province": "浙江"},
        {"school_id": None, "province": "河北"},             # 缺失院校 -> 不进入键集合
        {"school_id": "4004", "province": "香港特别行政区"},   # 省份无法转换 -> 不进入键集合
    ])
    b_std, _ = nz.normalize_b(b, {}, verbose=False)
    a = make_a_table([
        {"school_id": "1001", "province": "130000", "major_name": "甲"},      # 0 范围内
        {"school_id": "1001", "province": "330000", "major_name": "乙"},      # 1 省份不同
        {"school_id": "2002", "province": "330000", "major_name": "丙"},      # 2 范围内
        {"school_id": "3003", "province": "130000", "major_name": "丁"},      # 3 院校不同
        {"school_id": None, "province": "130000", "major_name": "戊"},        # 4 院校缺失
        {"school_id": "1001", "province": "abc", "major_name": "己"},         # 5 省份无法转换
        {"school_id": "1001", "province": 130000.0, "major_name": "庚"},      # 6 float 省份，范围内
        {"school_id": "2002", "province": "999999", "major_name": "辛"},      # 7 前两位查不到
        {"school_id": "1001", "province": "", "major_name": "壬"},            # 8 省份缺失（不算失败）
        {"school_id": " 1001.0 ", "province": "130000", "major_name": "癸"},  # 9 院校 .0 与空白，范围内
    ])
    return a, b_std


def test_scope_keys_only_complete_unique_pairs():
    _a, b_std = _scope_fixture()
    keys = nz.scope_keys(b_std)
    assert isinstance(keys, pd.MultiIndex)
    assert sorted(keys.tolist()) == [("1001", "河北"), ("2002", "浙江")]


def test_split_scope_keeps_only_in_scope_rows_in_order():
    a, b_std = _scope_fixture()
    before = a.copy(deep=True)
    scope, rep = nz.split_scope(a, nz.scope_keys(b_std), verbose=False)

    assert list(scope["special_name"]) == ["甲", "丙", "庚", "癸"]
    assert list(scope.index) == [0, 2, 6, 9]                    # 索引 = 全量 A 的位置
    assert list(scope.columns) == list(a.columns)               # 原列、原顺序
    pd.testing.assert_frame_equal(a, before)                    # 全量 A 不被改动
    # 原值保持不变（float 省份、带空白的院校ID 都不被规范化）
    assert scope.loc[6, "province_id"] == 130000.0
    assert scope.loc[9, "school_id"] == " 1001.0 "

    assert (rep.total, rep.kept, rep.excluded) == (10, 4, 6)
    assert rep.kept + rep.excluded == rep.total


def test_split_scope_counts_province_failures_on_full_a():
    a, b_std = _scope_fixture()
    scope, rep = nz.split_scope(a, nz.scope_keys(b_std), verbose=False)
    # 行 5（abc）与行 7（999999）无法转换；二者都在范围外，仍计入；行 8 为缺失，不计入
    assert rep.province_failed == 2
    assert rep.province_failed_examples == ["abc", "999999"]
    assert "abc" not in set(scope["province_id"].astype(str))


def test_split_scope_examples_capped_at_20_distinct_in_row_order():
    b_std, _ = nz.normalize_b(_b_kept([{"school_id": "1001", "province": "河北"}]), {}, verbose=False)
    bad = [f"x{i}" for i in range(30)]
    rows = [{"school_id": "1001", "province": v} for v in bad + bad[:5]]   # 含重复
    _scope, rep = nz.split_scope(make_a_table(rows), nz.scope_keys(b_std), verbose=False)
    assert rep.province_failed == 35
    assert rep.province_failed_examples == bad[:20]


def test_split_scope_empty_inputs():
    b_std, _ = nz.normalize_b(_b_kept([{"school_id": "1001", "province": "河北"}]), {}, verbose=False)
    scope, rep = nz.split_scope(make_a_table([]).iloc[0:0], nz.scope_keys(b_std), verbose=False)
    assert len(scope) == 0 and (rep.total, rep.kept, rep.excluded) == (0, 0, 0)
    # B 没有任何有效键时 A_scope 为空
    empty_keys = nz.scope_keys(nz.normalize_b(_b_kept([{"school_id": None}]), {}, verbose=False)[0])
    scope2, rep2 = nz.split_scope(make_a_table([{}]), empty_keys, verbose=False)
    assert len(scope2) == 0 and rep2.excluded == 1


def test_split_scope_prints_unless_silenced(capsys):
    a, b_std = _scope_fixture()
    nz.split_scope(a, nz.scope_keys(b_std), verbose=False)
    assert capsys.readouterr().out == ""
    nz.split_scope(a, nz.scope_keys(b_std))
    out = capsys.readouterr().out
    assert "范围外排除=6" in out and "省份无法转换行数=2" in out


# ------------------------------------------------------------
# normalize_a
# ------------------------------------------------------------

A_MIXED_ROWS = [
    {"province": "130000", "batch": "1", "subject": "物理", "major_group": "第01组",
     "remark_major": "中外合作办学", "remark_plan": "非定向", "remark_direction": "蒙授",
     "remark_group": "联合培养", "major_code": "030101.0"},
    {"province": "130000", "batch": "2", "subject": "综合"},
    {"province": "330000", "batch": "1", "subject": "历史"},
    {"province": "510000", "batch": "1", "remark_plan": "定向"},        # 批次备注片段 蒙授 -> 民族类
    {"province": "650000.0", "batch": "3.0"},
    {"province": "130000", "batch": "10001"},                          # 查不到
    {"province": "130000", "batch": None},                             # batch_type 缺失，不算查不到
    {"province": "abc", "batch": "1"},                                 # 省份无法转换
    {"province": "130000", "batch": "1", "major_name": "法学（中外合作）"},
]


def _a_scope(rows=None):
    a = make_a_table(A_MIXED_ROWS if rows is None else rows)
    a.index = pd.Index(np.arange(len(a)) + 100)                      # 模拟 split_scope 后的位置索引
    return a


def test_normalize_a_does_not_touch_a_scope_raw(a_batch, legal):
    a = _a_scope()
    before = a.copy(deep=True)
    std, rep = nz.normalize_a(a, a_batch, legal, verbose=False)
    pd.testing.assert_frame_equal(a, before)                         # 行数、顺序、列、值、索引均不变
    assert len(std) == len(a) == rep.rows
    assert list(std.columns) == list(nz.A_STD_COLUMNS)
    assert list(std["a_pos"]) == list(range(len(a)))
    assert list(std["a_row"]) == [102 + i for i in range(len(a))]   # 索引 + 2
    assert not set(std.columns) & {"special_name", "province_id"}   # 标准列独立于原始列


def test_normalize_a_prefers_a_row_column(a_batch, legal):
    a = loader.assign_a_rows(make_a_table([{}, {}]))
    std, _ = nz.normalize_a(a, a_batch, legal, verbose=False)
    assert list(std["a_row"]) == [2, 3]


def test_normalize_a_matches_scalar_functions(a_batch, legal):
    a = _a_scope()
    std, _ = nz.normalize_a(a, a_batch, legal, verbose=False)
    for i in range(len(a)):
        row = a.iloc[i]
        pid, bt = row["province_id"], row["batch_type"]
        province = nz.norm_province_id(pid)
        name = nz.map_a_batch_name(pid, bt, a_batch)
        body, frags = nz.clean_batch(name, province)
        layer, early, seg, order = nz.extract_components(body)
        name_std = nz.compose_batch_name((layer, early, seg, order))
        expected = {
            "school_id": nz.norm_code(row["school_id"]),
            "province": province,
            "major_name": nz.norm_major(row["special_name"]),
            "major_code": nz.norm_code(row["special_code"]),
            "subject": nz.norm_subject_a(row["subject"]),
            "batch_name": name, "batch_body": body, "batch_remark": frags,
            "batch_layer": layer, "early": early, "segment": seg, "batch_order": order,
            "batch_name_std": name_std,
            "batch_check": nz.batch_check(name, name_std, province, legal),
            "layer": layer,
            "major_group": nz.norm_major_group(row["major_group"]),
        }
        for col, value in expected.items():
            assert std.at[i, col] == value, (i, col, std.at[i, col], value)
        mask = nz.or_masks(
            *(nz.tag_mask(nz.norm_text(row[A_COL[k]]))
              for k in ("remark_major", "remark_plan", "remark_direction", "remark_group")),
            nz.tag_mask(frags), nz.tag_mask(body))
        assert std.at[i, "remark_mask"] == mask, i


def test_normalize_a_six_remark_sources_or_together(a_batch, legal):
    std, _ = nz.normalize_a(_a_scope(), a_batch, legal, verbose=False)
    assert std["remark_mask"].dtype == object
    assert set(nz.mask_to_tags(std.at[0, "remark_mask"])) == {"中外合作", "民族类", "联合培养"}   # “非定向”被否定
    assert "民族类" in nz.mask_to_tags(std.at[3, "remark_mask"])      # 来自批次备注片段“蒙授”
    assert "定向" in nz.mask_to_tags(std.at[3, "remark_mask"])


def test_normalize_a_report_counts_partition_and_zero_printing(a_batch, legal):
    a = _a_scope()
    std, rep = nz.normalize_a(a, a_batch, legal, verbose=False)
    counts = rep.batch_check_counts
    assert list(counts) == ["空", "未识别", "合法", "不在本省清单"]
    assert sum(counts.values()) == rep.rows == len(a)
    assert counts == std["batch_check"].value_counts().reindex(list(counts)).fillna(0).astype(int).to_dict()
    assert rep.a_batch_unmapped == 2        # (130000, 10001)；(abc, 1) 无 exact 且 "1" 在全表对应多个名称
    assert rep.province_failed == 1 and rep.province_failed_examples == ["abc"]
    assert rep.province_rows == len(a)


def test_normalize_a_unmapped_counts_only_non_missing_batch_type(a_batch, legal):
    a = _a_scope([
        {"province": "130000", "batch": "10001"},
        {"province": "130000", "batch": None},
        {"province": "130000", "batch": ""},
        {"province": "130000", "batch": "1"},
    ])
    _std, rep = nz.normalize_a(a, a_batch, legal, verbose=False)
    assert rep.a_batch_unmapped == 1
    assert rep.a_batch_unmapped_examples == [("130000", "10001")]
    assert rep.batch_check_counts["空"] == 3


def test_normalize_a_province_failure_taken_from_scope_report(a_batch, legal):
    a = _a_scope([{"province": "130000"}])
    scope_rep = nz.ScopeReport(total=1000, kept=1, excluded=999, province_failed=7,
                               province_failed_examples=["q", "r"])
    _std, rep = nz.normalize_a(a, a_batch, legal, scope_report=scope_rep, verbose=False)
    assert rep.province_failed == 7 and rep.province_failed_examples == ["q", "r"]
    assert rep.province_rows == 1000 and rep.rows == 1


def test_normalize_a_empty_table_prints_zero_counts(a_batch, legal, capsys):
    a = make_a_table([]).iloc[0:0]
    std, rep = nz.normalize_a(a, a_batch, legal)
    assert len(std) == 0 and list(std.columns) == list(nz.A_STD_COLUMNS)
    assert rep.batch_check_counts == {"空": 0, "未识别": 0, "合法": 0, "不在本省清单": 0}
    out = capsys.readouterr().out
    assert "参与标准化行数=0" in out
    assert "空=0 未识别=0 合法=0 不在本省清单=0 | 合计=0" in out
    assert "batch_type 查不到批次名称行数=0" in out
    assert "选科置空" not in out                                       # 选科项只在 B 的报告里


def test_normalize_a_verbose_false_prints_nothing(a_batch, legal, capsys):
    nz.normalize_a(_a_scope(), a_batch, legal, verbose=False)
    assert capsys.readouterr().out == ""
    nz.normalize_a(_a_scope(), a_batch, legal)
    assert "[标准化诊断] 表=A" in capsys.readouterr().out


def test_normalize_a_examples_in_row_order_capped_at_20(a_batch, legal):
    rows = [{"province": "130000", "batch": str(20000 + i)} for i in range(25)]
    rows += [{"province": "130000", "batch": "20000"}]                # 重复值不重复列出
    _std, rep = nz.normalize_a(_a_scope(rows), a_batch, legal, verbose=False)
    assert rep.a_batch_unmapped == 26
    assert rep.a_batch_unmapped_examples == [("130000", str(20000 + i)) for i in range(20)]


# ------------------------------------------------------------
# normalize_b
# ------------------------------------------------------------

def test_normalize_b_does_not_touch_b_kept(legal):
    b = _b_kept(B_MIXED_ROWS)
    before = b.copy(deep=True)
    std, rep = nz.normalize_b(b, legal, verbose=False)
    pd.testing.assert_frame_equal(b, before)
    assert len(std) == len(b) == rep.rows
    assert list(std.columns) == list(nz.B_STD_COLUMNS)
    assert list(std["b_id"]) == list(b["b_id"])
    assert list(std["b_pos"]) == list(range(len(b)))


def test_normalize_b_generates_b_id_without_column(legal):
    b = make_b_table([{}, {}])
    b.index = pd.Index([5, 9])
    std, _ = nz.normalize_b(b, legal, verbose=False)
    assert list(std["b_id"]) == ["B_00000007", "B_00000011"]


def test_normalize_b_matches_scalar_functions(legal):
    b = _b_kept(B_MIXED_ROWS)
    std, _ = nz.normalize_b(b, legal, verbose=False)
    for i in range(len(b)):
        row = b.iloc[i]
        province = nz.norm_province_name(row[B_COL["province"]])
        raw_batch = row[B_COL["batch"]]
        body, frags = nz.clean_batch(raw_batch, province)
        b_layer, early, seg, order = nz.extract_components(body)
        name_std = nz.compose_batch_name((b_layer, early, seg, order))
        merged = nz.merge_batch_remark(row[B_COL["remark_batch"]], frags)
        layer = (nz.layer_from_column(row[B_COL["layer_primary"]])
                 or nz.layer_from_column(row[B_COL["layer_secondary"]]) or b_layer)
        expected = {
            "school_id": nz.norm_code(row[B_COL["school_id"]]),
            "province": province,
            "major_name": nz.norm_major(row[B_COL["major_name"]]),
            "major_code": nz.norm_code(row[B_COL["major_code"]]),
            "subject": nz.norm_subject_b(row[B_COL["subject"]], province),
            "batch_name": raw_batch, "batch_body": body, "batch_remark": frags,
            "batch_remark_merged": merged, "batch_layer": b_layer,
            "early": early, "segment": seg, "batch_order": order,
            "batch_name_std": name_std,
            "batch_check": nz.batch_check(raw_batch, name_std, province, legal),
            "layer": layer,
            "major_group": nz.norm_major_group(row[B_COL["major_group"]]),
        }
        for col, value in expected.items():
            assert std.at[i, col] == value, (i, col, std.at[i, col], value)
        mask = nz.or_masks(
            nz.tag_mask(nz.norm_text(row[B_COL["remark_major"]])),
            nz.tag_mask(merged),
            nz.tag_mask(nz.norm_text(row[B_COL["remark_plan"]])),
            nz.tag_mask(body))
        assert std.at[i, "remark_mask"] == mask, i
        for src, out in config.SCORE_COLUMNS:
            value = nz.norm_number(row[src])
            got = std.at[i, out]
            assert (np.isnan(got) if value is None else got == value), (i, out)


def test_normalize_b_three_level_layer_fill(legal):
    std, _ = nz.normalize_b(_b_kept([
        {"batch": "专科批", "layer_primary": "本科", "layer_secondary": "专科"},   # 层次列优先
        {"batch": "专科批", "layer_primary": None, "layer_secondary": "本科"},     # 层级列次之
        {"batch": "专科批", "layer_primary": "未知层", "layer_secondary": ""},     # 都不可判定 -> 批次
        {"batch": "特殊类型招生", "layer_primary": "", "layer_secondary": ""},     # 全空
        {"batch": "本科批", "layer_primary": "高职", "layer_secondary": None},
    ]), legal, verbose=False)
    assert list(std["layer"]) == ["本科", "本科", "专科", None, "专科"]
    assert std["layer"].iloc[3] is None


def test_normalize_b_scores_are_float_with_nan_for_missing_and_zero(legal):
    std, _ = nz.normalize_b(_b_kept([
        {"最低分": "612", "最高分": 0, "平均分": "暂无", "最低分位次": "1,234", "平均分位次": 9000.5, "录取人数": None},
    ]), legal, verbose=False)
    outs = [out for _src, out in config.SCORE_COLUMNS]
    assert all(std[o].dtype == np.float64 for o in outs)
    row = std.iloc[0]
    assert row["B_最低分"] == 612.0 and row["B_平均分位次"] == 9000.5
    assert all(np.isnan(row[o]) for o in ("B_最高分", "B_平均分", "B_最低分位次", "B_录取人数"))


def test_normalize_b_remark_mask_is_or_of_four_sources(legal):
    std, _ = nz.normalize_b(_b_kept([
        {"remark_major": "中外合作", "remark_batch": "联合培养", "remark_plan": "定向", "batch": "本科批地方专项计划"},
        {"remark_major": "中", "remark_batch": "外合作"},                    # 不跨来源
        {"remark_plan": "非定向"},
        {"batch": "本科批(预科)"},                                            # 括号片段进入合并批次备注
    ]), legal, verbose=False)
    tags = [set(nz.mask_to_tags(m)) for m in std["remark_mask"]]
    assert tags[0] == {"中外合作", "联合培养", "定向", "专项"}
    assert tags[1] == set() and tags[2] == set()
    assert tags[3] == {"预科"}


def test_normalize_b_report(legal):
    b = _b_kept(B_MIXED_ROWS)
    std, rep = nz.normalize_b(b, legal, verbose=False)
    assert rep.table == "B" and rep.rows == len(b) and rep.province_rows == len(b)
    assert sum(rep.batch_check_counts.values()) == rep.rows
    assert rep.province_failed == 1 and rep.province_failed_examples == ["香港特别行政区"]
    # 选科：浙江“物理”、河北“综合”、四川“不限”(综合) 被按省规则置空；空值/空串不算
    assert rep.subject_dropped == 3
    assert rep.subject_dropped_examples == [("浙江", "物理"), ("河北", "综合"), ("四川", "不限")]
    # 抽样核对示例：未识别的批次、缺失（None 与空串）批次计入“空”
    assert ("新疆", "特殊类型招生") in rep.unrecognized_examples
    assert rep.batch_check_counts["空"] == 2
    # 省份无法转换（标准省份 None）时没有省合法集合 -> 不在本省清单；河北“本科批”在清单内
    assert rep.not_in_list_examples[0] == (None, "本科批")
    assert std.at[0, "batch_check"] == "合法"


def test_normalize_b_remark_hits_units(legal):
    _std, rep = nz.normalize_b(_b_kept([
        {"remark_major": "中外合作", "remark_plan": "非定向", "remark_batch": "xx", "batch": "本科批(蒙授)"},
        {"remark_major": "中外合作", "remark_plan": "", "remark_batch": None, "batch": "本科批"},
    ]), legal, verbose=False)
    hits = rep.remark_hits
    assert hits[("中外合作", True)] == 2
    assert hits[("非定向", False)] == 1                                   # “非”否定前缀 -> 未命中
    assert hits[("xx", False)] == 1                                       # 合并批次备注的第一个片段
    assert hits[("蒙授", True)] == 1                                      # 括号片段
    assert sum(hits.values()) == 5


def test_normalize_b_report_print_format(legal, capsys):
    _std, rep = nz.normalize_b(_b_kept(B_MIXED_ROWS), legal)
    out = capsys.readouterr().out
    assert "[标准化诊断] 表=B | 参与标准化行数=8" in out
    assert "批次核查: 空=2" in out and "合计=8" in out
    assert "选科置空" in out and "batch_type 查不到" not in out
    # 示例取前 20 个不同值；计数为 0 也打印
    assert "香港特别行政区" in out
    empty_std, empty_rep = nz.normalize_b(make_b_table([]).iloc[0:0], legal)
    out2 = capsys.readouterr().out
    assert len(empty_std) == 0 and "参与标准化行数=0" in out2 and "选科置空（原值非缺失、标准选科为缺失）行数=0" in out2


def test_normalize_b_verbose_false_prints_nothing(legal, capsys):
    nz.normalize_b(_b_kept(B_MIXED_ROWS), legal, verbose=False)
    assert capsys.readouterr().out == ""


def test_normalize_b_not_in_list_and_legal(legal):
    std, rep = nz.normalize_b(_b_kept([
        {"province": "四川", "batch": "本科批次B段-蒙授"},   # 四川 legal 含“本科批B段”
        {"province": "河北", "batch": "本科批B段"},          # 河北 legal 不含
    ]), legal, verbose=False)
    assert list(std["batch_check"]) == ["合法", "不在本省清单"]
    assert rep.not_in_list_examples == [("河北", "本科批B段")]


# ------------------------------------------------------------
# 批次链函数对象共用（Req 16.2）与禁用的逐行操作（Req 3.12）
# ------------------------------------------------------------

def test_b_batch_chain_is_shared_module_level_function(legal, monkeypatch):
    bound = nz.bind_b_batch_chain(legal)
    assert bound.func is nz.b_batch_chain
    assert callable(nz.b_batch_chain) and nz.b_batch_chain.__module__ == "normalizer"
    # 经 vec_map_pair 用同一个函数对象得到的结果与 normalize_b 的批次列一致
    b = _b_kept(B_MIXED_ROWS)
    std, _ = nz.normalize_b(b, legal, verbose=False)
    province = nz.vec_map(b[B_COL["province"]], nz.norm_province_name)
    out = nz.vec_map_pair(province, b[B_COL["batch"]], bound, 10)
    assert list(out[6]) == list(std["batch_name_std"]) and list(out[7]) == list(std["batch_check"])

    # normalize_b 确实调用模块级 b_batch_chain（替换后能被观察到）
    calls = []
    original = nz.b_batch_chain

    def spy(*args, **kwargs):
        calls.append(args[:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(nz, "b_batch_chain", spy)
    nz.normalize_b(b, legal, verbose=False)
    assert calls, "normalize_b 没有经过 b_batch_chain"
    assert len(calls) <= len(b)


def test_batch_chain_called_once_per_unique_group(legal, monkeypatch):
    calls = []
    original = nz.b_batch_chain

    def spy(*args, **kwargs):
        calls.append(args[:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(nz, "b_batch_chain", spy)
    rows = [{"province": "河北", "batch": "本科批"}] * 50 + [{"province": "浙江", "batch": "一段"}] * 50
    nz.normalize_b(_b_kept(rows), legal, verbose=False)
    assert len(calls) == 2


def _forbidden_row_wise_calls(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("iterrows", "itertuples"):
            bad.append(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "apply":
            for kw in node.keywords:
                if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value in (1, "columns"):
                    bad.append("apply(axis=1)")
    return bad


@pytest.mark.parametrize("func", [
    nz.scope_keys, nz.split_scope, nz.normalize_a, nz.normalize_b,
    nz._batch_chain_core, nz.a_batch_chain, nz.b_batch_chain, nz.b_merged_remark_chain,
    nz._b_remark_hits, nz._first_distinct, nz._count_checks,
])
def test_table_level_functions_have_no_row_wise_iteration(func):
    assert _forbidden_row_wise_calls(func) == []


def test_normalize_b_scales_by_unique_values_not_rows(legal, monkeypatch):
    """5000 行、2 种 (省份, 批次) 组合：批次链只算 2 次，说明走的是唯一值广播。"""
    calls = []
    original = nz.b_merged_remark_chain

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(nz, "b_merged_remark_chain", spy)
    rows = [{"batch": "本科批" if i % 2 else "专科批"} for i in range(5000)]
    std, _ = nz.normalize_b(_b_kept(rows), legal, verbose=False)
    assert len(std) == 5000 and len(calls) == 2
