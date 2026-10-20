# -*- coding: utf-8 -*-
"""Data_Loader 测试。"""
import pandas as pd
from hypothesis import given, strategies as st

import config
import loader
from errors import FatalDataError
from tests.reference import naive_filter_b

_SD = config.B_COLUMNS["soft_delete"]
_CUR = config.B_COLUMNS["is_current"]
_GR = config.B_COLUMNS["granularity"]

_flag_values = st.sampled_from(
    [0, "0", 0.0, "0.0", " 0 ", "\u30000", 1, "1", 1.0, "1.0", " 1 ", "",
     None, "2", "x", "00", "01", "1.00", "-1", 2.0]
)
_gran_values = st.sampled_from(
    ["专业分", " 专业分 ", "专业分\u3000", "专业组分", "院校分", "", None, "专业 分"]
)
_rows = st.lists(
    st.fixed_dictionaries({_SD: _flag_values, _CUR: _flag_values, _GR: _gran_values}),
    min_size=0, max_size=40,
)


# Feature: admission-plan-matching-refactor, Property 1: B 过滤正确且计数守恒
# **Validates: Requirements 2.2, 2.3, 2.4**
@given(rows=_rows)
def test_property_1_filter_b_matches_naive_and_conserves_counts(rows):
    df = pd.DataFrame(rows, columns=[_SD, _CUR, _GR], dtype=object)
    df["tag"] = list(range(len(rows)))  # 用于核对保留的是哪些行

    expected_kept, expected_excl = naive_filter_b(rows, _SD, _CUR, _GR)

    if not expected_kept:
        try:
            loader.filter_b(df)
        except FatalDataError:
            return
        raise AssertionError("保留 0 条时应抛 FatalDataError")

    kept, report = loader.filter_b(df)

    assert list(kept["tag"]) == expected_kept
    assert report.kept_rows == len(expected_kept)
    assert report.raw_rows == len(rows)
    assert list(report.excluded.values()) == [
        expected_excl["soft_delete"], expected_excl["is_current"], expected_excl["granularity"]
    ]
    assert sum(report.excluded.values()) + report.kept_rows == len(rows)


# Feature: admission-plan-matching-refactor, Property 2: B_ID 稳定且唯一
# **Validates: Requirements 2.5**
@given(rows=_rows, n_extra=st.integers(min_value=0, max_value=5))
def test_property_2_b_id_stable_and_unique(rows, n_extra):
    good = {_SD: 0, _CUR: 1, _GR: "专业分"}
    all_rows = list(rows) + [good] * (n_extra + 1)  # 保证过滤后至少保留 1 条
    df = pd.DataFrame(all_rows, columns=[_SD, _CUR, _GR], dtype=object)
    df["tag"] = list(range(len(all_rows)))

    with_ids = loader.assign_b_ids(df)
    expected = [f"B_{i + 2:08d}" for i in range(len(all_rows))]
    assert list(with_ids["b_id"]) == expected
    assert len(set(with_ids["b_id"])) == len(all_rows)

    kept, _ = loader.filter_b(with_ids)
    before = dict(zip(with_ids["tag"], with_ids["b_id"]))
    assert all(before[t] == b for t, b in zip(kept["tag"], kept["b_id"]))


# ============================================================
# 任务 3.9：loader 单元测试
# **Validates: Requirements 1.5, 1.8, 1.9, 1.10, 2.1, 2.7, 2.8, 2.9, 2.10, 2.11**
# ============================================================
import pytest

from errors import InputError
from tests.factories import (
    make_a_batch_df, make_a_table, make_b_table, make_province_batch_df,
    write_a_batch_csv, write_a_xlsx, write_b_xlsx, write_province_batch_xlsx,
)

_SD_LABEL = f"{_SD}={config.B_FLAG_FILTERS[0][1]}"
_CUR_LABEL = f"{_CUR}={config.B_FLAG_FILTERS[1][1]}"


def test_read_table_keeps_leading_zero_text(tmp_path):
    col = config.A_COLUMNS["school_id"]
    df = make_a_table([{"school_id": "010101"}])
    path = write_a_xlsx(tmp_path, df)
    out = loader.read_table(path)
    assert out[col].iloc[0] == "010101"


def test_read_table_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError):
        loader.read_table(tmp_path / "不存在.xlsx")


def test_read_table_missing_sheet_raises_input_error(tmp_path):
    path = write_a_xlsx(tmp_path)
    with pytest.raises(InputError):
        loader.read_table(path, sheet="不存在的表")


def test_check_columns_b_missing_granularity_is_fatal():
    df = make_b_table().drop(columns=[_GR])
    with pytest.raises(FatalDataError):
        loader.check_columns(df, config.B_COLUMNS, "B")


def test_filter_b_all_soft_deleted_is_fatal():
    df = make_b_table([{"soft_delete": 1}, {"soft_delete": 1}])
    with pytest.raises(FatalDataError):
        loader.filter_b(df)


def _check_and_filter(df):
    missing = loader.check_columns(df, config.B_COLUMNS, "B")
    kept, report = loader.filter_b(df)
    return missing, kept, report


def test_missing_soft_delete_column_skips_only_that_filter(capsys):
    df = make_b_table([{}, {"is_current": 0}]).drop(columns=[_SD])
    missing, kept, report = _check_and_filter(df)
    out = capsys.readouterr().out
    assert missing == ["soft_delete"]
    assert out.count(f"B 表缺少列：{_SD}") == 1
    assert report.executed[_SD_LABEL] is False
    assert report.executed[_CUR_LABEL] is True
    assert report.excluded[_CUR_LABEL] == 1
    assert len(kept) == 1
    assert "未执行" in out


def test_missing_is_current_column_skips_only_that_filter(capsys):
    df = make_b_table([{}, {"soft_delete": 1}]).drop(columns=[_CUR])
    missing, kept, report = _check_and_filter(df)
    out = capsys.readouterr().out
    assert missing == ["is_current"]
    assert out.count(f"B 表缺少列：{_CUR}") == 1
    assert report.executed[_CUR_LABEL] is False
    assert report.executed[_SD_LABEL] is True
    assert report.excluded[_SD_LABEL] == 1
    assert len(kept) == 1
    assert "未执行" in out


def test_missing_both_flag_columns_skips_both_filters(capsys):
    df = make_b_table([{}, {}]).drop(columns=[_SD, _CUR])
    missing, kept, report = _check_and_filter(df)
    out = capsys.readouterr().out
    assert set(missing) == {"soft_delete", "is_current"}
    assert out.count(f"B 表缺少列：{_SD}") == 1
    assert out.count(f"B 表缺少列：{_CUR}") == 1
    assert report.executed[_SD_LABEL] is False
    assert report.executed[_CUR_LABEL] is False
    assert len(kept) == 2
    assert out.count("未执行") == 2


def test_missing_normal_mapping_columns_warn_each_and_pad_empty(capsys):
    cols = [config.A_COLUMNS["major_name"], config.A_COLUMNS["school_id"]]
    df = make_a_table().drop(columns=cols)
    missing = loader.check_columns(df, config.A_COLUMNS, "A")
    out = capsys.readouterr().out
    assert set(missing) == {"major_name", "school_id"}
    for std, real in (("major_name", cols[0]), ("school_id", cols[1])):
        assert f"标准字段={std} 真实列名={real}" in out
    assert out.count("[警告]") == 2
    for real in cols:
        assert list(df[real]) == [""]


# ---------------- read_a_batch_table ----------------

def test_read_a_batch_table_gbk_roundtrip_and_dot_zero(tmp_path):
    df = make_a_batch_df([("10001.0", "1", "本科批"), ("10002", "2", "专科批")])
    path = write_a_batch_csv(tmp_path, df)
    t = loader.read_a_batch_table(path)
    assert t.exact[("10001", "1")] == "本科批"
    assert t.exact[("10002", "2")] == "专科批"
    assert t.by_type["1"] == "本科批"


def test_read_a_batch_table_duplicate_rows_merge(tmp_path):
    df = make_a_batch_df([("10001", "1", "本科批"), ("10001", "1", "本科批")])
    t = loader.read_a_batch_table(write_a_batch_csv(tmp_path, df))
    assert t.exact == {("10001", "1"): "本科批"}
    assert t.ambiguous == {}


def test_read_a_batch_table_multiple_names_warns_and_treated_as_missing(tmp_path, capsys):
    df = make_a_batch_df([("10001", "1", "本科批"), ("10001", "1", "本科一批")])
    t = loader.read_a_batch_table(write_a_batch_csv(tmp_path, df))
    out = capsys.readouterr().out
    assert ("10001", "1") not in t.exact
    assert ("10001", "1") in t.ambiguous
    assert "[警告]" in out and "多个批次名称" in out
    assert "1" not in t.by_type


def test_read_a_batch_table_missing_column_raises_input_error(tmp_path):
    df = make_a_batch_df([("10001", "1", "本科批")]).drop(columns=["batch_name"])
    path = write_a_batch_csv(tmp_path, df)
    with pytest.raises(InputError) as ei:
        loader.read_a_batch_table(path)
    assert ei.value.exit_code == 3


def test_read_a_batch_table_invalid_gbk_bytes_raises_input_error(tmp_path):
    path = tmp_path / "批次.csv"
    path.write_bytes(b"province_id,year,batch_name,batch_type\n1,2026,\xff\xff,1\n")
    with pytest.raises(InputError) as ei:
        loader.read_a_batch_table(path)
    assert ei.value.exit_code == 3


def test_read_a_batch_table_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError):
        loader.read_a_batch_table(tmp_path / "不存在.csv")


# ---------------- read_province_batch_list ----------------

def test_read_province_batch_list_ok(tmp_path):
    df = make_province_batch_df([("130000", "本科批")])
    out = loader.read_province_batch_list(write_province_batch_xlsx(tmp_path, df))
    assert list(out.columns) == list(config.PROVINCE_BATCH_COLUMNS)
    assert len(out) == 1


def test_read_province_batch_list_missing_file_warns_none(tmp_path, capsys):
    assert loader.read_province_batch_list(tmp_path / "不存在.xlsx") is None
    assert "[警告]" in capsys.readouterr().out


def test_read_province_batch_list_missing_sheet_warns_none(tmp_path, capsys):
    path = write_province_batch_xlsx(tmp_path, sheet="其他表")
    assert loader.read_province_batch_list(path) is None
    assert "[警告]" in capsys.readouterr().out


def test_read_province_batch_list_missing_column_warns_none(tmp_path, capsys):
    cols = list(config.PROVINCE_BATCH_COLUMNS)
    df = make_province_batch_df([("130000", "本科批")]).drop(columns=[cols[-1]])
    assert loader.read_province_batch_list(write_province_batch_xlsx(tmp_path, df)) is None
    out = capsys.readouterr().out
    assert "[警告]" in out and cols[-1] in out
