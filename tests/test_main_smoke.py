# -*- coding: utf-8 -*-
"""run_matching / Output_Writer 的合成数据冒烟测试（不读取 ABC/data）。"""
import pandas as pd

import config
import loader
import main as app
from tests import factories as f


def _run(b_rows=None):
    a = f.make_a_table([
        {"special_name": "法学", "batch_type": "1"},          # 命中 B1
        {"special_name": "不存在专业"},                        # U01
        {"special_name": None},                               # U05
        {"school_id": "9999", "special_name": "法学"},        # 范围外
    ])
    b_raw = loader.assign_b_ids(
        f.make_b_table(b_rows if b_rows is not None else [{"major_name": "法学"}]))
    reasons = loader.filter_reasons(b_raw)
    b_kept, _ = loader.filter_b(b_raw)
    outcome = app.run_matching(a, b_kept, f.batch_table(), None, config)
    outcome.B_all_raw = b_raw
    outcome.b_filter_reasons = reasons
    outcome.sheet_a = app.build_sheet_a(outcome)
    outcome.sheet_b = app.build_sheet_b(outcome)
    outcome.sheet_match = app.build_sheet_match(outcome)
    return outcome


def test_run_matching_end_to_end():
    out = _run()
    assert out.A_raw_rows == 4
    assert len(out.final) == 3
    assert list(out.final["reason"].fillna("")) == ["", "U01", "U05"]
    assert out.final.loc[0, "status"] == "MATCH"
    assert set(out.stage_seconds) == set(app.STAGE_NAMES)


def test_group_code_conflict_is_soft_end_to_end():
    a = f.make_a_table([{
        "special_name": "法学", "batch_type": "1",
        "major_group": "01", "special_code": "C1",
    }])
    b_raw = loader.assign_b_ids(f.make_b_table([{
        "major_name": "法学", "major_group": "02", "major_code": "C2",
    }]))
    reasons = loader.filter_reasons(b_raw)
    b_kept, _ = loader.filter_b(b_raw)
    outcome = app.run_matching(a, b_kept, f.batch_table(), None, config)
    outcome.B_all_raw = b_raw
    outcome.b_filter_reasons = reasons
    outcome.sheet_a = app.build_sheet_a(outcome)
    outcome.sheet_b = app.build_sheet_b(outcome)
    outcome.sheet_match = app.build_sheet_match(outcome)

    row = outcome.sheet_a.iloc[0]
    assert row["match_status"] == "MATCH"
    assert "专业组不一致" in row["risk"]
    assert "专业代码不一致" in row["risk"]
    assert "U04" not in set(outcome.final["reason"].dropna())


def test_major_name_parentheses_are_recalled_by_base_and_plain_a_wins():
    a = f.make_a_table([
        {"special_name": "汉语言文学", "batch_type": "1"},
        {"special_name": "汉语言文学（高校专项计划）", "batch_type": "1"},
    ])
    b_raw = loader.assign_b_ids(f.make_b_table([{"major_name": "汉语言文学"}]))
    reasons = loader.filter_reasons(b_raw)
    b_kept, _ = loader.filter_b(b_raw)
    outcome = app.run_matching(a, b_kept, f.batch_table(), None, config)
    outcome.B_all_raw = b_raw
    outcome.b_filter_reasons = reasons
    outcome.sheet_a = app.build_sheet_a(outcome)
    outcome.sheet_b = app.build_sheet_b(outcome)
    outcome.sheet_match = app.build_sheet_match(outcome)

    assert outcome.sheet_a["match_status"].tolist() == ["MATCH", "REVIEW"]
    assert outcome.sheet_a.loc[0, "special_name"] == "汉语言文学"
    assert outcome.sheet_a.loc[1, "reason"] == "U03"


def test_sheet_a_columns_and_values():
    out = _run()
    sheet_a = out.sheet_a
    assert len(sheet_a) == 3  # A_scope（学校 9999 的行被排除在范围外）
    assert list(sheet_a.columns[-7:]) == list(app.JUDGMENT_COLUMNS)
    assert "B_ID" not in sheet_a.columns
    for _src, col in config.SCORE_COLUMNS:
        assert col not in sheet_a.columns
    assert sheet_a.loc[1, "match_level_cn"] == config.NOT_MATCH_CN
    assert sheet_a.loc[2, "reason_desc"].startswith(config.REASON_CODES["U05"])


def test_sheet_b_full_rows_and_status():
    out = _run(b_rows=[
        {"major_name": "法学"},                                      # 已匹配
        {"major_name": "数学"},                                       # 未匹配（通过索引但未被占用）
        {"soft_delete": 1, "major_name": "物理"},                     # 未参与匹配(软删除)
        {"soft_delete": 1, "is_current": 0, "major_name": "化学"},    # 多原因
        {"school_id": None, "major_name": "生物"},                    # 必需字段缺失
    ])
    sheet_b = out.sheet_b
    assert len(sheet_b) == 5  # B 全量，不裁剪
    assert "b_id" in sheet_b.columns and "匹配状态" in sheet_b.columns
    status = sheet_b["匹配状态"].tolist()
    assert status[0] == "已匹配"
    assert status[1] == "未匹配"
    assert status[2] == "未参与匹配(软删除)"
    assert status[3] == "未参与匹配(软删除|当前无效)"
    assert status[4] == "未参与匹配(必需字段缺失)"


def test_sheet_match_only_match_rows_with_suffixes():
    out = _run()
    sheet_match = out.sheet_match
    assert len(sheet_match) == 1  # 只有 1 条 MATCH
    assert any(c.endswith("_A") for c in sheet_match.columns)
    assert any(c.endswith("_B") for c in sheet_match.columns)
    for col in app.JUDGMENT_COLUMNS:
        assert col in sheet_match.columns
        assert not col.endswith("_A") and not col.endswith("_B")
    for _src, col in config.SCORE_COLUMNS:
        assert col not in sheet_match.columns
    assert sheet_match.loc[0, "match_status"] == "MATCH"


def test_write_output_and_stats(tmp_path, capsys):
    out = _run()
    path = app.write_output(
        {app.SHEET_NAME_A: out.sheet_a, app.SHEET_NAME_B: out.sheet_b,
         app.SHEET_NAME_MATCH: out.sheet_match},
        tmp_path / "r.xlsx")
    assert path.exists() and not list(tmp_path.glob(".*tmp*"))
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {app.SHEET_NAME_A, app.SHEET_NAME_B, app.SHEET_NAME_MATCH}
    assert len(sheets[app.SHEET_NAME_A]) == 3
    assert len(sheets[app.SHEET_NAME_B]) == 1
    assert len(sheets[app.SHEET_NAME_MATCH]) == 1
    app.print_stats(out, path, 0)
    app.print_unmatched_remarks(out)
    assert "R101: " in capsys.readouterr().out


def test_write_output_failure_leaves_no_tmp_file(tmp_path, monkeypatch):
    out = _run()
    bad_path = tmp_path / "r.xlsx"

    def _boom(*a, **k):
        raise RuntimeError("boom")

    # ExcelWriter 正常完成写入与关闭（文件 handle 已释放），只让最后的 os.replace 失败。
    monkeypatch.setattr(app.os, "replace", _boom)
    try:
        app.write_output({app.SHEET_NAME_A: out.sheet_a}, bad_path)
        assert False, "应抛出 OutputWriteError"
    except Exception as e:
        assert type(e).__name__ == "OutputWriteError"
    assert not list(tmp_path.glob(".*tmp*"))
    assert not bad_path.exists()


def test_write_keyword_report(tmp_path):
    out = _run()
    path = app.write_keyword_report(out, tmp_path / "keys.csv")
    assert path is not None and path.exists()
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["表", "来源", "原文", "次数", "已命中关键词"]
