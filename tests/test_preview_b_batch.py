# -*- coding: utf-8 -*-
"""预览工具：build_preview_frame（任务 15.1，Req 16.2–16.4、16.9、16.10）。"""

import pandas as pd
import pytest

import config
import normalizer as nz
import preview_b_batch as pv
from errors import FatalDataError
from loader import ABatchTable

B_COLS = list(config.B_COLUMNS.values())


def _table():
    return ABatchTable(exact={("130000", "1"): "本科批"}, by_type={"1": "本科批"}, ambiguous={})


def _b(rows, cols=None):
    cols = cols or ["院校ID", "省份", "批次", "批次备注", "专业备注", "计划性质（如 非定向）"]
    return pd.DataFrame(rows, columns=cols, dtype=object)


def test_columns_and_values():
    B = _b([["001", "河北省", "本科批(国家专项)", "仅限男生", "", ""],
            ["002", "河北", "", "", "", ""]])
    out = pv.build_preview_frame(B, _table(), None)
    assert list(out.columns) == list(B.columns) + list(pv.PREVIEW_APPEND_COLUMNS)
    assert len(out) == 2
    assert out["院校ID"].tolist() == ["001", "002"]  # 前导零保留
    assert out["B_ID"].tolist() == ["B_00000002", "B_00000003"]
    assert out["标准省份"].tolist() == ["河北", "河北"]
    assert out["规范批次名"][0] == "本科批"
    assert out["批次核查结果"].tolist() == ["合法", "空"]
    assert out["抽出批次备注"][0] == "国家专项"
    assert out["合并批次备注"][0].startswith("仅限男生" + config.REMARK_SEPARATOR)
    assert out.loc[1, ["批次主体", "规范批次名", "提前标记", "段", "批序", "备注标签"]].tolist() == [""] * 6
    assert not out.isin(["nan", "None"]).any().any()


def test_missing_columns_warn_and_empty(capsys):
    B = _b([["001", "河北"]], cols=["院校ID", "省份"])
    out = pv.build_preview_frame(B, _table(), None)
    warned = capsys.readouterr().out
    assert warned.count("[警告]") == 4
    assert list(out.columns) == ["院校ID", "省份"] + list(pv.PREVIEW_APPEND_COLUMNS)
    assert out.loc[0, "原批次"] == "" and out.loc[0, "批次核查结果"] == "空"


def test_empty_b_raises():
    with pytest.raises(FatalDataError):
        pv.build_preview_frame(_b([]), _table(), None)


def test_consistent_with_normalize_b():
    B = _b([["1", "四川省", "本科批B段(定向)", "只招男生", "不招色盲", "非定向"],
            ["2", "河北", "高职(专科)提前批", "", "", "定向"]])
    legal = nz.build_province_legal_sets(_table(), None)
    std, _ = nz.normalize_b(B.copy(), legal, verbose=False)
    out = pv.build_preview_frame(B, _table(), None)
    for i in range(len(B)):
        assert out.loc[i, "规范批次名"] == (std.loc[i, "batch_name_std"] or "")
        assert out.loc[i, "批次核查结果"] == std.loc[i, "batch_check"]
        tags = out.loc[i, "备注标签"]
        expect = config.REMARK_SEPARATOR.join(nz.mask_to_tags(std.loc[i, "remark_mask"]))
        assert tags == expect
