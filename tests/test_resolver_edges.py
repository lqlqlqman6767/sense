# -*- coding: utf-8 -*-
"""resolver 边表比较与排序键的基础检查（合成数据）。"""
import numpy as np
import pandas as pd

import config
import resolver
import rule_engine

COLS = ["layer", "subject", "major_group", "major_code", "early", "segment",
        "batch_order", "remark_mask"]


def _frame(rows):
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[:-1]:
        df[c] = df[c].astype(object)
    return df


def test_states_and_key_code(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "layer", True)
    nan = float("nan")
    A = _frame([
        ["本科", "物理类", "01", "C1", False, "A", None, 1],
        ["本科", None, "", nan, False, None, None, 0],
    ])
    B = _frame([
        ["本科", "物理类", "01", "C1", False, None, None, 1],
        ["专科", nan, "", nan, True, None, None, 2],
    ])
    s = resolver.compare_edges(A, B, [0, 1], [0, 1])
    assert list(s["layer"]) == [1, -1]
    assert list(s["subject"]) == [1, 0]
    assert list(s["major_group"]) == [1, 0]
    assert list(s["code"]) == [1, 0]  # 双方 NaN 也是 UNKNOWN
    assert list(s["batch"]) == [1, -1]
    assert list(s["remark"]) == [1, 0]
    assert list(s["hard_conflict"]) == [False, True]
    assert s["conflicts"][0] is None and s["conflicts"][1] == ("layer",)

    k = resolver.encode_keys(s)
    assert k["key_code"][0] == 12 * 15 + 4 * 2 + 2 + 1
    assert k["core_rule"][0] == "R401"
    assert k["key_code"][0] <= 191


def _soft_conflict_pair():
    A = _frame([["本科", "物理类", "01", "C1", False, "A", None, 0]])
    B = _frame([["本科", "物理类", "02", "C2", False, None, None, 0]])
    return A, B


def test_group_and_code_conflicts_are_soft_by_default():
    A, B = _soft_conflict_pair()
    s = resolver.compare_edges(A, B, [0], [0])
    assert s["hard_conflict"].tolist() == [False]
    assert s["conflicts"][0] is None
    tags = rule_engine.risk_tags(int(resolver.encode_keys(s)["risk_mask"][0]))
    assert "专业组不一致" in tags
    assert "专业代码不一致" in tags


def test_group_and_code_conflicts_can_be_hard(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "major_group", True)
    monkeypatch.setitem(config.HARD_CONFLICT, "code", True)
    A, B = _soft_conflict_pair()
    s = resolver.compare_edges(A, B, [0], [0])
    assert s["hard_conflict"].tolist() == [True]
    assert s["conflicts"][0] == ("major_group", "code")
