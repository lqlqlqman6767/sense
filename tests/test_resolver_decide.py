# -*- coding: utf-8 -*-
"""decide_for_a / resolve_one 的基本场景（Req 7.9、8.6）。合成数据。"""

import numpy as np
import pandas as pd

import config
import resolver
from resolver import resolve_one, resolve_candidates, bulk_unmatched
from candidate import CandidateSet

BASE = {"school_id": "1", "province": "河北", "major_name": "法学"}


def _a(**kw):
    return {**BASE, **kw}


def test_r601_batch_only_better():
    a = _a(early="否", segment="A", remark_mask=1)
    b1 = {"b_id": "B_1", "early": "否", "segment": "B", "remark_mask": 1}   # 备注M 批次C
    b2 = {"b_id": "B_2", "early": "否", "segment": "A", "remark_mask": 0}   # 备注U 批次M
    r = resolve_one(a, [b1, b2])
    assert (r.status, r.bi, r.rule, r.level) == ("MATCH", "B_2", "R601", "WEAK")
    assert "辅助证据消歧" in r.risk


def test_r501_remark_only_better():
    a = _a(early="否", remark_mask=1)
    b1 = {"b_id": "B_1", "remark_mask": 1}                 # 备注M 批次U
    b2 = {"b_id": "B_2", "early": "否", "remark_mask": 0}  # 备注U 批次M
    r = resolve_one(a, [b2, b1])
    assert (r.bi, r.rule, r.level) == ("B_1", "R501", "WEAK")


def test_r602_both_better():
    a = _a(early="否", remark_mask=1)
    b1 = {"b_id": "B_1", "early": "否", "remark_mask": 1}
    b2 = {"b_id": "B_2", "remark_mask": 0}
    r = resolve_one(a, [b1, b2])
    assert (r.bi, r.rule, r.level) == ("B_1", "R602", "WEAK")


def test_r201_decided_by_hard_keys():
    a = _a(subject="物理类", major_group="1", early="否", remark_mask=1)
    b1 = {"b_id": "B_1", "subject": "物理类"}
    b2 = {"b_id": "B_2", "major_group": "1", "early": "否", "remark_mask": 1}
    r = resolve_one(a, [b1, b2])
    assert (r.bi, r.rule, r.level) == ("B_1", "R201", "STRONG")
    assert "辅助证据消歧" not in r.risk
    assert r.evidence == {"subject": True, "major_group": False, "batch": False,
                          "remark": False, "code": False}
    assert [len(g) for g in r.priority] == [1, 1]


def test_single_candidate_core_rule():
    r = resolve_one(_a(), [{"b_id": "B_1"}])
    assert (r.status, r.rule, r.level, r.reason) == ("MATCH", "R101", "NORMAL", None)


def test_u04_all_hard_conflict_sorted_by_b_id(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "layer", True)
    monkeypatch.setitem(config.HARD_CONFLICT, "subject", True)
    a = _a(layer="本科", subject="物理类")
    r = resolve_one(a, [{"b_id": "B_2", "layer": "专科"},
                        {"b_id": "B_1", "layer": "专科", "subject": "历史类"}])
    assert (r.status, r.reason, r.bi) == ("UNMATCH", "U04", None)
    assert r.diagnostics.conflicts == (
        ("B_1", ("layer", "subject")), ("B_2", ("layer",)))


def test_hard_conflict_candidate_dropped_not_u04(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "layer", True)
    r = resolve_one(_a(layer="本科"), [{"b_id": "B_1", "layer": "专科"},
                                       {"b_id": "B_2"}])
    assert (r.status, r.bi) == ("MATCH", "B_2")


def test_u02_tie_listed_ascending():
    r = resolve_one(_a(), [{"b_id": "B_3"}, {"b_id": "B_1"}])
    assert (r.status, r.reason, r.bi, r.rule, r.level) == ("REVIEW", "U02", None, None, None)
    assert r.diagnostics.tied == ("B_1", "B_3")


def test_u05_and_u01():
    r = resolve_one({"school_id": "1", "province": None, "major_name": None}, [])
    assert (r.reason, r.diagnostics.missing) == ("U05", ("province", "major_name"))
    assert resolve_one(_a(), []).reason == "U01"


def test_resolve_candidates_matches_resolve_one_and_bulk():
    A_std = pd.DataFrame({
        "a_pos": [0, 1, 2], "a_row": [2, 3, 4],
        "layer": [None] * 3, "subject": ["物理类", None, None],
        "major_group": [None] * 3, "major_code": [None] * 3,
        "early": [None] * 3, "segment": [None] * 3, "batch_order": [None] * 3,
        "remark_mask": [0, 0, 0],
    }, dtype=object)
    B_std = pd.DataFrame({
        "b_pos": [0, 1], "layer": [None] * 2, "subject": ["物理类", None],
        "major_group": [None] * 2, "major_code": [None] * 2,
        "early": [None] * 2, "segment": [None] * 2, "batch_order": [None] * 2,
        "remark_mask": [0, 0],
    }, dtype=object)
    edges = pd.DataFrame({
        "a_pos": np.array([0, 0], dtype=np.int64),
        "b_pos": np.array([0, 1], dtype=np.int64),
        "b_id": ["B_1", "B_2"],
        "recall_mask": np.array([1, 3], dtype=np.int8),
    })
    cs = CandidateSet(edges=edges, u05_pos=np.array([1]), u05_missing=[("province",)],
                      empty_pos=np.array([2]), cand_pos=np.array([0]))
    res = resolve_candidates(A_std, B_std, cs)
    assert len(res) == 1 and (res[0].a_row, res[0].bi, res[0].rule) == (2, "B_1", "R201")
    assert res[0].priority[1][0].recall_sources == frozenset({"基础召回", "映射召回"})
    bulk = bulk_unmatched(A_std, cs)
    assert bulk["reason"].tolist() == ["U05", "U01"]
    assert bulk["missing"].tolist() == [("province",), ()]
    assert bulk["a_row"].tolist() == [3, 4]


SOFT_A = {**BASE, "layer": "本科", "subject": "物理类",
          "major_group": "01", "major_code": "C1", "remark_mask": 0}
SOFT_B = {"b_id": "B_1", "layer": "本科", "subject": "物理类",
          "major_group": "02", "major_code": "C2", "remark_mask": 0}


def test_group_and_code_conflicts_are_not_hard_by_default():
    r = resolve_one(SOFT_A, [SOFT_B])
    assert (r.status, r.rule, r.level) == ("MATCH", "R201", "STRONG")
    assert "专业组不一致" in r.risk
    assert "专业代码不一致" in r.risk
    assert r.diagnostics.conflicts == ()


def test_group_conflict_can_be_configured_as_hard(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "major_group", True)
    r = resolve_one(SOFT_A, [SOFT_B])
    assert (r.status, r.reason, r.bi) == ("UNMATCH", "U04", None)
    assert r.diagnostics.conflicts == (("B_1", ("major_group",)),)


def test_code_conflict_can_be_configured_as_hard(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "code", True)
    r = resolve_one(SOFT_A, [SOFT_B])
    assert (r.status, r.reason, r.bi) == ("UNMATCH", "U04", None)
    assert r.diagnostics.conflicts == (("B_1", ("code",)),)


def test_layer_and_subject_conflicts_are_soft_by_default():
    a = {**BASE, "layer": "本科", "subject": "物理类", "remark_mask": 0}
    b = {"b_id": "B_1", "layer": "专科", "subject": "历史类", "remark_mask": 0}
    r = resolve_one(a, [b])
    assert (r.status, r.rule, r.level) == ("MATCH", "R101", "NORMAL")
    assert "层次不一致" in r.risk
    assert "选科不一致" in r.risk


def test_layer_conflict_can_be_configured_as_hard(monkeypatch):
    monkeypatch.setitem(config.HARD_CONFLICT, "layer", True)
    a = {**BASE, "layer": "本科", "subject": "物理类", "remark_mask": 0}
    b = {"b_id": "B_1", "layer": "专科", "subject": "历史类", "remark_mask": 0}
    r = resolve_one(a, [b])
    assert (r.status, r.reason, r.bi) == ("UNMATCH", "U04", None)
