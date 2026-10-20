# -*- coding: utf-8 -*-
"""Global_Checker 的竞争、回退、唯一性与 Assigner 边界测试（合成数据）。"""

import dataclasses

import pandas as pd
import pytest

import config
import normalizer as nz
from assign import assign_scores
from errors import FatalDataError
from global_check import allocate
from resolver import resolve_one

BASE = {"school_id": "1", "province": "河北", "major_name": "法学"}


def _result(a, candidates, a_pos, a_row):
    """走生产同一个 resolve_one，再设置 a_pos/a_row 以便 allocate 使用。"""
    return dataclasses.replace(resolve_one(a, candidates), a_pos=a_pos, a_row=a_row)


def test_higher_similarity_wins_and_loser_falls_back():
    a1 = {**BASE, "layer": "本科", "subject": "物理类",
          "major_group": "01", "remark_mask": 1}
    a2 = {**BASE, "layer": "本科", "remark_mask": 1}
    b1 = {"b_id": "B_1", "layer": "本科", "subject": "物理类",
          "major_group": "01", "remark_mask": 1}
    b2 = {"b_id": "B_2", "layer": "本科", "remark_mask": 0}

    out, stats = allocate([
        _result(a1, [b1, b2], 0, 2),
        _result(a2, [b1, b2], 1, 3),
    ], None)

    assert [(r.status, r.bi, r.fallback) for r in out] == [
        ("MATCH", "B_1", False),
        ("MATCH", "B_2", True),
    ]
    assert stats.first_match == 1
    assert stats.fallback_match == 1
    assert config.RISK_FALLBACK in out[1].risk


def test_tied_competition_stays_review_u03():
    a = {**BASE, "layer": "本科", "remark_mask": 1}
    b = {"b_id": "B_1", "layer": "本科", "remark_mask": 1}

    out, stats = allocate([
        _result(a, [b], 0, 2),
        _result(a, [b], 1, 3),
    ], None)

    assert [(r.status, r.reason, r.bi) for r in out] == [
        ("REVIEW", "U03", None),
        ("REVIEW", "U03", None),
    ]
    assert stats.tie_u03 == 2
    assert stats.unassigned_b == 1


def test_multiple_b_candidates_pick_highest():
    a = {**BASE, "layer": "本科", "subject": "物理类",
         "major_group": "01", "remark_mask": 1}
    b1 = {"b_id": "B_1", "layer": "本科", "subject": "物理类",
          "major_group": "01", "remark_mask": 1}
    b2 = {"b_id": "B_2", "layer": "本科"}

    out, stats = allocate([_result(a, [b1, b2], 0, 2)], None)
    assert (out[0].status, out[0].bi, out[0].fallback) == ("MATCH", "B_1", False)
    assert stats.first_match == 1


def test_exhausted_after_losing_b_marks_u03():
    a1 = {**BASE, "layer": "本科", "subject": "物理类",
          "major_group": "01", "remark_mask": 1}
    a2 = {**BASE, "layer": "本科", "remark_mask": 0}
    b1 = {"b_id": "B_1", "layer": "本科", "subject": "物理类",
          "major_group": "01", "remark_mask": 1}

    out, stats = allocate([
        _result(a1, [b1], 0, 2),
        _result(a2, [b1], 1, 3),
    ], None)

    assert (out[0].status, out[0].bi) == ("MATCH", "B_1")
    assert (out[1].status, out[1].reason, out[1].bi) == ("REVIEW", "U03", None)
    assert stats.exhausted_u03 == 1


def test_assign_scores_rejects_duplicate_b():
    out_cols = [out for _src, out in config.SCORE_COLUMNS]
    B_std = pd.DataFrame({
        "b_id": ["B_1"],
        **{col: [500.0] for col in out_cols},
    })
    final = pd.DataFrame({
        "a_row": [2, 3],
        "status": ["MATCH", "MATCH"],
        "bi": ["B_1", "B_1"],
        "level": ["STRONG", "STRONG"],
    })

    with pytest.raises(FatalDataError, match="同一 B_ID"):
        assign_scores(final, B_std)


def test_assign_scores_only_fills_allowed_match_rows():
    out_cols = [out for _src, out in config.SCORE_COLUMNS]
    B_std = pd.DataFrame({
        "b_id": ["B_1"],
        **{col: [500.0] for col in out_cols},
    })
    final = pd.DataFrame({
        "a_row": [2, 3],
        "status": ["MATCH", "REVIEW"],
        "bi": ["B_1", None],
        "level": ["STRONG", None],
    })

    out = assign_scores(final, B_std)
    assert out.loc[0, "B_最低分"] == 500.0
    assert pd.isna(out.loc[1, "B_最低分"])


def test_major_keyword_tiebreak_prefers_plain_a_when_b_has_no_tag():
    tag = nz.tag_mask("专项")
    b = {"b_id": "B_1", "remark_mask": 0, "major_name_mask": 0}

    out, _stats = allocate([
        _result({**BASE, "major_name_mask": 0}, [b], 0, 2),
        _result({**BASE, "major_name_mask": tag}, [b], 1, 3),
    ], None)

    assert (out[0].status, out[0].bi) == ("MATCH", "B_1")
    assert (out[1].status, out[1].reason) == ("REVIEW", "U03")


def test_major_keyword_tiebreak_prefers_exact_tag_over_missing():
    tag = nz.tag_mask("专项")
    b = {"b_id": "B_1", "remark_mask": 0, "major_name_mask": tag}

    out, _stats = allocate([
        _result({**BASE, "major_name_mask": tag}, [b], 0, 2),
        _result({**BASE, "major_name_mask": 0}, [b], 1, 3),
    ], None)

    assert (out[0].status, out[0].bi) == ("MATCH", "B_1")
    assert (out[1].status, out[1].reason) == ("REVIEW", "U03")


def test_major_keyword_tiebreak_prefers_fewer_extra_tags():
    base_tag = nz.tag_mask("专项")
    extra1 = nz.tag_mask("中外合作")
    extra2 = nz.tag_mask("联合培养")
    b = {"b_id": "B_1", "remark_mask": 0, "major_name_mask": base_tag}

    out, _stats = allocate([
        _result({**BASE, "major_name_mask": base_tag | extra1}, [b], 0, 2),
        _result({**BASE, "major_name_mask": base_tag | extra1 | extra2}, [b], 1, 3),
    ], None)

    assert (out[0].status, out[0].bi) == ("MATCH", "B_1")
    assert (out[1].status, out[1].reason) == ("REVIEW", "U03")
