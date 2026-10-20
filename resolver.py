# -*- coding: utf-8 -*-
"""Resolver：边表向量化比较、排序键与按 A 分组决策。

1. :func:`compare_edges`  按 a_pos / b_pos ``take`` 两侧字段，计算各字段三态、
   ``hard_conflict`` 与硬冲突字段名（每个字段都计算，不在第一个 CONFLICT 处停止）。
   哪些字段算硬冲突由 ``config.HARD_CONFLICT`` 决定，默认只保留 layer/subject。
2. :func:`encode_keys`    由三态计算 ``key_code``、核心规则与风险掩码。
   :func:`evaluate_edges` 把两步合成一个入口。
3. :func:`decide_for_a` / :func:`resolve_candidates` / :func:`resolve_one`
   按 A 分组选择最高相似度候选，并把剩余候选按顺序保留给 Global_Checker 回退。

三态内部编码：MATCH=1、UNKNOWN=0、CONFLICT=-1（int8）。
"""

import numpy as np
import pandas as pd

import config
import rule_engine

MATCH = 1
UNKNOWN = 0
CONFLICT = -1

# 标准批次参与比较的三个成分（层次成分与批次备注不参与）
BATCH_COMPONENTS = ("early", "segment", "batch_order")


def hard_state_fields():
    """当前生效的硬冲突字段（顺序 = CONFLICT_FIELDS 顺序，读取 config 运行时值）。"""
    return tuple(
        name for name in config.CONFLICT_FIELDS if config.HARD_CONFLICT[name]
    )


# ``code`` 对应的 A_std / B_std 列名
_STATE_COLUMN = {
    "layer": "layer",
    "subject": "subject",
    "major_group": "major_group",
    "code": "major_code",
}

EDGE_STATE_NAMES = (
    "layer", "subject", "major_group", "code", "batch", "remark", "major_tag",
)


def _mask_array(frame, column):
    """取标签掩码列；测试用合成表没有该列时按 0 处理。支持超过 64 位。"""
    if column in frame.columns:
        return frame[column].to_numpy(dtype=object)
    return np.zeros(len(frame), dtype=object)


# ============================================================
# 第一步：边表向量化比较
# ============================================================


def _present(values):
    """object 数组 -> 非缺失掩码（None、NaN、空串都算缺失）。"""
    s = pd.Series(values, dtype=object)
    return (s.notna() & (s != "")).to_numpy(dtype=bool)


def _take(frame, column, pos):
    return frame[column].to_numpy(dtype=object)[pos]


def _tri(a_vals, b_vals):
    """三态精确比较：任一方缺失 -> 0（双方都缺失也是 0）；相等 -> 1；其它 -> -1。"""
    both = _present(a_vals) & _present(b_vals)
    # 不相等比较只在 both 为真处有意义，其余位置被掩码屏蔽
    eq = np.zeros(len(a_vals), dtype=bool)
    if both.any():
        eq[both] = np.asarray(a_vals[both] == b_vals[both], dtype=bool)
    return np.where(both, np.where(eq, MATCH, CONFLICT), UNKNOWN).astype(np.int8)


def _batch_state(A_std, B_std, a_pos, b_pos):
    """标准批次：按提前标记、段、批序逐成分比较，不同成分之间不交叉比较。"""
    n = len(a_pos)
    diff_any = np.zeros(n, dtype=bool)
    both_any = np.zeros(n, dtype=bool)
    for comp in BATCH_COMPONENTS:
        av = _take(A_std, comp, a_pos)
        bv = _take(B_std, comp, b_pos)
        both = pd.notna(pd.Series(av, dtype=object)).to_numpy(dtype=bool) & \
            pd.notna(pd.Series(bv, dtype=object)).to_numpy(dtype=bool)
        diff = np.zeros(n, dtype=bool)
        if both.any():
            diff[both] = np.asarray(av[both] != bv[both], dtype=bool)
        diff_any |= diff
        both_any |= both
    return np.where(diff_any, CONFLICT, np.where(both_any, MATCH, UNKNOWN)).astype(np.int8)


def _remark_state(A_std, B_std, a_pos, b_pos):
    """备注：任一方掩码为 0 -> UNKNOWN；有公共位 -> MATCH；否则 CONFLICT。"""
    am = _mask_array(A_std, "remark_mask")
    bm = _mask_array(B_std, "remark_mask")
    out = np.empty(len(a_pos), dtype=np.int8)
    for i, (a, b) in enumerate(zip(am[a_pos], bm[b_pos])):
        a, b = int(a), int(b)
        out[i] = UNKNOWN if (a == 0 or b == 0) else (MATCH if (a & b) else CONFLICT)
    return out


def _major_tag_fields(A_std, B_std, a_pos, b_pos):
    """备注关键词的兼容状态与惩罚分。

    返回 (state, penalty)：
    - state: MATCH（完全一致）/ UNKNOWN（A 比 B 多标签）/ CONFLICT（B 有 A 缺失的标签）
    - penalty: 64 * B 有而 A 缺的标签数 + A 有而 B 多的标签数
      缺失 B 标签权重更高；用于 key_code 相同时的细粒度择优。
    """
    am = _mask_array(A_std, "remark_mask")[a_pos] | \
        _mask_array(A_std, "major_name_mask")[a_pos]
    bm = _mask_array(B_std, "remark_mask")[b_pos] | \
        _mask_array(B_std, "major_name_mask")[b_pos]
    state = np.empty(len(a_pos), dtype=np.int8)
    penalty = np.empty(len(a_pos), dtype=np.int64)
    for i, (a, b) in enumerate(zip(am, bm)):
        a, b = int(a), int(b)
        missing_bits = b & ~a
        extra_bits = a & ~b
        state[i] = CONFLICT if missing_bits else (UNKNOWN if extra_bits else MATCH)
        penalty[i] = 64 * missing_bits.bit_count() + extra_bits.bit_count()
    return state, penalty


def compare_edges(A_std, B_std, a_pos, b_pos):
    """对每条边计算全部字段三态。

    返回 dict：``layer/subject/major_group/code/batch/remark`` -> int8 数组、
    ``hard_conflict`` -> bool 数组、``conflicts`` -> object 数组
    （硬冲突边为冲突字段名元组，其余为 None）。

    哪些字段算硬冲突由 ``config.HARD_CONFLICT`` 决定；非硬字段的 CONFLICT
    不进入 ``hard_conflict``/``conflicts``，只通过 ``risk_mask`` 输出风险。
    """
    a_pos = np.asarray(a_pos, dtype=np.int64)
    b_pos = np.asarray(b_pos, dtype=np.int64)

    states = {}
    for name in config.CONFLICT_FIELDS:
        col = _STATE_COLUMN[name]
        states[name] = _tri(_take(A_std, col, a_pos), _take(B_std, col, b_pos))
    states["batch"] = _batch_state(A_std, B_std, a_pos, b_pos)
    states["remark"] = _remark_state(A_std, B_std, a_pos, b_pos)
    states["major_tag"], states["major_tag_penalty"] = _major_tag_fields(
        A_std, B_std, a_pos, b_pos)

    hard_fields = hard_state_fields()
    hard = np.zeros(len(a_pos), dtype=bool)
    for name in hard_fields:
        hard |= states[name] == CONFLICT

    conflicts = np.empty(len(a_pos), dtype=object)
    conflicts[:] = None
    for i in np.flatnonzero(hard):
        conflicts[i] = tuple(
            name for name in hard_fields if states[name][i] == CONFLICT
        )

    states["hard_conflict"] = hard
    states["conflicts"] = conflicts
    return states


# ============================================================
# 第二步：排序键编码、核心规则、风险掩码
# ============================================================


def encode_keys(states):
    """由三态计算 key_code、rank_code、core_rule、risk_mask。

    规范排序键 (k1,k2,k3,k4,-c5,k6,k7)：
    ``evidence14 = 8*k1+4*k2+2*k3+k4``，
    ``key_code = 12*evidence14 + 4*(2-c5) + 2*k6 + k7``。
    其中 k1..k4 只表示对应字段是否 MATCH；非硬字段的 CONFLICT 与 UNKNOWN
    一样为 0，不额外降权，只通过 risk_mask 记录风险。
    ``rank_code = key_code*4096 + (4095 - major_tag_penalty)``；
    major_tag_penalty 只用于 key_code 相同时的备注关键词择优。
    返回 dict：key_code(int16)、rank_code(int32)、core_rule(object)、risk_mask(int64)。
    """
    k1 = (states["layer"] == MATCH).astype(np.int64)
    k2 = (states["subject"] == MATCH).astype(np.int64)
    k3 = (states["major_group"] == MATCH).astype(np.int64)
    k4 = (states["code"] == MATCH).astype(np.int64)
    remark = states["remark"]
    batch = states["batch"]

    c5 = (remark == CONFLICT).astype(np.int64) + (batch == CONFLICT).astype(np.int64)
    k6 = (remark == MATCH).astype(np.int64)
    k7 = (batch == MATCH).astype(np.int64)

    evidence14 = 8 * k1 + 4 * k2 + 2 * k3 + k4
    key_code = (12 * evidence14 + 4 * (2 - c5) + 2 * k6 + k7).astype(np.int16)

    # 备注关键词作为 key_code 同等时的次级排序：惩罚分越小越优先。
    # 低 12 位放“4095 - 惩罚分”，保证 key_code 相差 1 时仍压过关键词差异。
    penalty = states["major_tag_penalty"].astype(np.int64)
    rank_code = (key_code.astype(np.int64) * 4096 +
                 (4095 - np.minimum(penalty, 4095))).astype(np.int32)

    # 核心证据 (code, subject, major_group)
    core = rule_engine.core_rule_vec(k4.astype(bool), k2.astype(bool), k3.astype(bool))

    risk = rule_engine.risk_mask_vec({
        name: states[name] for name in EDGE_STATE_NAMES
    })

    return {
        "key_code": key_code,
        "rank_code": rank_code,
        "core_rule": core,
        "risk_mask": risk,
    }


def evaluate_edges(A_std, B_std, edges):
    """边表 -> 比较结果与排序键（合并为一个 dict，行与 edges 对齐）。"""
    states = compare_edges(
        A_std, B_std,
        edges["a_pos"].to_numpy(dtype=np.int64),
        edges["b_pos"].to_numpy(dtype=np.int64),
    )
    states.update(encode_keys(states))
    return states


# ============================================================
# 按 A 分组决策
# ============================================================

from dataclasses import dataclass, field
from typing import Optional

_STATE_TEXT = {MATCH: "MATCH", UNKNOWN: "UNKNOWN", CONFLICT: "CONFLICT"}

# 结果字典的键（Req 6.1、6.9）
_FIELD_STATE_KEYS = ("layer", "subject", "major_group", "code", "remark", "batch")
_EVIDENCE_KEYS = ("subject", "major_group", "batch", "remark", "code")


@dataclass(frozen=True)
class CandidateEntry:
    """有效候选在优先列表中的条目。"""

    bi: str
    rule: str
    level: str
    evidence: dict
    field_states: dict
    risk: tuple
    sort_key: tuple
    key_code: int
    rank_code: int
    recall_sources: frozenset


@dataclass(frozen=True)
class Diagnostics:
    tied: tuple = ()          # U02：并列 B_ID 升序
    conflicts: tuple = ()     # U04：((B_ID, 冲突字段名元组), ...) 按 B_ID 升序
    missing: tuple = ()       # U05：缺失的必需字段
    u03_detail: Optional[str] = None   # 由 Global_Checker 填写


@dataclass(frozen=True)
class ResolveResult:
    a_pos: int
    a_row: int
    status: str
    bi: Optional[str]
    rule: Optional[str]
    level: Optional[str]
    evidence: Optional[dict]
    field_states: Optional[dict]
    risk: tuple
    reason: Optional[str]
    priority: tuple = ()
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def _recall_sources(mask):
    mask = int(mask)
    return frozenset(
        name for bit, name in enumerate(config.RECALL_SOURCES) if mask >> bit & 1
    )


def _make_entry(sl, i, rule, extra_risk_mask=0):
    """由切片第 i 行构造 CandidateEntry。"""
    st = {name: int(sl[name][i]) for name in EDGE_STATE_NAMES}
    field_states = {k: _STATE_TEXT[st[k]] for k in _FIELD_STATE_KEYS}
    evidence = {k: st[k] == MATCH for k in _EVIDENCE_KEYS}
    c5 = int(st["remark"] == CONFLICT) + int(st["batch"] == CONFLICT)
    sort_key = (
        int(st["layer"] == MATCH), int(st["subject"] == MATCH),
        int(st["major_group"] == MATCH), int(st["code"] == MATCH),
        -c5, int(st["remark"] == MATCH), int(st["batch"] == MATCH),
        int(st["major_tag"]),
    )
    return CandidateEntry(
        bi=sl["b_id"][i],
        rule=rule,
        level=rule_engine.level_of(rule),
        evidence=evidence,
        field_states=field_states,
        risk=rule_engine.risk_tags(int(sl["risk_mask"][i]) | extra_risk_mask),
        sort_key=sort_key,
        key_code=int(sl["key_code"][i]),
        rank_code=int(sl["rank_code"][i]),
        recall_sources=_recall_sources(sl["recall_mask"][i]),
    )


def decide_for_a(a_pos, a_row, sl):
    """单条 A 的决策（生产路径与测试入口共用）。

    sl: 该 A 的候选边切片（dict，列均为等长数组），须已按
    (hard_conflict, -rank_code, b_id) 排序。
    """
    hard = np.asarray(sl["hard_conflict"], dtype=bool)
    valid = np.flatnonzero(~hard)

    if len(valid) == 0:
        # U04：全部候选硬冲突
        conflicts = sorted(
            (sl["b_id"][i], tuple(sl["conflicts"][i])) for i in range(len(hard))
        )
        return ResolveResult(
            a_pos=int(a_pos), a_row=int(a_row), status="UNMATCH",
            bi=None, rule=None, level=None, evidence=None, field_states=None,
            risk=(), reason="U04", priority=(),
            diagnostics=Diagnostics(conflicts=tuple(conflicts)),
        )

    # 按 rank_code 切成连续并列组（有效候选已按 rank_code 降序）
    rank_code = np.asarray(sl["rank_code"])
    key_code = np.asarray(sl["key_code"])
    groups = []  # 每组是切片下标列表
    for i in valid:
        if groups and rank_code[groups[-1][0]] == rank_code[i]:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    # 组内按 b_id 升序（仅用于展示）
    groups = [sorted(g, key=lambda j: sl["b_id"][j]) for g in groups]

    weak_bit = rule_engine.risk_bit(config.RISK_WEAK)
    priority = []
    for gi, g in enumerate(groups):
        entries = []
        for i in g:
            soft = (
                len(g) == 1 and gi + 1 < len(groups)
                and key_code[i] // 12 == key_code[groups[gi + 1][0]] // 12
                and key_code[i] != key_code[groups[gi + 1][0]]
            )
            if soft:
                nxt = groups[gi + 1]
                remark_better = sl["remark"][i] > min(sl["remark"][j] for j in nxt)
                batch_better = sl["batch"][i] > min(sl["batch"][j] for j in nxt)
                assert remark_better or batch_better, "键 5–7 分出优劣时必有软字段更优"
                rule = rule_engine.soft_rule(remark_better, batch_better)
                entries.append(_make_entry(sl, i, rule, weak_bit))
            else:
                entries.append(_make_entry(sl, i, str(sl["core_rule"][i])))
        priority.append(tuple(entries))
    priority = tuple(priority)

    first = priority[0]
    if len(first) == 1:
        e = first[0]
        return ResolveResult(
            a_pos=int(a_pos), a_row=int(a_row), status="MATCH",
            bi=e.bi, rule=e.rule, level=e.level,
            evidence=dict(e.evidence), field_states=dict(e.field_states),
            risk=e.risk, reason=None, priority=priority,
        )
    return ResolveResult(
        a_pos=int(a_pos), a_row=int(a_row), status="REVIEW",
        bi=None, rule=None, level=None, evidence=None, field_states=None,
        risk=(), reason="U02", priority=priority,
        diagnostics=Diagnostics(tied=tuple(e.bi for e in first)),
    )


def _sorted_edge_arrays(A_std, B_std, edges):
    """评估边表并按 (a_pos, hard_conflict, -rank_code, b_id) 排序，返回 (a_pos 数组, 列字典)。"""
    ev = evaluate_edges(A_std, B_std, edges)
    cols = {name: ev[name] for name in EDGE_STATE_NAMES}
    cols.update(
        hard_conflict=ev["hard_conflict"], conflicts=ev["conflicts"],
        key_code=ev["key_code"], rank_code=ev["rank_code"],
        core_rule=ev["core_rule"], risk_mask=ev["risk_mask"],
        b_id=edges["b_id"].to_numpy(dtype=object),
        recall_mask=edges["recall_mask"].to_numpy(),
    )
    a_pos = edges["a_pos"].to_numpy(dtype=np.int64)
    # object 数组不能直接 lexsort，先把 b_id 换成字典序名次
    _, b_rank = np.unique(cols["b_id"].astype(str), return_inverse=True)
    order = np.lexsort((
        b_rank, -cols["rank_code"].astype(np.int64),
        cols["hard_conflict"].astype(np.int8), a_pos,
    ))
    return a_pos[order], {k: np.asarray(v)[order] for k, v in cols.items()}


def resolve_candidates(A_std, B_std, cs):
    """生产路径：只处理 ``cs.cand_pos`` 中的 A，返回按 a_pos 升序的 ResolveResult 列表。"""
    edges = cs.edges
    if len(edges) == 0:
        return []
    a_sorted, cols = _sorted_edge_arrays(A_std, B_std, edges)
    a_rows = A_std["a_row"].to_numpy(dtype=np.int64)
    bounds = np.concatenate((
        [0], np.flatnonzero(np.diff(a_sorted)) + 1, [len(a_sorted)],
    ))
    results = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        a = int(a_sorted[lo])
        sl = {k: v[lo:hi] for k, v in cols.items()}
        results.append(decide_for_a(a, a_rows[a], sl))
    return results


def bulk_unmatched(A_std, cs):
    """批量生成 U05 与 U01 的结果列（不创建结果对象，也不调用 decide_for_a）。

    返回 DataFrame：a_pos, a_row, status, reason, missing。
    U05 的 missing 为缺失必需字段名元组；U01 的 missing 为空元组。
    """
    a_rows = A_std["a_row"].to_numpy(dtype=np.int64)
    u05 = np.asarray(cs.u05_pos, dtype=np.int64)
    u01 = np.asarray(cs.empty_pos, dtype=np.int64)
    pos = np.concatenate((u05, u01))
    missing = np.empty(len(pos), dtype=object)
    for i in range(len(pos)):
        missing[i] = ()
    for i, m in enumerate(cs.u05_missing):
        missing[i] = tuple(m)
    reason = np.concatenate((
        np.full(len(u05), "U05", dtype=object), np.full(len(u01), "U01", dtype=object),
    ))
    return pd.DataFrame({
        "a_pos": pos,
        "a_row": a_rows[pos] if len(pos) else np.empty(0, dtype=np.int64),
        "status": np.full(len(pos), "UNMATCH", dtype=object),
        "reason": reason,
        "missing": missing,
    })


_RESOLVE_ONE_DEFAULTS = (
    "layer", "subject", "major_group", "major_code",
    "early", "segment", "batch_order",
)


def resolve_one(a, candidates):
    """测试入口：a 与 candidates 为标准化后的字段映射，走同一个 decide_for_a。

    a 需含 school_id/province/major_name（缺任一返回 U05）；候选需含 b_id。
    其余标准列缺省为 None；remark_mask 与 major_name_mask 缺省为 0；
    a 可带 a_row（缺省 2）。
    """
    required = ("school_id", "province", "major_name")
    miss = tuple(f for f in required if a.get(f) is None)
    a_row = int(a.get("a_row", 2))
    if miss:
        return ResolveResult(
            a_pos=0, a_row=a_row, status="UNMATCH", bi=None, rule=None,
            level=None, evidence=None, field_states=None, risk=(), reason="U05",
            diagnostics=Diagnostics(missing=miss),
        )
    if len(candidates) == 0:
        return ResolveResult(
            a_pos=0, a_row=a_row, status="UNMATCH", bi=None, rule=None,
            level=None, evidence=None, field_states=None, risk=(), reason="U01",
        )

    def frame(rows):
        data = {k: [r.get(k) for r in rows] for k in _RESOLVE_ONE_DEFAULTS}
        data["remark_mask"] = [int(r.get("remark_mask", 0)) for r in rows]
        data["major_name_mask"] = [int(r.get("major_name_mask", 0)) for r in rows]
        return pd.DataFrame(data, dtype=object)

    A_std = frame([a])
    B_std = frame(list(candidates))
    n = len(candidates)
    edges = pd.DataFrame({
        "a_pos": np.zeros(n, dtype=np.int64),
        "b_pos": np.arange(n, dtype=np.int64),
        "b_id": np.array([c["b_id"] for c in candidates], dtype=object),
        "recall_mask": np.array([c.get("recall_mask", 1) for c in candidates], dtype=np.int8),
    })
    _, cols = _sorted_edge_arrays(A_std, B_std, edges)
    return decide_for_a(0, a_row, cols)
