# -*- coding: utf-8 -*-
"""Global_Checker：一条 B 最多对应一条 A 的全局唯一检查（Req 9、13.6、13.7、14.1、14.3）。

执行“A 提议、B 择优、并列持有（H20）”的延迟接受分配：

* 只有首选结果为 MATCH 的 A 参与；其余结果原样保留（含 bi 为空的 REVIEW / UNMATCH）。
* 按 (院校ID, 省份, 专业名称) 用并查集分组（优先列表跨多个专业名称的 A 把这些组合并），
  各组独立运行；组内只用 dict / set 查找。
* B 只依据排序键 ``rank_code``（key_code + 备注关键词次级分）择优；不使用 A 行号、
  B_ID 大小、输入顺序或随机数做选择。行号 / B_ID 的排序只出现在生成说明文字时，与选择无关。
* 被更高相似度 A 挤出的 A 会继续使用自己剩余候选（回退），因此一条 A 最终最多持有
  一条 B；标为 MATCH 的结果中 B_ID 必须唯一，函数末尾有断言兜底。
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import config

# ============================================================
# 数据结构
# ============================================================


@dataclass(frozen=True)
class FinalResult:
    """Global_Checker 输出的最终结果（与输入 ResolveResult 一一对应、顺序一致）。"""

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
    reason_detail: str = ""
    assignable: bool = False     # status == MATCH 且 level 在允许赋分等级内（Req 14.1）
    fallback: bool = False       # 是否使用次优候选


@dataclass(frozen=True)
class AllocationStats:
    participants: int = 0        # 参与分配的 A 数
    first_match: int = 0         # 获得首选 B 的 MATCH 数
    fallback_match: int = 0      # 经回退获得次优候选的 MATCH 数
    tie_u03: int = 0             # 因并列竞争输出 U03 的 A 数
    exhausted_u03: int = 0       # 因列表耗尽输出 U03 的 A 数
    tie_group_u02: int = 0       # 回退中遇到并列组输出 U02 的 A 数
    unassigned_b: int = 0        # 因并列竞争未分配的 B 数


# ============================================================
# 说明文字
# ============================================================


def _rows_text(rows):
    return ",".join(str(r) for r in sorted(rows))


def detail_text(result):
    """非参与结果（REVIEW/U02、UNMATCH/U04、U05 …）的明细文本，供原样保留的结果使用。"""
    d = result.diagnostics
    if result.reason == "U02" and d.tied:
        return "TIED=" + ",".join(d.tied)
    if result.reason == "U04" and d.conflicts:
        return "CONFLICTS=" + "|".join(
            f"{b}:{','.join(config.FIELD_LABELS.get(f, f) for f in fields)}"
            for b, fields in d.conflicts
        )
    if result.reason == "U05" and d.missing:
        return "MISSING=" + ",".join(config.FIELD_LABELS.get(f, f) for f in d.missing)
    if d.u03_detail:
        return d.u03_detail
    return ""


def _risk_with_fallback(risk):
    tags = set(risk) | {config.RISK_FALLBACK}
    return tuple(t for t in config.RISK_TAGS if t in tags)


# ============================================================
# 分组（并查集）
# ============================================================


class _UnionFind:
    def __init__(self):
        self._parent = {}

    def find(self, x):
        parent = self._parent
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:       # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _key_lookup(edge_keys):
    """edge_keys -> 函数 b_id -> 分组键。

    支持：dict / Series（b_id -> 分组键）、含 b_id、school_id、province、major_base 列的
    DataFrame、None（此时每条 B 自成一个键；仍保证正确，因为争抢同一 B 的 A 必然同组）。
    """
    if edge_keys is None:
        return lambda b: ("B", b)
    if hasattr(edge_keys, "columns"):   # DataFrame
        cols = ("school_id", "province", "major_base")
        mapping = dict(zip(
            edge_keys["b_id"].tolist(),
            zip(*(edge_keys[c].tolist() for c in cols)),
        ))
    else:
        mapping = dict(edge_keys)
    return lambda b: mapping.get(b, ("B", b))


# ============================================================
# 组内延迟接受
# ============================================================


def _run_group(members, priorities):
    """members: 参与 A 的下标列表；priorities[i]: 该 A 的优先列表（并列组序列）。

    返回 (ptr, state, holders, best, rejected)：
    state[a] ∈ {"HOLD", "EXHAUSTED", ("TIE", group)}。
    """
    ptr = {a: 0 for a in members}
    state = {}
    holders = {}
    best = {}
    rejected = {a: [] for a in members}
    free = deque(members)

    while free:
        a = free.popleft()
        groups = priorities[a]
        if ptr[a] == len(groups):
            state[a] = "EXHAUSTED"
            continue
        g = groups[ptr[a]]
        if len(g) >= 2:
            state[a] = ("TIE", g)
            continue
        b = g[0].bi
        k = g[0].rank_code
        if b not in best or k > best[b]:
            # 严格更优：原并列持有者全部被拒并回退
            for x in holders.get(b, ()):
                rejected[x].append(b)
                ptr[x] += 1
                free.append(x)
            holders[b] = {a}
            best[b] = k
            state[a] = "HOLD"
        elif k == best[b]:
            holders[b].add(a)
            state[a] = "HOLD"
        else:
            rejected[a].append(b)
            ptr[a] += 1
            free.append(a)
    return ptr, state, holders, best, rejected


# ============================================================
# 入口
# ============================================================


def _keep(r):
    """原样保留的非参与结果。"""
    return FinalResult(
        a_pos=r.a_pos, a_row=r.a_row, status=r.status, bi=r.bi, rule=r.rule,
        level=r.level, evidence=r.evidence, field_states=r.field_states,
        risk=tuple(r.risk), reason=r.reason, reason_detail=detail_text(r),
        assignable=_assignable(r.status, r.level), fallback=False,
    )


def _assignable(status, level):
    return status == "MATCH" and level in config.ALLOW_ASSIGN_LEVELS


def _unmatched(r, reason, detail):
    return FinalResult(
        a_pos=r.a_pos, a_row=r.a_row, status="REVIEW", bi=None, rule=None,
        level=None, evidence=None, field_states=None, risk=(), reason=reason,
        reason_detail=detail, assignable=False, fallback=False,
    )


def allocate(results, edge_keys=None):
    """延迟接受分配。返回 (list[FinalResult], AllocationStats)，条数与顺序同输入。"""
    results = list(results)
    key_of = _key_lookup(edge_keys)

    participants = [
        i for i, r in enumerate(results) if r.status == "MATCH" and r.priority
    ]
    priorities = {i: results[i].priority for i in participants}

    # 并查集：同一 A 优先列表里所有 B 的分组键合并
    uf = _UnionFind()
    root_of_member = {}
    for i in participants:
        first_key = None
        for grp in priorities[i]:
            for e in grp:
                k = key_of(e.bi)
                if first_key is None:
                    first_key = k
                    uf.find(k)
                else:
                    uf.union(first_key, k)
        root_of_member[i] = first_key
    buckets = {}
    for i in participants:
        buckets.setdefault(uf.find(root_of_member[i]), []).append(i)

    out = [None] * len(results)
    first_match = fallback_match = tie_u03 = exhausted_u03 = tie_group_u02 = 0
    unassigned_b = 0

    for members in buckets.values():
        ptr, state, holders, _best, rejected = _run_group(members, priorities)
        # B 最终持有者行号（只在生成说明文字时使用）
        holder_rows = {
            b: [results[a].a_row for a in hs] for b, hs in holders.items()
        }
        for b, hs in holders.items():
            if len(hs) >= 2:
                unassigned_b += 1
                detail = f"{config.U03_MARK};B_ID={b};A_ROWS={_rows_text(holder_rows[b])}"
                for a in hs:
                    out[a] = _unmatched(results[a], "U03", detail)
                    tie_u03 += 1
            else:
                (a,) = tuple(hs)
                r = results[a]
                if ptr[a] == 0:
                    out[a] = FinalResult(
                        a_pos=r.a_pos, a_row=r.a_row, status="MATCH", bi=r.bi,
                        rule=r.rule, level=r.level, evidence=r.evidence,
                        field_states=r.field_states, risk=tuple(r.risk), reason=None,
                        assignable=_assignable("MATCH", r.level), fallback=False,
                    )
                    first_match += 1
                else:
                    e = priorities[a][ptr[a]][0]
                    out[a] = FinalResult(
                        a_pos=r.a_pos, a_row=r.a_row, status="MATCH", bi=e.bi,
                        rule=e.rule, level=e.level, evidence=dict(e.evidence),
                        field_states=dict(e.field_states),
                        risk=_risk_with_fallback(e.risk), reason=None,
                        assignable=_assignable("MATCH", e.level), fallback=True,
                    )
                    fallback_match += 1
        for a in members:
            st = state[a]
            if st == "EXHAUSTED":
                parts = [
                    f"{b}[A_ROWS={_rows_text(holder_rows.get(b, ()))}]"
                    for b in rejected[a]
                ]
                detail = f"{config.U03_MARK};REJECTED=" + ",".join(parts)
                out[a] = _unmatched(results[a], "U03", detail)
                exhausted_u03 += 1
            elif st != "HOLD":
                tied = sorted(e.bi for e in st[1])
                out[a] = FinalResult(
                    a_pos=results[a].a_pos, a_row=results[a].a_row, status="REVIEW",
                    bi=None, rule=None, level=None, evidence=None, field_states=None,
                    risk=(), reason="U02", reason_detail="TIED=" + ",".join(tied),
                    assignable=False, fallback=False,
                )
                tie_group_u02 += 1

    for i, r in enumerate(results):
        if out[i] is None:
            out[i] = _keep(r)

    stats = AllocationStats(
        participants=len(participants), first_match=first_match,
        fallback_match=fallback_match, tie_u03=tie_u03,
        exhausted_u03=exhausted_u03, tie_group_u02=tie_group_u02,
        unassigned_b=unassigned_b,
    )
    assert stats.participants == (
        stats.first_match + stats.fallback_match + stats.tie_u03
        + stats.exhausted_u03 + stats.tie_group_u02
    ), "Global_Checker 统计不守恒"
    assert len(out) == len(results)
    # 一条 B 最多对应一条 A：任何路径都不得把同一 B_ID 写成多条 MATCH。
    used_bis = [r.bi for r in out if r.status == "MATCH"]
    assert len(used_bis) == len(set(used_bis)), "Global_Checker 破坏了 B 唯一性"
    print_stats(stats)
    return out, stats


def print_stats(stats):
    print(f"[全局唯一检查] 参与分配的 A 记录数: {stats.participants}")
    print(f"  获得首选 B 的 MATCH: {stats.first_match}")
    print(f"  经回退获得次优候选的 MATCH: {stats.fallback_match}")
    print(f"  因并列竞争输出 U03: {stats.tie_u03}")
    print(f"  因列表耗尽输出 U03: {stats.exhausted_u03}")
    print(f"  回退中遇到并列组输出 U02: {stats.tie_group_u02}")
    print(f"  因并列竞争未分配的 B 记录数: {stats.unassigned_b}")
