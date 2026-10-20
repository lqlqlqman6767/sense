# -*- coding: utf-8 -*-
"""招生计划 A 与 官网 B 匹配系统：主程序入口。

* ``run_matching``：纯内存核心流程（6 个阶段，不做任何文件 I/O），测试直接调用。
* Output_Writer：``build_sheet_a`` / ``build_sheet_b`` / ``build_sheet_match`` /
  ``write_output`` / ``write_keyword_report`` / ``print_stats`` / ``print_unmatched_remarks``。
* ``main``：I/O 编排；所有致命检查都在写文件之前完成，最外层统一捕获 MatchingError。

输出为单个 Excel，含 3 个 sheet：
* ``ASHEET``（招生计划表）：A_scope 原始字段 + 判定结果列。
* ``BSHEET``（官网数据表）：B 表全部原始数据（不裁剪行）+ b_id + 匹配状态。
* ``匹配表``：只含 status=MATCH 的记录，A 字段加 ``_A`` 后缀、B 字段加 ``_B`` 后缀，
  判定结果列不加后缀。
* 另写 ``output/备注关键词候选.csv`` 作为维护人员逐步补充关键词的诊断文件。
"""

import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
import loader
import normalizer
from assign import assign_scores
from candidate import build_candidates
from errors import MatchingError, OutputWriteError
from global_check import allocate
from index import build_index
from resolver import bulk_unmatched, resolve_candidates
from review_fallback import apply_review_fallback

# 6 个阶段，按顺序（Req 12.5）
STAGE_NAMES = ("标准化", "索引构建", "候选召回", "候选判断", "全局唯一检查", "赋分")
STATUSES = ("MATCH", "REVIEW", "UNMATCH")
LEVELS = ("STRONG", "NORMAL", "WEAK")
# 判定结果列（ASHEET 与 匹配表 共用，不加 _A/_B 后缀）
JUDGMENT_COLUMNS = (
    "match_status", "match_rule", "match_level", "match_level_cn",
    "reason", "reason_desc", "risk",
)
FINAL_COLUMNS = (
    "a_pos", "a_row", "status", "bi", "rule", "level", "reason", "risk",
    "reason_detail", "assignable", "fallback",
)
UNMATCHED_REMARK_TOP = 50

# BSHEET 匹配状态文案
B_STATUS_MATCHED = "已匹配"
B_STATUS_UNMATCHED = "未匹配"
B_STATUS_MISSING_REQUIRED = "未参与匹配(必需字段缺失)"

SHEET_NAME_A = "ASHEET"
SHEET_NAME_B = "BSHEET"
SHEET_NAME_MATCH = "匹配表"


# ============================================================
# 阶段计时
# ============================================================


class _StageInfo:
    """stage 上下文里可在进入后再设置的行数。"""

    def __init__(self, rows):
        self.rows = rows


@contextmanager
def stage(name, rows=None, seconds=None):
    """打印 ``[阶段] 名称 | 行数=… | 耗时=…s``；成功结束时把耗时记入 ``seconds``。"""
    info = _StageInfo(rows)
    t = time.perf_counter()
    yield info
    elapsed = time.perf_counter() - t
    if seconds is not None:
        seconds[name] = elapsed
    print(f"[阶段] {name} | 行数={info.rows} | 耗时={elapsed:.2f}s")


# ============================================================
# 结果结构
# ============================================================


@dataclass
class MatchOutcome:
    A_raw_rows: int = 0
    A_scope_raw: Optional[pd.DataFrame] = None
    final: Optional[pd.DataFrame] = None
    # B 全量原始数据（check_columns 之后、filter_b 之前），含生成的 b_id 列
    B_all_raw: Optional[pd.DataFrame] = None
    # 每条 B 全量记录的过滤原因（""=通过过滤），与 B_all_raw 索引对齐
    b_filter_reasons: Optional[pd.Series] = None
    # B_std 中 school_id/province/major_name 任一缺失的 b_id 集合（必需字段缺失，未入索引）
    b_missing_required_ids: frozenset = field(default_factory=frozenset)
    sheet_a: Optional[pd.DataFrame] = None
    sheet_b: Optional[pd.DataFrame] = None
    sheet_match: Optional[pd.DataFrame] = None
    b_filter_report: object = None
    normalize_reports: dict = field(default_factory=dict)
    scope_report: object = None
    index_excluded: int = 0
    allocation_stats: object = None
    stage_seconds: dict = field(default_factory=dict)
    b_remark_raw_counts: Counter = field(default_factory=Counter)


# ============================================================
# 核心流程（纯内存）
# ============================================================


def _empty_tuples(n):
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = ()
    return arr


def _missing_detail(missing):
    """U05 明细：MISSING=缺失字段中文名；按唯一元组缓存，避免重复拼接。"""
    cache = {}
    out = np.empty(len(missing), dtype=object)
    for i, t in enumerate(missing):
        t = tuple(t)
        if t not in cache:
            cache[t] = ("MISSING=" + ",".join(config.FIELD_LABELS.get(f, f) for f in t)) if t else ""
        out[i] = cache[t]
    return out


def _build_final(A_std, results, bulk):
    """把 Global_Checker 输出与 U05/U01 批量结果按 A_scope 位置合并为 final。"""
    n = len(A_std)
    cand = pd.DataFrame({
        "a_pos": [r.a_pos for r in results],
        "a_row": [r.a_row for r in results],
        "status": [r.status for r in results],
        "bi": [r.bi for r in results],
        "rule": [r.rule for r in results],
        "level": [r.level for r in results],
        "reason": [r.reason for r in results],
        "risk": pd.Series([tuple(r.risk) for r in results], dtype=object),
        "reason_detail": [r.reason_detail for r in results],
        "assignable": [bool(r.assignable) for r in results],
        "fallback": [bool(r.fallback) for r in results],
    }, columns=list(FINAL_COLUMNS))

    m = len(bulk)
    none_col = np.full(m, None, dtype=object)
    bulk_df = pd.DataFrame({
        "a_pos": bulk["a_pos"].to_numpy(dtype=np.int64),
        "a_row": bulk["a_row"].to_numpy(dtype=np.int64),
        "status": bulk["status"].to_numpy(dtype=object),
        "bi": none_col, "rule": none_col, "level": none_col,
        "reason": bulk["reason"].to_numpy(dtype=object),
        "risk": pd.Series(_empty_tuples(m), dtype=object),
        "reason_detail": _missing_detail(bulk["missing"].tolist()),
        "assignable": np.zeros(m, dtype=bool),
        "fallback": np.zeros(m, dtype=bool),
    }, columns=list(FINAL_COLUMNS))

    final = pd.concat([cand, bulk_df], ignore_index=True)
    final = final.sort_values("a_pos", kind="stable").reset_index(drop=True)
    assert len(final) == n and (final["a_pos"].to_numpy() == np.arange(n)).all(), \
        "最终结果必须覆盖 A_scope 每个位置恰好一次"
    return final


def _unmatched_remark_counts(b_report):
    """未命中任何备注标签的原文 -> 出现次数（Req 11.14）。"""
    counts = Counter()
    for (text, hit), n in b_report.remark_hits.items():
        if not hit:
            counts[text] += n
    return counts


def run_matching(A_raw, B_kept, a_batch, prov_list, cfg=config):
    """纯内存匹配流程：不做文件 I/O，返回 MatchOutcome。"""
    seconds = {}
    outcome = MatchOutcome(A_raw_rows=len(A_raw), stage_seconds=seconds)

    with stage("标准化", None, seconds) as st:
        legal = normalizer.build_province_legal_sets(a_batch, prov_list)
        B_std, b_report = normalizer.normalize_b(B_kept, legal)
        keys = normalizer.scope_keys(B_std)
        A_scope_raw, scope_report = normalizer.split_scope(A_raw, keys)
        A_std, a_report = normalizer.normalize_a(A_scope_raw, a_batch, legal, scope_report)
        st.rows = len(A_std)
    outcome.A_scope_raw = A_scope_raw
    outcome.scope_report = scope_report
    outcome.normalize_reports = {"A": a_report, "B": b_report}
    outcome.b_remark_raw_counts = _unmatched_remark_counts(b_report)

    required = ["school_id", "province", "major_name"]
    missing_required = B_std[required[0]].isna() | B_std[required[1]].isna() | B_std[required[2]].isna()
    outcome.b_missing_required_ids = frozenset(B_std.loc[missing_required, "b_id"])

    with stage("索引构建", len(B_std), seconds):
        index = build_index(B_std)
    outcome.index_excluded = index.excluded

    with stage("候选召回", len(A_std), seconds):
        cs = build_candidates(A_std, index, dict(getattr(cfg, "MAJOR_NAME_ALIAS", {}) or {}))

    with stage("候选判断", len(cs.cand_pos), seconds):
        results = resolve_candidates(A_std, B_std, cs)
        bulk = bulk_unmatched(A_std, cs)

    with stage("全局唯一检查", len(A_std), seconds):
        finals, alloc_stats = allocate(results, index.frame)
        final = _build_final(A_std, finals, bulk)
        final, fallback_count = apply_review_fallback(
            final, A_std, B_std, A_raw=A_scope_raw, B_raw=B_kept)
        print(f"[REVIEW补救] 弱匹配 {fallback_count} 条")
    outcome.allocation_stats = alloc_stats

    with stage("赋分", len(A_std), seconds):
        final = assign_scores(final, B_std)
    outcome.final = final

    print(f"匹配阶段总耗时: {sum(seconds.values()):.2f}s")
    return outcome


# ============================================================
# Output_Writer
# ============================================================


def _judgment_frame(final):
    """由 final 计算判定结果列（JUDGMENT_COLUMNS），索引与 final 对齐（0..n-1）。"""
    final = final.reset_index(drop=True)
    status = final["status"]
    is_match = status == "MATCH"
    level_cn = final["level"].map(config.LEVEL_CN)
    risk_map = {t: ("|".join(t) or None) for t in set(final["risk"])}
    detail = final["reason_detail"].fillna("").astype(object)
    desc = final["reason"].map(config.REASON_CODES)
    tail = ("；" + detail).where(detail != "", "")
    reason_desc = (desc + tail).where(final["reason"].notna(), None)

    return pd.DataFrame({
        "match_status": status,
        "match_rule": final["rule"].where(final["rule"].notna(), None),
        "match_level": final["level"].where(final["level"].notna(), None),
        "match_level_cn": level_cn.where(is_match, config.NOT_MATCH_CN).fillna(config.NOT_MATCH_CN),
        "reason": final["reason"].where(final["reason"].notna(), None),
        "reason_desc": reason_desc,
        "risk": final["risk"].map(risk_map),
    }, columns=list(JUDGMENT_COLUMNS))


def _province_display(value):
    """A 表省份列：6 位省份 ID -> 省份名称；已是名称或无法识别时保留原值。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    name = normalizer.norm_province_id(value)
    return name if name is not None else value


def build_sheet_a(outcome):
    """ASHEET：A_scope 原始列（空串 -> None）+ 判定结果列。不含 B_ID、不含回填分数列。"""
    out = outcome.A_scope_raw.reset_index(drop=True).copy()
    out = out.where(out != "", None)
    if "province_id" in out.columns:
        out["province_id"] = out["province_id"].map(_province_display)
    judgment = _judgment_frame(outcome.final)
    for col in JUDGMENT_COLUMNS:
        out[col] = judgment[col]
    return out


def build_sheet_b(outcome):
    """BSHEET：B 全量原始列（空串 -> None）+ b_id + 匹配状态。不裁剪行。"""
    out = outcome.B_all_raw.reset_index(drop=True).copy()
    out = out.where(out != "", None)

    b_id = out["b_id"]
    matched_ids = frozenset(outcome.final.loc[outcome.final["status"] == "MATCH", "bi"].dropna())
    filter_reason = outcome.b_filter_reasons.reset_index(drop=True)

    is_filtered = filter_reason != ""
    is_missing_required = b_id.isin(outcome.b_missing_required_ids)
    is_matched = b_id.isin(matched_ids)

    status = pd.Series(B_STATUS_UNMATCHED, index=out.index, dtype=object)
    status = status.where(~is_matched, B_STATUS_MATCHED)
    status = status.where(~(is_missing_required & ~is_filtered), B_STATUS_MISSING_REQUIRED)
    filtered_label = "未参与匹配(" + filter_reason + ")"
    status = status.where(~is_filtered, filtered_label)
    out["匹配状态"] = status
    return out


def build_sheet_match(outcome):
    """匹配表：只含 status=MATCH 的记录，A 字段加 _A、B 字段加 _B，判定结果列不加后缀。"""
    final = outcome.final.reset_index(drop=True)
    is_match = (final["status"] == "MATCH").to_numpy()

    a_cols = outcome.A_scope_raw.reset_index(drop=True)
    a_part = a_cols.loc[is_match].reset_index(drop=True)
    a_part = a_part.where(a_part != "", None)
    a_part.columns = [f"{c}_A" for c in a_part.columns]

    judgment = _judgment_frame(final).loc[is_match].reset_index(drop=True)

    bi = final.loc[is_match, "bi"].reset_index(drop=True).to_numpy()
    b_all = outcome.B_all_raw.reset_index(drop=True).copy()
    b_all = b_all.where(b_all != "", None)
    b_part = b_all.set_index("b_id").reindex(bi).reset_index()
    assert not b_part["b_id"].isna().any() and len(b_part) == len(bi), \
        "匹配表构建失败：final 中的 bi 必须都能在 B 全量数据中找到"
    b_part = b_part.drop(columns=["b_id"])
    b_part.columns = [f"{c}_B" for c in b_part.columns]

    out = pd.concat(
        [a_part.reset_index(drop=True), b_part.reset_index(drop=True),
         judgment.reset_index(drop=True)],
        axis=1,
    )
    return out


def _tmp_path(path):
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")


def write_output(sheets, path):
    """写出 {sheet 名: DataFrame} 到单个 Excel（多 sheet）。

    临时文件 + os.replace；失败抛 OutputWriteError，不留临时文件。
    任一 sheet 为空时只写表头并警告；任一 sheet 超过 Excel 单表行数上限时打印警告
    （仍写入该 sheet，Excel 会在打开时截断/报错，这里不静默丢数据，不做自动改写 CSV
    —— 多 sheet 场景下 CSV 无法承载，由用户决定如何处理超限数据）。
    """
    path = Path(path)
    for name, df in sheets.items():
        if len(df) == 0:
            print(f"[警告] sheet「{name}」行数为 0，仅写出表头")
        if len(df) > config.EXCEL_MAX_DATA_ROWS:
            print(f"[警告] sheet「{name}」行数 {len(df)} 超过 Excel 单表上限 "
                  f"{config.EXCEL_MAX_DATA_ROWS}，数据可能无法完整写入")

    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 - 统一包装
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise OutputWriteError(f"输出文件写入失败：{path}，原因：{type(e).__name__}: {e}") from e
    return path


def _print_counts(title, labels, series):
    counts = series.value_counts() if series is not None else {}
    print(f"{title}:")
    for label in labels:
        print(f"  {label}: {int(counts.get(label, 0))}")


def _u04_field_distribution(final):
    detail = final.loc[final["reason"] == "U04", "reason_detail"]
    if detail.empty:
        return pd.Series(dtype="int64")
    parts = (detail.str.removeprefix("CONFLICTS=").str.split("|").explode()
             .str.split(":").str[-1].str.split(",").explode())
    return parts[parts != ""].value_counts()


def print_stats(outcome, path, t0):
    """输出成功后的统计（Config 中每个取值都打印，0 也打印）。"""
    final = outcome.final
    sheet_a = outcome.sheet_a
    sheet_b = outcome.sheet_b
    sheet_match = outcome.sheet_match
    print("\n========== 完成 ==========")
    print(f"输出文件: {path}")
    print(f"A 表原始行数: {outcome.A_raw_rows}")
    print(f"ASHEET 行数: {len(sheet_a)}")
    print(f"被输出范围排除的 A 行数: {outcome.A_raw_rows - len(sheet_a)}")
    print(f"BSHEET 行数（B 表全量）: {len(sheet_b)}")
    print(f"匹配表行数（status=MATCH）: {len(sheet_match)}")
    _print_counts("match_status 数量", STATUSES, final["status"])
    _print_counts("match_level 数量", LEVELS, final["level"])
    cn_labels = [config.LEVEL_CN[k] for k in LEVELS] + [config.NOT_MATCH_CN]
    _print_counts("match_level_cn 数量", cn_labels, sheet_a["match_level_cn"])
    _print_counts("规则命中数量", [r[0] for r in config.RULE_TABLE], final["rule"])
    _print_counts("reason 数量", list(config.REASON_CODES), final["reason"])
    risk_counts = final["risk"].explode().value_counts()
    print("风险标签出现次数:")
    for tag in config.RISK_TAGS:
        print(f"  {tag}: {int(risk_counts.get(tag, 0))}")

    dist = _u04_field_distribution(final)
    print("U04 冲突字段分布（候选级）:")
    if dist.empty:
        print("  （无）")
    for name, cnt in dist.items():
        print(f"  {name}: {int(cnt)}")

    for table in ("A", "B"):
        rep = outcome.normalize_reports.get(table)
        if rep is None:
            continue
        parts = " ".join(f"{k}={rep.batch_check_counts.get(k, 0)}" for k in normalizer.BATCH_CHECK_VALUES)
        print(f"{table} 批次核查: {parts}")
    print(f"总耗时: {time.time() - t0:.1f}s")


def print_unmatched_remarks(outcome, top=UNMATCHED_REMARK_TOP):
    """A/B 侧未命中任何备注标签的原文：次数降序、原文升序，前 top 条。"""
    for table in ("A", "B"):
        rep = outcome.normalize_reports.get(table)
        if rep is None:
            continue
        counts = Counter()
        for (text, hit), n in rep.remark_hits.items():
            if not hit:
                counts[text] += n
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        print(f"\n{table} 侧未命中备注标签的原文（前{top}条）:")
        for text, cnt in ordered[:top]:
            print(f"  {cnt}\t{text}")
        print(f"未命中原文的不同取值总数: {len(ordered)}")


def print_unmatched_major_keywords(outcome, top=UNMATCHED_REMARK_TOP):
    """A/B 专业名称括号片段中未命中备注关键词的原文，按侧分别输出。"""
    for table in ("A", "B"):
        rep = outcome.normalize_reports.get(table)
        if rep is None:
            continue
        counts = Counter()
        for (text, hit), n in rep.major_name_hits.items():
            if not hit:
                counts[text] += n
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        print(f"\n{table} 侧专业名称括号片段未命中备注关键词（前{top}条）:")
        for text, cnt in ordered[:top]:
            print(f"  {cnt}\t{text}")
        print(f"未命中片段的不同取值总数: {len(ordered)}")


def keyword_candidates_frame(outcome):
    """把所有 A/B 备注/专业名称片段整理成可编辑的关键词候选表。"""
    rows = []
    for table in ("A", "B"):
        rep = outcome.normalize_reports.get(table)
        if rep is None:
            continue
        for source, hits in (("备注", rep.remark_hits),
                             ("专业名称括号", rep.major_name_hits)):
            for (text, hit), n in hits.items():
                rows.append({
                    "表": table,
                    "来源": source,
                    "原文": text,
                    "次数": int(n),
                    "已命中关键词": bool(hit),
                })
    frame = pd.DataFrame(rows, columns=["表", "来源", "原文", "次数", "已命中关键词"])
    if len(frame):
        frame = frame.sort_values(
            ["已命中关键词", "次数", "原文"], ascending=[True, False, True],
            kind="stable").reset_index(drop=True)
    return frame


def write_keyword_report(outcome, path=None):
    """写出关键词候选 CSV；失败只警告，不影响主匹配结果。"""
    path = Path(config.KEYWORD_REPORT if path is None else path)
    frame = keyword_candidates_frame(outcome)
    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        print(f"\n关键词候选报告: {path}（{len(frame)} 行，按“未命中、次数降序”排序）")
        return path
    except Exception as e:  # noqa: BLE001 - 诊断文件失败不影响主结果
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        print(f"[警告] 关键词候选报告写入失败：{path}，原因：{type(e).__name__}: {e}")
        return None


# ============================================================
# 入口编排
# ============================================================


def main():
    t0 = time.time()
    config.validate_config()

    a_batch = loader.read_a_batch_table(config.A_BATCH_TABLE)
    prov_list = loader.read_province_batch_list(config.PROVINCE_BATCH_LIST, config.PROVINCE_BATCH_SHEET)

    B_raw = loader.read_table(config.INPUT_B, config.SHEET_B)
    B_raw = loader.assign_b_ids(B_raw)
    loader.check_columns(B_raw, config.B_COLUMNS, "B")
    B_all_raw = B_raw  # 全量（check_columns 之后、filter_b 之前），BSHEET 使用
    b_filter_reasons = loader.filter_reasons(B_raw)
    B_kept, b_filter_report = loader.filter_b(B_raw)

    A_raw = loader.read_table(config.INPUT_A, config.SHEET_A)
    loader.check_columns(A_raw, config.A_COLUMNS, "A")

    outcome = run_matching(A_raw, B_kept, a_batch, prov_list, config)
    outcome.b_filter_report = b_filter_report
    outcome.B_all_raw = B_all_raw
    outcome.b_filter_reasons = b_filter_reasons

    outcome.sheet_a = build_sheet_a(outcome)
    outcome.sheet_b = build_sheet_b(outcome)
    outcome.sheet_match = build_sheet_match(outcome)

    path = write_output({
        SHEET_NAME_A: outcome.sheet_a,
        SHEET_NAME_B: outcome.sheet_b,
        SHEET_NAME_MATCH: outcome.sheet_match,
    }, config.OUTPUT)
    print_stats(outcome, path, t0)
    print_unmatched_remarks(outcome)
    print_unmatched_major_keywords(outcome)
    write_keyword_report(outcome)
    return 0


if __name__ == "__main__":
    try:
        main()
    except MatchingError as err:
        print(f"[错误] {type(err).__name__}: {err}")
        sys.exit(err.exit_code)
