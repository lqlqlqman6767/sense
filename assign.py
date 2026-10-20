# -*- coding: utf-8 -*-
"""Assigner：按 B_ID 向量化回填 B 的 7 个分数字段。

只读取 final 的 ``status``、``bi``、``level``；只写 7 个 ``B_`` 输出列。
不改动任何判定字段，也不改动 A 的原始列。
"""

import numpy as np
import pandas as pd

import config
from errors import FatalDataError

_MAX_LISTED = 20


def _fmt(values, limit=_MAX_LISTED):
    values = list(values)
    text = ",".join(str(v) for v in values[:limit])
    return text + (f"…（共 {len(values)} 个）" if len(values) > limit else "")


def assign_scores(final: pd.DataFrame, B_std: pd.DataFrame) -> pd.DataFrame:
    """返回 final 的副本，追加 7 个分数输出列（行数、行序、索引不变）。"""
    out_cols = [out for _src, out in config.SCORE_COLUMNS]

    status = final["status"]
    bi = final["bi"]
    level = final["level"]

    need = ((status == "MATCH") & bi.notna()).to_numpy()

    # bi 必须存在于 B_std（Req 10.8）
    known = bi.isin(set(B_std["b_id"]))
    bad = need & ~known.to_numpy()
    if bad.any():
        bad_rows = final.loc[bad]
        a_rows = (bad_rows["a_row"].tolist() if "a_row" in bad_rows.columns
                  else bad_rows.index.tolist())
        raise FatalDataError(
            "赋分失败：最终结果中的 B_ID 不在参与匹配的 B 中；"
            f"B_ID={_fmt(sorted(set(bad_rows['bi'])))}；A 行号={_fmt(a_rows)}")

    # 一条 B 最多服务一条 A：Global_Checker 已保证，这里再做一次赋分边界兜底。
    used = bi[need]
    duplicated = used[used.duplicated()].drop_duplicates()
    if len(duplicated):
        dup_mask = need & bi.isin(set(duplicated)).to_numpy()
        dup_rows = final.loc[dup_mask]
        a_rows = (dup_rows["a_row"].tolist() if "a_row" in dup_rows.columns
                  else dup_rows.index.tolist())
        raise FatalDataError(
            "赋分失败：同一 B_ID 被多条 A 使用；"
            f"B_ID={_fmt(sorted(set(duplicated)))}；A 行号={_fmt(a_rows)}")

    n = len(final)
    right = B_std[["b_id"] + out_cols]
    left = final.drop(columns=[c for c in out_cols if c in final.columns])
    merged = left.merge(right, how="left", left_on="bi", right_on="b_id",
                        validate="many_to_one", suffixes=("", "__b"))
    if len(merged) != n:
        raise FatalDataError(f"赋分合并后行数发生变化：{n} -> {len(merged)}")
    if "b_id" in merged.columns and "b_id" not in left.columns:
        merged = merged.drop(columns=["b_id"])
    elif "b_id__b" in merged.columns:
        merged = merged.drop(columns=["b_id__b"])
    merged.index = final.index

    ok = need & level.isin(config.ALLOW_ASSIGN_LEVELS).to_numpy()
    for col in out_cols:
        merged[col] = merged[col].where(pd.Series(ok, index=merged.index), np.nan)
    return merged
