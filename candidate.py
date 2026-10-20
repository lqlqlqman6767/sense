# -*- coding: utf-8 -*-

"""
候选召回模块（Candidate_Builder）

- 必需字段（院校ID、省份、专业名称）任一缺失的 A 进入 U05 分区，不查询索引
- 基础召回与映射召回各做一次 merge，按 (院校ID, 省份, 专业名称主体) 精确连接
- 专业名称主体去掉括号内容；括号内容作为备注关键词在 resolver 中比较
- 不比较选科、层次、批次、专业组、备注、专业代码，不限制候选数
- 同一 (A, B) 只保留一行，召回来源以位掩码表示（位0基础召回、位1映射召回）
- 全部为列式向量化操作，不逐行遍历
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

_KEY_COLUMNS = ["school_id", "province", "major_base"]
_EDGE_COLUMNS = ["a_pos", "b_pos", "b_id", "recall_mask"]
_FIELD_NAMES = ("school_id", "province", "major_name")


@dataclass(frozen=True)
class CandidateSet:
    """候选召回结果。

    edges:       候选边表，列 [a_pos, b_pos, b_id, recall_mask]，按 (a_pos, b_id) 升序
    u05_pos:     必需字段缺失的 A 位置
    u05_missing: 与 u05_pos 对齐，每条缺失的必需字段名
    empty_pos:   必需字段齐全但无候选的 A 位置（U01）
    cand_pos:    至少有 1 个候选的 A 位置
    """

    edges: pd.DataFrame
    u05_pos: np.ndarray
    u05_missing: list
    empty_pos: np.ndarray
    cand_pos: np.ndarray


def _recall(keys, index_frame, mask_bit):
    """keys: 列 [a_pos, school_id, province, major_base]；返回边表片段。"""
    if len(keys) == 0 or len(index_frame) == 0:
        return pd.DataFrame({
            "a_pos": np.empty(0, dtype=np.int64),
            "b_pos": np.empty(0, dtype=np.int64),
            "b_id": pd.Series([], dtype=object),
            "recall_mask": np.empty(0, dtype=np.int8),
        })
    merged = keys.merge(index_frame, on=_KEY_COLUMNS, how="inner")
    return pd.DataFrame({
        "a_pos": merged["a_pos"].to_numpy(dtype=np.int64),
        "b_pos": merged["b_pos"].to_numpy(dtype=np.int64),
        "b_id": merged["b_id"].to_numpy(dtype=object),
        "recall_mask": np.full(len(merged), mask_bit, dtype=np.int8),
    })


def build_candidates(A_std, index, alias):
    """由 A_std 与 BIndex 生成 CandidateSet。alias: A 专业名称主体 -> B 专业名称主体（标准化后）。"""
    a_pos = A_std["a_pos"].to_numpy(dtype=np.int64)
    present = {f: A_std[f].notna().to_numpy() for f in _FIELD_NAMES}
    req_ok = present["school_id"] & present["province"] & present["major_name"]
    req_ok &= A_std["major_base"].notna().to_numpy()

    # U05：必需字段缺失，不查索引
    u05_idx = np.flatnonzero(~req_ok)
    u05_pos = a_pos[u05_idx]
    u05_missing = [
        tuple(f for f in _FIELD_NAMES if not present[f][i]) for i in u05_idx
    ]

    keys = pd.DataFrame({
        "a_pos": a_pos[req_ok],
        "school_id": A_std["school_id"].to_numpy(dtype=object)[req_ok],
        "province": A_std["province"].to_numpy(dtype=object)[req_ok],
        "major_base": A_std["major_base"].to_numpy(dtype=object)[req_ok],
    })

    # 基础召回
    parts = [_recall(keys, index.frame, 1)]

    # 映射召回：只对别名表中列出的专业名称主体
    if alias:
        mapped = keys["major_base"].map(alias)
        has = mapped.notna().to_numpy()
        if has.any():
            akeys = keys.loc[has].copy()
            akeys["major_base"] = mapped[has].to_numpy(dtype=object)
            parts.append(_recall(akeys, index.frame, 2))

    edges = pd.concat(parts, ignore_index=True)
    if len(edges) > 0:
        # 每种来源对同一 (A, B) 至多一行，各来源位互不重叠，求和即按位或
        edges["recall_mask"] = edges["recall_mask"].astype(np.int64)
        edges = edges.groupby(["a_pos", "b_pos"], sort=False, as_index=False).agg(
            b_id=("b_id", "first"), recall_mask=("recall_mask", "sum")
        )
    edges = edges[_EDGE_COLUMNS].copy()
    edges["a_pos"] = edges["a_pos"].astype(np.int64)
    edges["b_pos"] = edges["b_pos"].astype(np.int64)
    edges["recall_mask"] = edges["recall_mask"].astype(np.int8)
    edges = edges.sort_values(["a_pos", "b_id"], kind="stable").reset_index(drop=True)

    cand_pos = np.unique(edges["a_pos"].to_numpy(dtype=np.int64))
    ok_pos = a_pos[req_ok]
    empty_pos = ok_pos[~np.isin(ok_pos, cand_pos)]

    return CandidateSet(
        edges=edges,
        u05_pos=u05_pos,
        u05_missing=u05_missing,
        empty_pos=empty_pos,
        cand_pos=cand_pos,
    )


def recall_sources_text(mask):
    """召回来源位掩码 -> 文本，按 RECALL_SOURCES 顺序用“|”连接。"""
    mask = int(mask)
    return "|".join(
        name for bit, name in enumerate(config.RECALL_SOURCES) if mask >> bit & 1
    )
