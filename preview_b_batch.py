# -*- coding: utf-8 -*-
"""B_Batch_Preview_Tool：B 批次清洗预览（Req 16）。

本文件目前包含读取流程与 ``build_preview_frame``（任务 15.1）；
写出、统计与 ``main`` 由任务 15.2 追加。

不过滤 B、不读 A、不执行任何匹配阶段。批次链、合并批次备注与备注掩码都调用
normalizer 中与 ``normalize_b`` 相同的函数对象（D11），不复制任何规则。
"""

from datetime import datetime

import numpy as np
import pandas as pd

import config
import loader
import normalizer
from errors import FatalDataError

# 预览工具只检查这 5 个标准字段对应的列（Req 16.9）
PREVIEW_FIELDS = ("province", "batch", "remark_batch", "remark_major", "remark_plan")

# 原始列之后追加的 13 列（Req 16.4），顺序固定
PREVIEW_APPEND_COLUMNS = (
    "B_ID", "标准省份", "原批次", "批次主体", "规范批次名", "层次", "提前标记", "段",
    "批序", "抽出批次备注", "合并批次备注", "备注标签", "批次核查结果",
)


def _cell(value):
    """None / 缺失 / 空串 / 空元组 -> ""；片段与标签元组用分隔符连接；其余原样。"""
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return config.REMARK_SEPARATOR.join(str(v) for v in value)
    if isinstance(value, float) and value != value:
        return ""
    return value


def _cells(series):
    return [_cell(v) for v in series.to_numpy(dtype=object)]


def read_preview_inputs():
    """读取流程（Req 16.1、16.7、16.10）：返回 (ts, a_table, prov_list, B_raw)。

    顺序：记录启动时刻 -> validate_config -> A 批次对照表 -> 各省批次清单（失败只警告）
    -> B 表（第一个工作表）-> 原始行数为 0 时抛 FatalDataError。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.validate_config()
    a_table = loader.read_a_batch_table(config.A_BATCH_TABLE)
    prov_list = loader.read_province_batch_list(
        config.PROVINCE_BATCH_LIST, config.PROVINCE_BATCH_SHEET)
    B_raw = loader.read_table(config.INPUT_B, config.SHEET_B)
    _require_rows(B_raw)
    return ts, a_table, prov_list, B_raw


def _require_rows(B_raw):
    if len(B_raw) == 0:
        msg = f"B 表没有数据行：{config.INPUT_B}"
        print(f"[错误] {msg}")
        raise FatalDataError(msg)


def build_preview_frame(B_raw, a_table, prov_list) -> pd.DataFrame:
    """纯内存：B 全部原始列（原名、原序、原值）之后追加 13 列。不过滤、不修改 B_raw。"""
    _require_rows(B_raw)
    original_cols = list(B_raw.columns)
    B = loader.assign_b_ids(B_raw)
    n = len(B)

    # 只检查 5 列；缺列警告并补空列（补出的列不属于原始列，不进入输出）
    subset = {f: config.B_COLUMNS[f] for f in PREVIEW_FIELDS}
    loader.check_columns(B, subset, "B")

    cols = config.B_COLUMNS
    raw_province = normalizer._get_col(B, cols["province"])
    raw_batch = normalizer._get_col(B, cols["batch"])
    raw_remark_batch = normalizer._get_col(B, cols["remark_batch"])
    raw_remark_major = normalizer._get_col(B, cols["remark_major"])
    raw_remark_plan = normalizer._get_col(B, cols["remark_plan"])

    legal = normalizer.build_province_legal_sets(a_table, prov_list)
    province = normalizer.vec_map(raw_province, normalizer.norm_province_name)

    chain = normalizer.bind_b_batch_chain(legal)
    (body, fragments, b_layer, early, segment, order, name_std, check,
     _mask_fragments, mask_body) = normalizer.vec_map_pair(province, raw_batch, chain, 10)
    merged, mask_merged = normalizer.vec_map_pair(
        raw_remark_batch, raw_batch, normalizer.b_merged_remark_chain, 2)

    sources = {
        "remark_major": normalizer._mask_column(raw_remark_major),
        "batch_remark_merged": mask_merged.to_numpy(dtype=object),
        "remark_plan": normalizer._mask_column(raw_remark_plan),
        "batch_body": mask_body.to_numpy(dtype=object),
        "batch_remark": _mask_fragments.to_numpy(dtype=object),
    }
    mask = normalizer._or_sources(sources, config.B_REMARK_SOURCES, n)
    tags = [tuple(normalizer.mask_to_tags(m)) for m in mask.tolist()]

    appended = {
        "B_ID": B["b_id"].tolist(),
        "标准省份": _cells(province),
        "原批次": _cells(raw_batch),
        "批次主体": _cells(body),
        "规范批次名": _cells(name_std),
        "层次": _cells(b_layer),
        "提前标记": _cells(early),
        "段": _cells(segment),
        "批序": _cells(order),
        "抽出批次备注": _cells(fragments),
        "合并批次备注": _cells(merged),
        "备注标签": [_cell(t) for t in tags],
        "批次核查结果": _cells(check),
    }

    out = B[original_cols].copy().reset_index(drop=True)
    for name in PREVIEW_APPEND_COLUMNS:
        out[name] = appended[name]
    return out
