# -*- coding: utf-8 -*-
"""合成测试夹具（Req 15.21）：A/B 行与表、A 批次对照表、各省批次清单及其文件写出辅助函数。

默认值：院校ID 双方相同；省份河北（A province_id=130000、B“河北”）；B 软删除标记 0、
当前有效标记 1、颗粒度“专业分”；其余匹配字段双方缺失；回填字段为非零数值。
覆盖参数既可用标准字段名（如 ``major_name``），也可用真实列名（如 ``special_name``）。
所有函数只使用合成数据，不读取 ``ABC/data/``。
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

_ABC_DIR = Path(__file__).resolve().parent.parent
if str(_ABC_DIR) not in sys.path:
    sys.path.insert(0, str(_ABC_DIR))

import config  # noqa: E402

DEFAULT_SCHOOL_ID = "1001"
DEFAULT_A_PROVINCE_ID = "130000"
DEFAULT_B_PROVINCE = "河北"

# 回填字段默认值：B 源列 -> 非零数值
DEFAULT_SCORES = {
    "最低分": 500,
    "最高分": 520,
    "平均分": 510,
    "最低分位次": 10000,
    "平均分位次": 9000,
    "最高分位次": 8000,
    "录取人数": 3,
}

A_BATCH_DF_COLUMNS = ("province_id", "year", "batch_name", "batch_type")


def _resolve_overrides(overrides, mapping, extra_columns=()):
    """把覆盖参数的键统一换成真实列名。未知键抛 KeyError。"""
    real_cols = set(mapping.values()) | set(extra_columns)
    out = {}
    for key, value in overrides.items():
        if key in mapping:
            out[mapping[key]] = value
        elif key in real_cols:
            out[key] = value
        else:
            raise KeyError(f"未知字段 {key!r}：既不是标准字段名也不是真实列名")
    return out


def make_a_row(**overrides) -> dict:
    """A 表一行，键为真实列名（config.A_COLUMNS 的值）。"""
    row = {col: None for col in config.A_COLUMNS.values()}
    row[config.A_COLUMNS["school_id"]] = DEFAULT_SCHOOL_ID
    row[config.A_COLUMNS["province"]] = DEFAULT_A_PROVINCE_ID
    row.update(_resolve_overrides(overrides, config.A_COLUMNS))
    return row


def make_b_row(**overrides) -> dict:
    """B 表一行，键为真实列名（config.B_COLUMNS 的值与 SCORE_COLUMNS 的源列）。"""
    row = {col: None for col in config.B_COLUMNS.values()}
    row[config.B_COLUMNS["school_id"]] = DEFAULT_SCHOOL_ID
    row[config.B_COLUMNS["province"]] = DEFAULT_B_PROVINCE
    row[config.B_COLUMNS["soft_delete"]] = 0
    row[config.B_COLUMNS["is_current"]] = 1
    row[config.B_COLUMNS["granularity"]] = config.GRANULARITY_MAJOR
    for src, _out in config.SCORE_COLUMNS:
        row[src] = DEFAULT_SCORES.get(src, 1)
    score_cols = [src for src, _ in config.SCORE_COLUMNS]
    row.update(_resolve_overrides(overrides, config.B_COLUMNS, score_cols))
    return row


def make_a_table(rows=None) -> pd.DataFrame:
    """由行（dict 列表，缺省键补默认值前的原样）组成 A 表；rows 中每项可为覆盖参数 dict。"""
    cols = list(config.A_COLUMNS.values())
    data = [make_a_row(**r) for r in (rows if rows is not None else [{}])]
    return pd.DataFrame(data, columns=cols, dtype=object)


def make_b_table(rows=None) -> pd.DataFrame:
    """由覆盖参数 dict 列表组成 B 表，列为全部真实列（含回填源列）。"""
    cols = list(config.B_COLUMNS.values()) + [s for s, _ in config.SCORE_COLUMNS]
    data = [make_b_row(**r) for r in (rows if rows is not None else [{}])]
    return pd.DataFrame(data, columns=cols, dtype=object)


# ------------------------------------------------------------
# A 批次对照表与各省批次清单
# ------------------------------------------------------------

def make_a_batch_df(entries=()) -> pd.DataFrame:
    """A 批次对照表 DataFrame。

    entries 每项为 (province_id, batch_type, batch_name) 或 (province_id, year, batch_name, batch_type)
    的 dict/tuple；dict 使用同名键，year 缺省 2026。
    """
    records = []
    for e in entries:
        if isinstance(e, dict):
            rec = {"province_id": e["province_id"], "year": e.get("year", 2026),
                   "batch_name": e["batch_name"], "batch_type": e["batch_type"]}
        elif len(e) == 3:
            pid, bt, name = e
            rec = {"province_id": pid, "year": 2026, "batch_name": name, "batch_type": bt}
        else:
            pid, year, name, bt = e
            rec = {"province_id": pid, "year": year, "batch_name": name, "batch_type": bt}
        records.append(rec)
    return pd.DataFrame(records, columns=list(A_BATCH_DF_COLUMNS), dtype=object)


def make_a_batch_table(entries=()):
    """构造 loader.ABatchTable：经写出 GBK csv 再用 loader.read_a_batch_table 读取，逻辑与真实一致。"""
    import loader

    with tempfile.TemporaryDirectory() as d:
        path = write_a_batch_csv(d, make_a_batch_df(entries))
        return loader.read_a_batch_table(path)


def batch_table(province_id=DEFAULT_A_PROVINCE_ID, batch_type="1", batch_name="本科批", extra=()):
    """常用简写：把 (province_id, batch_type) 映射到 batch_name，可附加其它条目。"""
    return make_a_batch_table([(province_id, batch_type, batch_name), *extra])


def make_province_batch_df(entries=()) -> pd.DataFrame:
    """各省批次清单 DataFrame，entries 为 (province_id, batch_name) 序列。"""
    return pd.DataFrame(list(entries), columns=list(config.PROVINCE_BATCH_COLUMNS), dtype=object)


# ------------------------------------------------------------
# 文件写出辅助
# ------------------------------------------------------------

def write_b_xlsx(directory, df=None, name="B.xlsx") -> Path:
    path = Path(directory) / name
    (make_b_table() if df is None else df).to_excel(path, index=False)
    return path


def write_a_xlsx(directory, df=None, name="A.xlsx") -> Path:
    path = Path(directory) / name
    (make_a_table() if df is None else df).to_excel(path, index=False)
    return path


def write_a_batch_csv(directory, df=None, name="批次.csv") -> Path:
    path = Path(directory) / name
    (make_a_batch_df() if df is None else df).to_csv(
        path, index=False, encoding=config.A_BATCH_TABLE_ENCODING)
    return path


def write_province_batch_xlsx(directory, df=None, name="各省批次映射.xlsx",
                              sheet=None) -> Path:
    path = Path(directory) / name
    (make_province_batch_df() if df is None else df).to_excel(
        path, index=False, sheet_name=sheet or config.PROVINCE_BATCH_SHEET)
    return path
