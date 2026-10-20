# -*- coding: utf-8 -*-
"""Data_Loader：读取 Excel、行号与 B_ID 生成。

行号约定：DataFrame 第 i 行（从 0 开始）对应 Excel 第 i+2 行（表头占第 1 行）。
后续任务（3.4/3.5）会在本文件追加 check_columns / filter_b 等函数。
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
from errors import InputError

_ENGINE_FALLBACK = "openpyxl"
_calamine_missing_warned = False


def excel_row(position: int) -> int:
    """DataFrame 位置（0 起）-> Excel 行号。"""
    return position + 2


def _resolve_engine(engine):
    """calamine 未安装时警告并回退 openpyxl。"""
    global _calamine_missing_warned
    if engine == "calamine":
        try:
            import python_calamine  # noqa: F401
        except ImportError:
            if not _calamine_missing_warned:
                print("[警告] 未安装 python-calamine，回退使用 openpyxl 读取 "
                      "(pip install python-calamine 可加速)")
                _calamine_missing_warned = True
            return _ENGINE_FALLBACK
    return engine


def read_table(path, sheet=0, engine=None) -> pd.DataFrame:
    """读取 Excel 工作表，所有单元格保留原始值（dtype=object，空单元格为 ""）。

    失败（文件不存在、工作表不存在、无法解析）统一抛 InputError。
    """
    if engine is None:
        engine = config.EXCEL_READ_ENGINE
    abs_path = Path(path).resolve()
    engine = _resolve_engine(engine)

    def fail(reason):
        return InputError(f"无法读取 Excel：{abs_path}，工作表={sheet!r}，原因：{reason}")

    if not abs_path.is_file():
        raise fail("文件不存在")
    try:
        df = pd.read_excel(
            abs_path, sheet_name=sheet, dtype=object, engine=engine,
            keep_default_na=False, na_values=[],
        )
    except Exception as e:  # noqa: BLE001 - 统一包装
        raise fail(f"{type(e).__name__}: {e}") from e

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    print(f"[读取] {abs_path.name} | 工作表={sheet!r} | 原始数据行数={len(df)}")
    return df


def assign_a_rows(A_raw: pd.DataFrame) -> pd.DataFrame:
    """为 A 增加 a_row（Excel 行号 = 位置 + 2）。返回新表。"""
    out = A_raw.copy()
    out["a_row"] = [excel_row(i) for i in range(len(out))]
    return out


def assign_b_ids(B_raw: pd.DataFrame) -> pd.DataFrame:
    """在任何过滤之前为 B 生成 b_id = "B_" + 8 位零填充 Excel 行号。返回新表。"""
    out = B_raw.copy()
    out["b_id"] = [f"B_{excel_row(i):08d}" for i in range(len(out))]
    return out


# ============================================================
# 缺列检查与 B 过滤（任务 3.4）
# ============================================================

import re
from dataclasses import dataclass, field

from errors import FatalDataError

_WHOLE_DOT_ZERO = re.compile(r"^[0-9]+\.0$")
_NOT_RUN = "未执行"
# 3 个特殊列的标准字段
_SPECIAL_B_FIELDS = ("granularity", "soft_delete", "is_current")


def check_columns(df: pd.DataFrame, mapping: dict, table: str) -> list:
    """检查 mapping 中的列是否存在，返回缺失的标准字段列表。

    - 表头与真实列名都去首尾空白后逐字符精确比较（df 的表头会被原地去空白）。
    - 缺列：逐条警告并补一列全空值（""）。
    - B 表：granularity 缺失抛 FatalDataError；soft_delete / is_current 缺失只警告，不补列。
    - B 表的完整映射（含全部 B_COLUMNS 键）额外检查 7 个回填源列；
      预览工具只传 5 列子集时不检查回填源列。
    """
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    present = set(df.columns)
    missing = []

    for std, real in mapping.items():
        real_s = real.strip()
        if table == "B" and std in _SPECIAL_B_FIELDS:
            if real_s in present:
                continue
            missing.append(std)
            if std == config.GRANULARITY_FIELD:
                raise FatalDataError(f"B 表缺少“数据颗粒度”列：{real_s}")
            print(f"[警告] B 表缺少列：{real_s}，跳过该项过滤")
            continue
        if real_s in present:
            continue
        print(f"[警告] {table} 表缺少列：标准字段={std} 真实列名={real_s}")
        df[real_s] = ""
        present.add(real_s)
        missing.append(std)

    if table == "B" and set(config.B_COLUMNS) <= set(mapping):
        for src, _out in config.SCORE_COLUMNS:
            s = src.strip()
            if s in present:
                continue
            print(f"[警告] B 表缺少列：标准字段=回填源列 真实列名={s}")
            df[s] = ""
            present.add(s)
            missing.append(s)
    return missing


def _norm_flag(x) -> str:
    """标记值/颗粒度比较：只做 strip 与整串 ``.0`` 去除。"""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    s = str(x).strip()
    if _WHOLE_DOT_ZERO.match(s):
        s = s[:-2]
    return s


@dataclass
class BFilterReport:
    raw_rows: int = 0
    kept_rows: int = 0
    # 条件名 -> 排除数；未执行的条件排除数为 0
    excluded: dict = field(default_factory=dict)
    # 条件名 -> 是否已执行
    executed: dict = field(default_factory=dict)


def _flag_series(B: pd.DataFrame, std: str) -> pd.Series:
    return B[config.B_COLUMNS[std].strip()].map(_norm_flag)


def filter_b(B_raw: pd.DataFrame):
    """按 软删除 → 当前有效 → 颗粒度 过滤 B，返回 (B_kept, BFilterReport)。"""
    B_raw.columns = [c.strip() if isinstance(c, str) else c for c in B_raw.columns]
    cols = set(B_raw.columns)
    n = len(B_raw)
    report = BFilterReport(raw_rows=n)
    alive = pd.Series(True, index=B_raw.index)

    conditions = [(std, keep) for std, keep in config.B_FLAG_FILTERS]
    conditions.append((config.GRANULARITY_FIELD, config.GRANULARITY_MAJOR))

    for std, keep in conditions:
        real = config.B_COLUMNS[std].strip()
        label = f"{real}={keep}"
        if real not in cols:
            report.excluded[label] = 0
            report.executed[label] = False
            continue
        passed = _flag_series(B_raw, std) == keep
        fail_first = alive & ~passed
        report.excluded[label] = int(fail_first.sum())
        report.executed[label] = True
        alive &= passed

    B_kept = B_raw[alive].copy()
    report.kept_rows = len(B_kept)

    parts = []
    for label, cnt in report.excluded.items():
        parts.append(f"{label}: {cnt if report.executed[label] else _NOT_RUN + '(0)'}")
    print(f"[B 过滤] 原始 {n} 行 | 排除 " + " | ".join(parts) + f" | 保留 {report.kept_rows} 行")

    assert sum(report.excluded.values()) + report.kept_rows == n, "B 过滤计数不守恒"
    if report.kept_rows == 0:
        raise FatalDataError("过滤后参与匹配的 B 记录为 0；各条件排除数：" + "；".join(parts))
    return B_kept, report


# 条件标准字段 -> 原因文案
_FILTER_REASON_LABELS = {
    "soft_delete": "软删除",
    "is_current": "当前无效",
    "granularity": "颗粒度不符",
}


def filter_reasons(B_raw: pd.DataFrame) -> pd.Series:
    """逐行计算未通过的过滤条件原因，列式向量化（按条件逐列计算 bool，不逐行遍历业务判定）。

    与 filter_b 条件一致（软删除 → 当前有效 → 颗粒度，Config.B_FLAG_FILTERS + 颗粒度）；
    一行可同时未通过多个条件，全部列出（不是只取第一个），用 config.REMARK_SEPARATOR 连接。
    缺列的条件视为“未执行”，不计入原因（与 filter_b 对缺列的处理一致）。
    返回与 B_raw 索引对齐的 Series[str]；全部通过的行为空字符串 ""。
    """
    cols = set(B_raw.columns)
    n = len(B_raw)
    conditions = [(std, keep) for std, keep in config.B_FLAG_FILTERS]
    conditions.append((config.GRANULARITY_FIELD, config.GRANULARITY_MAJOR))

    label_cols = {}
    for std, keep in conditions:
        real = config.B_COLUMNS[std].strip()
        if real not in cols:
            continue
        fail = _flag_series(B_raw, std) != keep
        label_cols[_FILTER_REASON_LABELS[std]] = fail

    sep = config.REMARK_SEPARATOR
    if not label_cols:
        return pd.Series([""] * n, index=B_raw.index, dtype=object)

    # 按条件顺序逐列拼接：命中则在已有文本后追加 "label"（非空时先接分隔符），全列向量化。
    result = pd.Series([""] * n, index=B_raw.index, dtype=object)
    for label, fail in label_cols.items():
        fail = fail.to_numpy()
        has_prev = result.to_numpy() != ""
        sep_or_empty = np.where(fail & has_prev, sep, "")
        label_or_empty = np.where(fail, label, "")
        result = result + pd.Series(sep_or_empty, index=B_raw.index) \
            + pd.Series(label_or_empty, index=B_raw.index)
    return result


def select_granularity(B: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """颗粒度选择扩展入口。

    默认只允许“专业分”；“专业组分/院校分”需在 config.GRANULARITY_EXTENSIONS 中启用，
    否则抛 FatalDataError。
    """
    if granularity != config.GRANULARITY_MAJOR and not config.GRANULARITY_EXTENSIONS.get(granularity, False):
        raise FatalDataError(f"颗粒度“{granularity}”未启用（config.GRANULARITY_EXTENSIONS）")
    real = config.B_COLUMNS[config.GRANULARITY_FIELD].strip()
    return B[B[real].map(_norm_flag) == granularity.strip()].copy()


# ============================================================
# A 批次对照表与各省批次清单（任务 3.5）
# ============================================================

@dataclass(frozen=True)
class ABatchTable:
    # (province_id, batch_type) -> 批次名称，组合只对应一个名称
    exact: dict
    # batch_type -> 批次名称，只放全表（不分省）只对应一个名称的编码
    by_type: dict
    # (province_id, batch_type) -> 多个不同名称的原值元组，按查不到处理
    ambiguous: dict


def read_a_batch_table(path) -> ABatchTable:
    """读取 A 批次对照表（GBK csv）。失败统一抛 InputError。"""
    from normalizer import norm_code, norm_text

    abs_path = Path(path).resolve()

    def fail(reason):
        return InputError(f"无法读取 A 批次对照表：{abs_path}，原因：{reason}")

    if not abs_path.is_file():
        raise fail("文件不存在")
    try:
        df = pd.read_csv(abs_path, encoding=config.A_BATCH_TABLE_ENCODING,
                         dtype=str, keep_default_na=False)
    except UnicodeDecodeError as e:
        raise fail(f"{config.A_BATCH_TABLE_ENCODING} 解码失败：{e}") from e
    except Exception as e:  # noqa: BLE001 - 统一包装
        raise fail(f"{type(e).__name__}: {e}") from e

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    missing = [c for c in config.A_BATCH_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise fail("缺少列：" + "、".join(missing))

    # (province_id, batch_type) -> {标准化名称: 首次出现的原值}
    groups = {}
    for pid_raw, name_raw, bt_raw in zip(df["province_id"], df["batch_name"], df["batch_type"]):
        pid, bt, name = norm_code(pid_raw), norm_code(bt_raw), norm_text(name_raw)
        if pid is None or bt is None or name is None:
            continue
        groups.setdefault((pid, bt), {}).setdefault(name, name_raw)

    exact, ambiguous = {}, {}
    type_names = {}
    for (pid, bt), names in groups.items():
        for name in names:
            type_names.setdefault(bt, set()).add(name)
        if len(names) == 1:
            exact[(pid, bt)] = next(iter(names))
        else:
            originals = tuple(names.values())
            ambiguous[(pid, bt)] = originals
            print(f"[警告] A 批次对照表 (province_id={pid}, batch_type={bt}) "
                  f"对应多个批次名称：{'、'.join(originals)}（原值），按查不到处理")
    by_type = {bt: next(iter(ns)) for bt, ns in type_names.items() if len(ns) == 1}

    print(f"[读取] {abs_path.name} | 精确组合={len(exact)} | 编码唯一={len(by_type)} "
          f"| 多名称组合={len(ambiguous)}")
    return ABatchTable(exact=exact, by_type=by_type, ambiguous=ambiguous)


def read_province_batch_list(path, sheet=None):
    """读取各省批次清单，返回含原值 province_id / batch_name 两列的表；不可用时警告并返回 None。"""
    if sheet is None:
        sheet = config.PROVINCE_BATCH_SHEET
    abs_path = Path(path).resolve()

    def unavailable(reason):
        print(f"[警告] 各省批次清单不可用：{abs_path}：{reason}；省合法批次集合只用 A 批次对照表计算")
        return None

    if not abs_path.is_file():
        return unavailable("文件不存在")
    try:
        df = pd.read_excel(
            abs_path, sheet_name=sheet, dtype=object,
            engine=_resolve_engine(config.EXCEL_READ_ENGINE),
            keep_default_na=False, na_values=[],
        )
    except Exception as e:  # noqa: BLE001 - 只警告
        return unavailable(f"{type(e).__name__}: {e}")

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    cols = list(config.PROVINCE_BATCH_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return unavailable("缺少列：" + "、".join(missing))
    return df[cols].copy()
