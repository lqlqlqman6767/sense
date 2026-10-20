# -*- coding: utf-8 -*-

"""
B表索引模块（Index_Builder）

只建 (院校ID, 省份, 专业名称主体) 基础索引：
- 必需字段任一缺失的 B 行不入索引，只计数（Req 4.2）
- 不去重、不截断、不建专业代码/选科/专业组/备注索引（Req 4.1）
- 专业名称主体 = 去掉括号内容后的名称；括号内容作为备注关键词在 resolver 中比较
- 全部为列式向量化操作，不逐行遍历
"""

from dataclasses import dataclass, field

import pandas as pd

INDEX_COLUMNS = ["school_id", "province", "major_base", "b_pos", "b_id"]
_KEY_COLUMNS = ["school_id", "province", "major_base"]


@dataclass(frozen=True)
class BIndex:
    """B 基础键表。

    frame:    列 [school_id, province, major_base, b_pos, b_id]，按 b_id 升序
    excluded: 必需字段缺失未入索引的 B 行数
    """

    frame: pd.DataFrame
    excluded: int
    _lookup_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def lookup(self, key):
        """字典视图：key=(school_id, province, major_base) -> b_id 列表（升序）。

        首次调用时由 groupby 构建，仅供测试与诊断。
        """
        if "map" not in self._lookup_cache:
            mapping = {}
            if len(self.frame) > 0:
                grouped = self.frame.groupby(_KEY_COLUMNS, sort=False)["b_id"]
                mapping = {k: list(v) for k, v in grouped}
            self._lookup_cache["map"] = mapping
        return list(self._lookup_cache["map"].get(tuple(key), []))


def build_index(B_std):
    """由 B_std 构建基础索引，返回 BIndex，并打印未入索引的 B 行数。"""
    keep = B_std[_KEY_COLUMNS].notna().all(axis=1)
    frame = B_std.loc[keep, INDEX_COLUMNS]
    frame = frame.sort_values("b_id", kind="stable").reset_index(drop=True)
    excluded = int(len(B_std) - len(frame))
    print(f"B表未入索引行数（院校/省份/专业名称缺失）：{excluded}")
    return BIndex(frame=frame, excluded=excluded)
