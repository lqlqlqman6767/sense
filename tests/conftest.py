# -*- coding: utf-8 -*-
"""pytest 公共配置：sys.path、Hypothesis profile、真实数据防护。"""

import sys
from pathlib import Path

import pytest

ABC_DIR = Path(__file__).resolve().parent.parent
if str(ABC_DIR) not in sys.path:
    sys.path.insert(0, str(ABC_DIR))

try:
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "default",
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.load_profile("default")
except ImportError:  # hypothesis 未安装时跳过
    pass


@pytest.fixture(autouse=True)
def _guard_real_data(monkeypatch, tmp_path):
    """把 config 中的数据与输出路径指向 tmp_path 下不存在的位置。

    测试若误读真实数据会因文件不存在而失败，误写 ABC/output/ 也不会落到真实目录。
    需要文件的测试自行 monkeypatch 覆盖这些路径。
    """
    import config

    guard = tmp_path / "_no_real_data"
    monkeypatch.setattr(config, "INPUT_A", guard / "A.xlsx")
    monkeypatch.setattr(config, "INPUT_B", guard / "B.xlsx")
    monkeypatch.setattr(config, "OUTPUT", guard / "output" / "匹配结果.xlsx")
    monkeypatch.setattr(config, "KEYWORD_REPORT", guard / "output" / "备注关键词候选.csv")
    monkeypatch.setattr(config, "A_BATCH_TABLE", guard / "批次.csv")
    monkeypatch.setattr(config, "PROVINCE_BATCH_LIST", guard / "各省批次映射.xlsx")
    monkeypatch.setattr(config, "PREVIEW_OUTPUT_DIR", guard / "preview_output")
