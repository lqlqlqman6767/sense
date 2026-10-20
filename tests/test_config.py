# -*- coding: utf-8 -*-
"""config.py 单元测试。Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 3.16"""

import json
import subprocess
import sys

import pytest

import config
from errors import ConfigError


def test_validate_config_passes_on_default():
    config.validate_config()


def test_paths_independent_of_cwd(tmp_path):
    code = "import config; print(config.BASE_DIR); print(config.INPUT_A); print(config.OUTPUT)"
    outs = []
    for cwd in (config.BASE_DIR, tmp_path):
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=str(cwd), capture_output=True,
            text=True, encoding="utf-8",
            env={**__import__("os").environ, "PYTHONPATH": str(config.BASE_DIR),
                 "PYTHONIOENCODING": "utf-8"},
        )
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]
    assert str(config.BASE_DIR) in outs[0]


def test_resolve_path_relative_and_absolute(tmp_path):
    assert config.resolve_path("data/x.xlsx") == config.BASE_DIR / "data/x.xlsx"
    assert config.resolve_path(tmp_path) == tmp_path


def test_column_maps():
    assert len(config.A_COLUMNS) == 11
    assert len(config.B_COLUMNS) == 15
    assert len(set(config.A_COLUMNS.values())) == len(config.A_COLUMNS)
    assert len(set(config.B_COLUMNS.values())) == len(config.B_COLUMNS)


def test_score_columns_distinct():
    assert len(config.SCORE_COLUMNS) == 7
    outs = [o for _, o in config.SCORE_COLUMNS]
    srcs = [s for s, _ in config.SCORE_COLUMNS]
    assert len(set(outs)) == 7
    assert len(set(srcs)) == 7
    assert "min_score" not in outs and "min_ranks" not in outs


def test_reason_codes_and_strip_tokens():
    assert set(config.REASON_CODES) == {"U01", "U02", "U03", "U04", "U05"}
    assert tuple(config.MAJOR_GROUP_STRIP_TOKENS) == ("第", "专业组", "组")


def test_hard_conflict_defaults():
    assert tuple(config.CONFLICT_FIELDS) == ("layer", "subject", "major_group", "code")
    assert config.HARD_CONFLICT == {
        "layer": False,
        "subject": False,
        "major_group": False,
        "code": False,
    }
    for tag in ("层次不一致", "选科不一致", "专业组不一致", "专业代码不一致"):
        assert tag in config.RISK_TAGS
    assert config.RISK_BY_STATE[("major_group", "CONFLICT")] == "专业组不一致"
    assert config.RISK_BY_STATE[("code", "CONFLICT")] == "专业代码不一致"


@pytest.mark.parametrize("mutate", [
    lambda m: m.setattr(config, "HARD_CONFLICT", {
        "layer": True, "subject": True, "major_group": False}),
    lambda m: m.setattr(config, "HARD_CONFLICT", {
        **config.HARD_CONFLICT, "extra": True}),
    lambda m: m.setattr(config, "HARD_CONFLICT", {
        **config.HARD_CONFLICT, "major_group": "False"}),
    lambda m: m.setattr(config, "CONFLICT_FIELDS", ("layer", "subject", "major_group")),
])
def test_hard_conflict_config_errors(monkeypatch, mutate):
    mutate(monkeypatch)
    with pytest.raises(ConfigError) as ei:
        config.validate_config()
    assert ei.value.exit_code == 2


@pytest.mark.parametrize("mutate", [
    lambda m: m.setattr(config, "REASON_CODES", {"U01": "a", "U02": "b", "U03": "c", "U04": "d"}),
    lambda m: m.setattr(config, "REASON_CODES", {**config.REASON_CODES, "U06": "x"}),
    lambda m: m.setattr(config, "RISK_TAGS", config.RISK_TAGS + (config.RISK_TAGS[0],)),
    lambda m: m.setattr(config, "REMARK_SEPARATOR", "-"),
    lambda m: m.setattr(config, "REMARK_SEPARATOR", "("),
    lambda m: m.setattr(config, "REMARK_SEPARATOR", ""),
    lambda m: m.setattr(config, "REMARK_TAGS", {"标签": ("含|分隔符",)}),
    lambda m: m.setattr(config, "SUBJECT_MAP", {"物理": "物理类"}),
])
def test_validate_config_errors(monkeypatch, mutate):
    mutate(monkeypatch)
    with pytest.raises(ConfigError) as ei:
        config.validate_config()
    assert ei.value.exit_code == 2


def test_batch_alias_value_not_fixed_point_raises_config_error(monkeypatch):
    # 别名值 "本科二批X" 经清洗、抽取、合成后变为 "本科二批X段"，不等于自身
    monkeypatch.setattr(config, "BATCH_ALIAS", {("*", "某某批"): "本科二批X"})
    with pytest.raises(ConfigError, match="值不自洽") as ei:
        config.validate_config()
    assert ei.value.exit_code == 2


def test_batch_alias_same_province_same_body_conflict_raises_config_error(monkeypatch):
    # 同省两个键清洗后同为 "某批"（内部空白被格式标准化删除），值却不同
    monkeypatch.setattr(config, "BATCH_ALIAS", {
        ("四川", "某批"): "本科批",
        ("四川", "某 批"): "专科批",
    })
    with pytest.raises(ConfigError, match="冲突") as ei:
        config.validate_config()
    assert ei.value.exit_code == 2


def test_load_remark_tags_from_tmp_json(tmp_path):
    p = tmp_path / "tags.json"
    p.write_text(json.dumps(
        {"tags": {"定向": ["定向", "定向培养"]}, "negation_prefixes": ["非"]},
        ensure_ascii=False), encoding="utf-8")
    tags, prefixes = config.load_remark_tags(p)
    assert tags == {"定向": ("定向", "定向培养")}
    assert prefixes == ("非",)

    p.write_text(json.dumps(
        {"tags": {"定向": ["新词"]}, "negation_prefixes": []}, ensure_ascii=False),
        encoding="utf-8")
    tags, prefixes = config.load_remark_tags(p)
    assert tags == {"定向": ("新词",)}
    assert prefixes == ()
