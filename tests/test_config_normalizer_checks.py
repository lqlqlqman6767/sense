# -*- coding: utf-8 -*-
"""validate_config() 中依赖 normalizer 的校验。Validates: Requirements 3.10, 3.16, 3.20"""
import pytest

import config
from errors import ConfigError


def test_default_passes():
    config.validate_config()


def test_alias_same_body_different_value(monkeypatch):
    bad = {**config.BATCH_ALIAS, ("*", "专科 普通批"): "本科批"}
    monkeypatch.setattr(config, "BATCH_ALIAS", bad)
    with pytest.raises(ConfigError, match="专科"):
        config.validate_config()


def test_alias_same_body_same_value_ok(monkeypatch):
    ok = {("*", "高职高专普通批"): "专科批", ("*", "专科普通批"): "专科批"}
    monkeypatch.setattr(config, "BATCH_ALIAS", ok)
    config.validate_config()


def test_alias_same_body_in_different_provinces_ok(monkeypatch):
    ok = {("四川", "X批"): "专科批", ("*", "X批"): "本科批"}
    monkeypatch.setattr(config, "BATCH_ALIAS", ok)
    config.validate_config()


def test_alias_value_not_idempotent(monkeypatch):
    bad = {("*", "某某批"): "随便写的名字"}
    monkeypatch.setattr(config, "BATCH_ALIAS", bad)
    with pytest.raises(ConfigError, match="某某批.*随便写的名字|随便写的名字"):
        config.validate_config()


def test_alias_value_checked_under_province_alias(monkeypatch):
    # "*" 的值 "本科批" 在四川下会被四川键改写为 "专科批"，不再等于自身
    bad = {("*", "甲批"): "本科批", ("四川", "本科批"): "专科批"}
    monkeypatch.setattr(config, "BATCH_ALIAS", bad)
    with pytest.raises(ConfigError, match="四川"):
        config.validate_config()


def test_empty_remark_keyword(monkeypatch):
    monkeypatch.setattr(config, "REMARK_TAGS", {"定向": ("定向", " ")})
    with pytest.raises(ConfigError, match="为空"):
        config.validate_config()


def test_empty_negation_prefix(monkeypatch):
    monkeypatch.setattr(config, "NEGATION_PREFIXES", ("非", "无"))  # "无" 是缺失值记号
    with pytest.raises(ConfigError, match="否定前缀"):
        config.validate_config()


def test_separator_appears_after_normalization(monkeypatch):
    # 全角竖线经 NFKC 变为 "|"
    monkeypatch.setattr(config, "REMARK_TAGS", {"定向": ("定向", "甲｜乙")})
    with pytest.raises(ConfigError, match="REMARK_SEPARATOR"):
        config.validate_config()
