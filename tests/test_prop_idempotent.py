# -*- coding: utf-8 -*-
"""标准化幂等属性测试（Req 3.13、15.25）。"""

# Feature: admission-plan-matching-refactor, Property 9: 标准化幂等

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import config
from normalizer import (
    canonical_batch_name,
    clean_batch,
    clean_batch_steps_1_4,
    compose_batch_name,
    extract_components,
    get_alias_table,
    layer_from_column,
    norm_code,
    norm_major,
    norm_major_group,
    norm_province_name,
    norm_subject_a,
    norm_subject_b,
    norm_text,
    tag_mask,
)
from tests.strategies import charset_text

_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

_PROVINCE = st.one_of(st.none(), st.sampled_from(sorted(config.PROVINCE_NAMES)))

# Req 15.25：Glossary 缺失值标记、批次关键字、备注关键词（独立于生成器，按配置计算）
_KEYWORDS = sorted(
    {t.lower() for t in config.MISSING_TOKENS if t}
    | {k.lower() for k in config.BATCH_KEYWORDS}
    | {w.lower() for ws in config.REMARK_TAGS.values() for w in ws}
)


def _has_keyword(x):
    low = x.lower()
    return any(k in low for k in _KEYWORDS)


@pytest.fixture(scope="module")
def sample_counter():
    """module 作用域计数：生成的样本数与含缺失值标记/批次关键字/备注关键词的样本数。"""
    return {"total": 0, "hits": 0}


def _record(counter, x):
    counter["total"] += 1
    if _has_keyword(x):
        counter["hits"] += 1


def _batch_body_has_normalize_term(x):
    """x 第一次清洗经步骤④得到的批次主体是否含 Config 批次名称归一表任一词条。"""
    body, _ = clean_batch_steps_1_4(x)
    if body is None:
        return False
    terms = [t for t in (norm_text(t0) for t0 in config.BATCH_NAME_NORMALIZE) if t]
    return any(t in body for t in terms)


# ---------------------------------------------------------------
# 前提与固定示例
# ---------------------------------------------------------------

def test_default_alias_keys_are_not_canonical_names():
    """Req 3.13 前提：按省批次别名表的每个键清洗(①–④)后合成都不等于其自身。"""
    for (_province, text), _value in config.BATCH_ALIAS.items():
        body, _ = clean_batch_steps_1_4(text)
        assert body is not None
        assert compose_batch_name(extract_components(body)) != body, text


def test_province_abbreviations_fixed_points():
    for name in config.PROVINCE_NAMES:
        assert norm_province_name(name) == name


def test_subject_targets_fixed_points():
    for target in ("物理类", "历史类", "综合"):
        assert norm_subject_a(target) == target


def test_layer_values_fixed_points():
    assert layer_from_column("本科") == "本科"
    assert layer_from_column("专科") == "专科"


@pytest.mark.parametrize("raw", ["12.0", "１２．０", " 007.0 ", "1.0.0", "0.0", "12.50", "abc.0"])
def test_norm_code_dot_zero_examples(raw):
    once = norm_code(raw)
    assert norm_code(once) == once


# ---------------------------------------------------------------
# Property 9：格式标准化与逐字段标准化
# ---------------------------------------------------------------

@_SETTINGS
@given(x=charset_text, province=_PROVINCE)
def test_field_normalizers_idempotent(sample_counter, x, province):
    _record(sample_counter, x)

    for f in (norm_text, norm_code, norm_major, norm_province_name,
              norm_subject_a, layer_from_column, norm_major_group):
        once = f(x)
        assert f(once) == once, (f.__name__, x, once)

    once = norm_subject_b(x, province)
    assert norm_subject_b(once, province) == once, ("norm_subject_b", x, province, once)


@_SETTINGS
@given(x=charset_text)
def test_tag_mask_after_format_normalization_idempotent(sample_counter, x):
    _record(sample_counter, x)

    once = norm_text(x)
    twice = norm_text(once)
    assert tag_mask(once) == tag_mask(twice), (x, once)

    # 片段来源：批次清洗得到的片段再经一次 norm_text 后提取，与直接提取相同
    _body, fragments = clean_batch_steps_1_4(x)
    refragmented = tuple(f for f in (norm_text(f0) for f0 in fragments) if f is not None)
    assert tag_mask(fragments) == tag_mask(refragmented), (x, fragments)


# ---------------------------------------------------------------
# Property 9：批次清洗与按省别名替换
# ---------------------------------------------------------------

@_SETTINGS
@given(x=charset_text, province=_PROVINCE)
def test_batch_clean_idempotent(sample_counter, x, province):
    _record(sample_counter, x)
    alias = get_alias_table()

    # 规范批次名 n：再执行清洗、成分抽取、合成仍得到 n
    n = canonical_batch_name(x, province, alias)
    if n is not None:
        body2, _ = clean_batch(n, province, alias)
        assert compose_batch_name(extract_components(body2)) == n, (x, province, n)

    # 批次主体 b：前提为第一次清洗经步骤④得到的主体不含归一表词条
    assume(not _batch_body_has_normalize_term(x))
    b, _ = clean_batch(x, province, alias)
    if b is not None:
        b2, _ = clean_batch(b, province, alias)
        assert b2 == b, (x, province, b, b2)


# ---------------------------------------------------------------
# Req 15.25：100 组样本中至少 20 组含缺失值标记、批次关键字或备注关键词
# ---------------------------------------------------------------

def test_keyword_coverage_of_samples(sample_counter):
    @settings(max_examples=100, deadline=None, database=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    @given(x=charset_text)
    def run(x):
        assert 0 <= len(x) <= 50
        _record(sample_counter, x)

    run()
    total, hits = sample_counter["total"], sample_counter["hits"]
    assert total >= 100
    assert hits * 5 >= total, f"含关键词样本 {hits}/{total}，不足 20%"
