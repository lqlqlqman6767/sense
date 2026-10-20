# -*- coding: utf-8 -*-
"""备注标签提取与参考实现一致（属性测试）。"""

from contextlib import contextmanager

from hypothesis import example, given

import config
from normalizer import mask_to_tags, norm_text, or_masks, tag_mask
from tests.reference import naive_extract_tags
from tests.strategies import remark_sources, small_table_with_sources


def _normalize_sources(sources):
    out = []
    for s in sources:
        if s is None or isinstance(s, str):
            out.append(None if s is None else norm_text(s))
        else:
            out.append(tuple(norm_text(f) if f is not None else None for f in s))
    return out


def _actual_tags(sources):
    return mask_to_tags(or_masks(*(tag_mask(s) for s in sources)))


@contextmanager
def _patched_tables(tags, negations):
    old_tags, old_neg = config.REMARK_TAGS, config.NEGATION_PREFIXES
    config.REMARK_TAGS, config.NEGATION_PREFIXES = dict(tags), tuple(negations)
    try:
        yield
    finally:
        config.REMARK_TAGS, config.NEGATION_PREFIXES = old_tags, old_neg


# Feature: admission-plan-matching-refactor, Property 8: 备注标签提取与参考实现一致
# **Validates: Requirements 3.10, 3.11**
@given(remark_sources)
@example(["非定向"])
@example(["非定向", "定向"])
@example(["非中外合作办学"])
@example([("非定向", "定向")])
@example(["国家专项计划"])
@example(["非定向定向"])
def test_default_table_matches_reference(sources):
    sources = _normalize_sources(sources)
    expected = naive_extract_tags(sources, config.REMARK_TAGS, config.NEGATION_PREFIXES)
    assert _actual_tags(sources) == expected


# Feature: admission-plan-matching-refactor, Property 8: 备注标签提取与参考实现一致
# **Validates: Requirements 3.10, 3.11**
@given(small_table_with_sources())
def test_random_small_table_matches_reference(data):
    tags, negations, sources = data
    sources = _normalize_sources(sources)
    with _patched_tables(tags, negations):
        expected = naive_extract_tags(sources, tags, negations)
        assert _actual_tags(sources) == expected
