# -*- coding: utf-8 -*-
"""规范批次名往返属性测试。"""

from hypothesis import given

import normalizer
from tests.strategies import batch_components


# Feature: admission-plan-matching-refactor, Property 29: 规范批次名往返
# **Validates: Requirements 3.19**
@given(batch_components())
def test_canonical_batch_name_roundtrip(c):
    name = normalizer.compose_batch_name(c)
    if all(v is None for v in c):
        assert name is None
        assert normalizer.extract_components(None) == (None, None, None, None)
        return
    assert name is not None
    body, _ = normalizer.clean_batch(name, None)
    assert normalizer.extract_components(body) == c
