# -*- coding: utf-8 -*-
"""tag_mask / mask_to_tags 示例测试（Req 3.10、3.11、3.16）。"""

import pytest

import config
import normalizer
from normalizer import mask_to_tags, or_masks, tag_mask

TAG_ORDER = list(config.REMARK_TAGS)


def bit(tag):
    return 1 << TAG_ORDER.index(tag)


@pytest.mark.parametrize("value", [None, (), "", ("",), ("", "")])
def test_empty_inputs_give_zero(value):
    assert tag_mask(value) == 0


def test_negated_keyword_not_extracted():
    assert tag_mask("非定向") == 0


def test_negation_is_per_fragment():
    assert tag_mask(("非定向", "定向")) == bit("定向")


def test_negation_prefix_does_not_cross_fragments():
    # 前一片段以“非”结尾，不影响后一片段的“定向”
    assert tag_mask(("非", "定向")) == bit("定向")


def test_longer_keyword_covers_shorter_one():
    # “合作办学”被“中外合作办学”覆盖，不单独计为出现，因此仍被否定
    assert tag_mask("非中外合作办学") == 0


@pytest.mark.parametrize("text", ["中外合作", "中外合作办学", "合作办学"])
def test_cooperation_keywords(text):
    assert tag_mask(text) == bit("中外合作")


def test_split_keyword_across_fragments_not_matched():
    assert tag_mask(("中外", "合作")) == 0


def test_a_batch_remark_mongolian_taught():
    assert mask_to_tags(tag_mask(("蒙授",))) == ["民族类"]


def test_b_batch_body_local_special_plan():
    assert mask_to_tags(tag_mask("本科批地方专项计划")) == ["专项"]


def test_longest_keyword_counts_once_and_multiple_tags():
    text = "国家专项计划中外合作办学"
    assert tag_mask(text) == bit("专项") | bit("中外合作")


def test_occurrence_without_negation_wins():
    assert tag_mask("非定向,定向") == bit("定向")


def test_mask_to_tags_follows_config_order_and_or_masks():
    mask = or_masks(tag_mask("综合评价"), tag_mask("定向"), 0)
    assert mask_to_tags(mask) == [t for t in TAG_ORDER if t in ("定向", "综合评价")]
    assert mask_to_tags(0) == []


def test_mask_supports_many_tags():
    full = (1 << len(TAG_ORDER)) - 1
    assert full >= 0
    assert mask_to_tags(full) == list(TAG_ORDER)


def test_cache_rebuilds_when_config_changes(monkeypatch):
    monkeypatch.setattr(config, "REMARK_TAGS", {"甲": ("ＡＢＣ",), "乙": ("xy",)})
    monkeypatch.setattr(config, "NEGATION_PREFIXES", ("不",))
    # 关键词经 norm_text（全角 -> 半角）
    assert tag_mask("ABC") == 1
    assert tag_mask("不ABC") == 0
    assert tag_mask("非ABC") == 1
    assert mask_to_tags(tag_mask("xy|ABC".replace("|", ""))) == ["甲", "乙"]
    monkeypatch.setattr(config, "NEGATION_PREFIXES", ("非",))
    assert tag_mask("非ABC") == 0
    assert tag_mask("不ABC") == 1


def test_empty_tag_table_gives_zero(monkeypatch):
    monkeypatch.setattr(config, "REMARK_TAGS", {})
    assert tag_mask("定向") == 0
    assert normalizer.mask_to_tags(5) == []
