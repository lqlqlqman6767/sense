# -*- coding: utf-8 -*-
"""备注不跨来源与片段，按标签否定（属性测试）。

小表的关键词只含“甲乙丙丁戊”，否定前缀只含“非否”，填充字符取“的班”，
三组字符互不相交，所以填充与前缀不会凭空造出新的关键词命中。
"""

from contextlib import contextmanager

from hypothesis import assume, example, given
from hypothesis import strategies as st

import config
from normalizer import mask_to_tags, merge_batch_remark, norm_text, or_masks, tag_mask
from tests.reference import naive_extract_tags
from tests.strategies import remark_sources_for, small_remark_tables

_KW_ALPHABET = "甲乙丙丁戊"
_NEG_ALPHABET = "非否"
_FILL = "的班"


@contextmanager
def _patched_tables(tags, negations):
    old_tags, old_neg = config.REMARK_TAGS, config.NEGATION_PREFIXES
    config.REMARK_TAGS, config.NEGATION_PREFIXES = dict(tags), tuple(negations)
    try:
        yield
    finally:
        config.REMARK_TAGS, config.NEGATION_PREFIXES = old_tags, old_neg


def _tags_of(sources):
    """各来源（None / 文本 / 片段元组）独立提取后按位或，展开为标签列表。"""
    return mask_to_tags(or_masks(*(tag_mask(s) for s in sources)))


_pad = st.text(alphabet=_FILL, min_size=0, max_size=3)

# 至少含一个长度 >= 2 的关键词的小表
_tables_with_long_keyword = small_remark_tables().filter(
    lambda t: any(len(w) >= 2 for ws in t[0].values() for w in ws))
# 至少含一个否定前缀的小表
_tables_with_negation = small_remark_tables().filter(lambda t: len(t[1]) >= 1)


# ---------------------------------------------------------------------------
# 1. 关键词被拆开放在相邻来源/片段的末尾与开头：不提取该标签
# ---------------------------------------------------------------------------
# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.10, 3.22**
@given(_tables_with_long_keyword, st.data())
def test_split_keyword_never_joins(table, data):
    tags, negations = table
    pairs = [(t, w) for t, ws in tags.items() for w in ws if len(w) >= 2]
    tag, keyword = data.draw(st.sampled_from(pairs))
    cut = data.draw(st.integers(min_value=1, max_value=len(keyword) - 1))
    left = data.draw(_pad) + keyword[:cut]
    right = keyword[cut:] + data.draw(_pad)
    # 其余内容不含该标签的任何关键词
    assume(not any(norm_text(w) in left or norm_text(w) in right for w in tags[tag]))

    with _patched_tables(tags, negations):
        variants = {
            "两个来源": [left, right],
            "两个来源（片段元组）": [(left,), (right,)],
            "同一来源的两个片段": [(left, right)],
            "合并批次备注": [merge_batch_remark(left, (right,))],
            "备注分隔符拼接的文本": [left + config.REMARK_SEPARATOR + right],
        }
        for name, sources in variants.items():
            assert tag not in _tags_of(sources), name


# ---------------------------------------------------------------------------
# 2. 否定前缀与关键词分处两个来源/片段：不构成否定；同一片段内相邻才否定
# ---------------------------------------------------------------------------
# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.10, 3.11, 3.22**
@given(_tables_with_negation, st.data())
def test_negation_prefix_does_not_cross_boundary(table, data):
    tags, negations = table
    tag = data.draw(st.sampled_from(sorted(tags)))
    keyword = data.draw(st.sampled_from(tags[tag]))
    prefix = data.draw(st.sampled_from(negations))
    left = data.draw(_pad) + prefix
    right = keyword + data.draw(_pad)

    with _patched_tables(tags, negations):
        variants = {
            "两个来源": [left, right],
            "两个来源（片段元组）": [(left,), (right,)],
            "同一来源的两个片段": [(left, right)],
            "合并批次备注": [merge_batch_remark(left, (right,))],
            "备注分隔符拼接的文本": [left + config.REMARK_SEPARATOR + right],
        }
        for name, sources in variants.items():
            assert tag in _tags_of(sources), name
        # 对照：前缀与关键词在同一片段内紧邻时被否定
        assert tag not in _tags_of([left + right])
        assert tag not in _tags_of([(left + right,)])


# ---------------------------------------------------------------------------
# 3. 按标签否定：存在一次未被否定的命中就提取，全部被否定则不提取
# ---------------------------------------------------------------------------
@st.composite
def _tables_with_occurrence_plan(draw):
    """同长度（2 个字）且互不相同的关键词，保证无子串关系；返回 (tags, negations, sources, expected)。

    每个出现前面都有“的”，被否定的出现在“的”之后紧邻某个前缀，
    因此出现的有效性完全由是否带前缀决定。
    """
    words = draw(st.lists(
        st.text(alphabet=_KW_ALPHABET, min_size=2, max_size=2),
        min_size=2, max_size=6, unique=True))
    n_tags = draw(st.integers(min_value=1, max_value=min(3, len(words))))
    owner = [draw(st.integers(min_value=0, max_value=n_tags - 1)) for _ in words]
    tags = {}
    for i in range(n_tags):
        tags["标签%d" % i] = tuple(w for w, o in zip(words, owner) if o == i)
    # 保证每个标签至少有一个关键词
    tags = {t: ws for t, ws in tags.items() if ws}
    word_tag = {w: t for t, ws in tags.items() for w in ws}
    negations = tuple(draw(st.lists(
        st.text(alphabet=_NEG_ALPHABET, min_size=1, max_size=2),
        min_size=1, max_size=2, unique=True)))

    item = st.tuples(st.sampled_from(sorted(word_tag)), st.booleans(), st.sampled_from(negations))
    fragment = st.lists(item, min_size=1, max_size=3)
    source = st.lists(fragment, min_size=1, max_size=2)
    plan = draw(st.lists(source, min_size=0, max_size=4))

    expected = set()
    sources = []
    for src in plan:
        rendered = []
        for frag in src:
            text = ""
            for word, negated, prefix in frag:
                text += "的" + (prefix + word if negated else word)
                if not negated:
                    expected.add(word_tag[word])
            rendered.append(text)
        if len(rendered) == 1 and draw(st.booleans()):
            sources.append(rendered[0])
        else:
            sources.append(tuple(rendered))
    return tags, negations, sources, [t for t in tags if t in expected]


# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.10, 3.11**
@given(_tables_with_occurrence_plan())
def test_tag_extracted_iff_some_occurrence_not_negated(case):
    tags, negations, sources, expected = case
    with _patched_tables(tags, negations):
        assert _tags_of(sources) == expected
        assert expected == naive_extract_tags(sources, tags, negations)


# ---------------------------------------------------------------------------
# 4. 被最左最长匹配覆盖的较短关键词不单独计为出现
# ---------------------------------------------------------------------------
@st.composite
def _covered_keyword_case(draw):
    short = draw(st.text(alphabet="甲乙", min_size=1, max_size=2))
    head = draw(st.text(alphabet="丙", min_size=1, max_size=2))
    tail = draw(st.text(alphabet="丁", min_size=1, max_size=2))
    prefix = draw(st.text(alphabet=_NEG_ALPHABET, min_size=1, max_size=2))
    return short, head + short + tail, prefix


# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.11**
@given(_covered_keyword_case(), _pad, _pad)
def test_covered_short_keyword_not_counted(case, pad_left, pad_right):
    short, long_kw, prefix = case
    tags = {"短": (short,), "长": (long_kw,)}
    with _patched_tables(tags, (prefix,)):
        # 长关键词整体被否定：短关键词被它覆盖，不单独计为出现
        assert _tags_of([pad_left + prefix + long_kw + pad_right]) == []
        # 长关键词未被否定：只有长标签
        assert _tags_of([pad_left + long_kw + pad_right]) == ["长"]
        # 单独出现的短关键词仍然计入
        assert _tags_of([pad_left + short + pad_right]) == ["短"]
        # 长关键词被否定，另一处短关键词独立出现：提取短标签
        assert _tags_of([prefix + long_kw + "的" + short]) == ["短"]


# ---------------------------------------------------------------------------
# 5. B 的合并批次备注：原“批次备注”整体一个片段在前，批次备注片段在后，各自独立匹配
# ---------------------------------------------------------------------------
@st.composite
def _merge_case(draw):
    tags, negations = draw(small_remark_tables())
    keywords = sorted({w for ws in tags.values() for w in ws})
    sources = draw(remark_sources_for(keywords, negations, _KW_ALPHABET + _NEG_ALPHABET))
    raw = None
    if sources and (sources[0] is None or isinstance(sources[0], str)):
        raw = sources[0]
        sources = sources[1:]
    fragments = []
    for s in sources:
        if isinstance(s, str):
            fragments.append(s)
        elif s is not None:
            fragments.extend(s)
    return tags, negations, raw, tuple(fragments)


# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.10, 3.22**
@given(_merge_case())
def test_merged_batch_remark_keeps_fragment_boundaries(case):
    tags, negations, raw, fragments = case
    merged = merge_batch_remark(raw, fragments)

    # 原“批次备注”整体在前，之后是非空的批次备注片段
    expected_parts = []
    if raw is not None and norm_text(raw):
        expected_parts.append(norm_text(raw))
    expected_parts.extend(f for f in fragments if f)
    assert merged == tuple(expected_parts)

    with _patched_tables(tags, negations):
        actual = mask_to_tags(tag_mask(merged))
        reference = naive_extract_tags([(p,) for p in expected_parts], tags, negations)
        assert actual == reference


# ---------------------------------------------------------------------------
# 6. Req 3.11 的 3 个例子（默认标签表）；合并批次备注示例（Req 3.22）
# ---------------------------------------------------------------------------
# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.10, 3.11, 3.22**
def test_req_3_11_examples_with_default_table():
    def extract(*sources):
        return mask_to_tags(or_masks(*(tag_mask(norm_text(s)) for s in sources)))

    assert "定向" not in extract("非定向")
    assert "定向" in extract("非定向", "定向")
    assert "中外合作" not in extract("非中外合作办学")


# Feature: admission-plan-matching-refactor, Property 33: 备注不跨来源与片段，按标签否定
# **Validates: Requirements 3.22**
def test_merged_batch_remark_example_from_req_3_22():
    merged = merge_batch_remark("中外合作", ("物理",))
    assert merged == ("中外合作", "物理")
    assert mask_to_tags(tag_mask(merged)) == ["中外合作"]
    # 原“批次备注”末尾的前缀不影响批次备注片段开头的关键词
    assert "定向" in mask_to_tags(tag_mask(merge_batch_remark("非", ("定向",))))
    # 拆开的关键词不会跨来源拼接
    assert "中外合作" not in mask_to_tags(tag_mask(merge_batch_remark("中外", ("合作",))))
