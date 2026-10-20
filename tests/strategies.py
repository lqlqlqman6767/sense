# -*- coding: utf-8 -*-
"""Hypothesis 生成器（各测试文件共用，只追加不改动已有内容）。"""

from hypothesis import strategies as st

# --- messy_text：Property 3、4 ---
_MESSY_ALPHABET = (
    "abcXYZ019"
    "ＡＢＣａｂｃ０１２９"
    "法学中外合作办学专业"
    " \t\n\r\u3000\u00a0"
    "()（）[]-/.,"
)
_MISSING_NOISE = ["", "nan", "NaN", "None", "NULL", "null", "无", "未知", "不详",
                  "-", "--", "/", "ＮＯＮＥ", " none ", "\u3000NULL\u3000", "－", "／"]

messy_text = st.one_of(
    st.text(alphabet=_MESSY_ALPHABET, min_size=0, max_size=50),
    st.sampled_from(_MISSING_NOISE),
    st.builds(
        lambda a, m, b: (a + m + b)[:50],
        st.text(alphabet=" \t\n\u3000\u00a0", max_size=3),
        st.sampled_from(_MISSING_NOISE),
        st.text(alphabet=" \t\n\u3000\u00a0", max_size=3),
    ),
)

# --- batch_components：Property 29（Req 3.19 取值域；层次/段/批序任一非空时提前标记非空）---
_BATCH_SEGMENTS = st.one_of(
    st.none(),
    st.sampled_from([chr(c) for c in range(ord("A"), ord("Z") + 1)]),
    st.from_regex(r"[0-9]{1,3}", fullmatch=True),
    st.sampled_from(list("一二三四五六七八九十")),
)


@st.composite
def batch_components(draw):
    layer = draw(st.sampled_from(["本科", "专科", None]))
    early = draw(st.sampled_from(["是", "否", None]))
    segment = draw(_BATCH_SEGMENTS)
    order = draw(st.sampled_from(["一", "二", "三", None]))
    if early is None and (layer is not None or segment is not None or order is not None):
        early = draw(st.sampled_from(["是", "否"]))
    return (layer, early, segment, order)

# --- province_subject：Property 31（省份:综合省份 / 非综合省份 / None ≈ 1:1:0.2；选科取表内键、表外值、缺失记号、杂乱文本）---
import config as _config_ps  # noqa: E402

_COMPREHENSIVE_PROVINCE_LIST = sorted(_config_ps.COMPREHENSIVE_PROVINCES)
_NON_COMPREHENSIVE_PROVINCE_LIST = sorted(
    _config_ps.PROVINCE_NAMES - _config_ps.COMPREHENSIVE_PROVINCES)
_SUBJECT_OUT_OF_TABLE = ["物化", "物理/历史", "化生"]
_SUBJECT_CHARSET = (
    "".join(chr(c) for c in range(0x20, 0x7F))
    + "".join(chr(c) for c in range(0xFF01, 0xFF5F))
    + "物理历史综合不限理文科类"
    + "（）\u2460\u2461\u2473 \u3000\t\n\u00a0"
)

_province_for_subject = st.one_of(
    st.sampled_from(_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_NON_COMPREHENSIVE_PROVINCE_LIST),
    st.just(None),
    # 权重 1:1:0.2 -> 5:5:1
    st.sampled_from(_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_NON_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_NON_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_NON_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_COMPREHENSIVE_PROVINCE_LIST),
    st.sampled_from(_NON_COMPREHENSIVE_PROVINCE_LIST),
)

_subject_text = st.one_of(
    st.sampled_from(sorted(_config_ps.SUBJECT_MAP)),
    st.sampled_from(_SUBJECT_OUT_OF_TABLE),
    st.sampled_from(_MISSING_NOISE),
    st.text(alphabet=_SUBJECT_CHARSET, min_size=0, max_size=50),
    # 表内键前后夹空白，覆盖格式标准化后再查表
    st.builds(
        lambda a, k, b: a + k + b,
        st.text(alphabet=" \t\n\u3000\u00a0", max_size=3),
        st.sampled_from(sorted(_config_ps.SUBJECT_MAP)),
        st.text(alphabet=" \t\n\u3000\u00a0", max_size=3),
    ),
)

province_subject = st.tuples(_province_for_subject, _subject_text)


# --- remark_sources：Property 8、33（备注来源列表；每个来源为 None / 文本 / 片段元组）---
import config as _config_rs  # noqa: E402

_REMARK_FILL_DEFAULT = "的计划班男女专业招生类型限制说明地区"  # 不含分隔符 "|"，不含缺失值记号


def remark_sources_for(keywords, prefixes, fill):
    """用给定关键词、否定前缀与填充字符生成来源列表：0–6 个来源，每个 0–3 个片段。"""
    keywords = list(keywords)
    prefixes = list(prefixes)
    pieces = [st.text(alphabet=fill, min_size=1, max_size=3)]
    if keywords:
        pieces.append(st.sampled_from(keywords))
    if prefixes:
        pieces.append(st.sampled_from(prefixes))
    fragment = st.lists(st.one_of(*pieces), min_size=0, max_size=4).map("".join)
    source = st.one_of(
        st.none(),
        fragment,
        st.lists(fragment, min_size=0, max_size=3).map(tuple),
    )
    return st.lists(source, min_size=0, max_size=6)


def _default_remark_keywords():
    return sorted({w for ws in _config_rs.REMARK_TAGS.values() for w in ws})


remark_sources = remark_sources_for(
    _default_remark_keywords(), list(_config_rs.NEGATION_PREFIXES), _REMARK_FILL_DEFAULT)

_SMALL_TABLE_ALPHABET = "甲乙丙丁戊"
_SMALL_NEGATION_ALPHABET = "非否"


@st.composite
def small_remark_tables(draw):
    """少量汉字随机生成的小关键词表：返回 (tags: dict[str, tuple], negations: tuple)。"""
    n_tags = draw(st.integers(min_value=1, max_value=4))
    tags = {}
    for i in range(n_tags):
        words = draw(st.lists(
            st.text(alphabet=_SMALL_TABLE_ALPHABET, min_size=1, max_size=3),
            min_size=1, max_size=3, unique=True))
        tags["标签%d" % i] = tuple(words)
    negations = tuple(draw(st.lists(
        st.text(alphabet=_SMALL_NEGATION_ALPHABET, min_size=1, max_size=2),
        min_size=0, max_size=2, unique=True)))
    return tags, negations


@st.composite
def small_table_with_sources(draw):
    """(tags, negations, sources)：sources 取自小表的关键词、前缀与填充字符。"""
    tags, negations = draw(small_remark_tables())
    keywords = sorted({w for ws in tags.values() for w in ws})
    sources = draw(remark_sources_for(keywords, negations, _SMALL_TABLE_ALPHABET + _SMALL_NEGATION_ALPHABET))
    return tags, negations, sources


# --- charset_text：Property 7、9（Req 3.13 字符集，长度 0–50；混入缺失值记号、批次关键字、备注关键词）---
import config as _config_cs  # noqa: E402

_CS_WHITESPACE = [" ", "\u3000", "\t", "\n", "\u00a0"]
_CS_CIRCLED = [chr(c) for c in range(0x2460, 0x2474)]  # ①–⑳


def charset_char_ok(ch):
    """字符是否属于 Req 3.13 字符集。"""
    o = ord(ch)
    return (
        0x20 <= o <= 0x7E
        or 0xFF01 <= o <= 0xFF5E
        or 0x4E00 <= o <= 0x9FFF
        or 0x2460 <= o <= 0x2473
        or ch in _CS_WHITESPACE
    )


def _cs_pool(items):
    """去重并只保留全部字符都在 Req 3.13 字符集内的非空词条。"""
    return sorted({s for s in items if s and all(charset_char_ok(c) for c in s)})


_cs_char = st.one_of(
    st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    st.characters(min_codepoint=0xFF01, max_codepoint=0xFF5E),
    st.characters(min_codepoint=0x4E00, max_codepoint=0x9FFF),
    st.sampled_from(_CS_CIRCLED),
    st.sampled_from(_CS_WHITESPACE),
)

_CS_MISSING_POOL = _cs_pool(
    list(_config_cs.MISSING_TOKENS)
    + [t.upper() for t in _config_cs.MISSING_TOKENS]
    + ["NaN", "None", "ＮＯＮＥ", "ｎａｎ", "－", "／", "\u3000NULL\u3000", " none "]
)
_CS_BATCH_POOL = _cs_pool(
    list(_config_cs.BATCH_KEYWORDS)
    + list(_config_cs.BATCH_NAME_NORMALIZE)
    + ["本科批", "专科批", "提前批", "本科普通批", "普通本科批", "高职高专普通批",
       "A", "B", "C", "A段", "B段", "1段", "一段", "二批", "（物理）", "(物理)",
       "（中外合作）", "-"]
)
_CS_REMARK_POOL = _cs_pool(
    [w for ws in _config_cs.REMARK_TAGS.values() for w in ws]
    + list(_config_cs.NEGATION_PREFIXES)
)
_CS_TABLE_POOL = _cs_pool(
    list(_config_cs.SUBJECT_MAP)
    + sorted(_config_cs.PROVINCE_NAMES)
    + [p + s for p in ("北京", "新疆", "广西", "内蒙古") for s in _config_cs.PROVINCE_SUFFIXES]
    + list(_config_cs.MAJOR_GROUP_STRIP_TOKENS)
    + ["01", "001", "0", "00", "1.0", "12.0", "０１．０", "①", "⑩", "⑳"]
)
_CS_PUNCT_POOL = ["(", ")", "（", "）", "-", "--", "()", "（）", "((", "))"]

_cs_plain = st.text(alphabet=_cs_char, min_size=1, max_size=8)
_cs_block = st.one_of(
    _cs_plain,
    _cs_plain,
    st.sampled_from(_CS_MISSING_POOL),
    st.sampled_from(_CS_BATCH_POOL),
    st.sampled_from(_CS_BATCH_POOL),
    st.sampled_from(_CS_REMARK_POOL),
    st.sampled_from(_CS_REMARK_POOL),
    st.sampled_from(_CS_TABLE_POOL),
    st.sampled_from(_CS_PUNCT_POOL),
    st.sampled_from(_CS_PUNCT_POOL),
)

# 先拼接 0–6 个块，再截断到 50 字符
charset_text = st.lists(_cs_block, min_size=0, max_size=6).map(lambda bs: "".join(bs)[:50])
