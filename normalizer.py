# -*- coding: utf-8 -*-
"""标准化函数。

本文件由 ``norm_text``、``norm_code`` 起步，后续任务继续在此追加其它标准化函数。
"""

import functools
import math
import numbers
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

_WS_RE = re.compile(r"\s+")
_INT_DOT_ZERO_RE = re.compile(r"^[0-9]+\.0$")
_MISSING_LOWER = frozenset(t.lower() for t in config.MISSING_TOKENS)


def norm_text(x):
    """str(x) -> NFKC -> 删除全部空白 -> 全角括号转半角 -> 缺失值记号返回 None。"""
    s = unicodedata.normalize("NFKC", str(x))
    s = _WS_RE.sub("", s)
    s = s.replace("（", "(").replace("）", ")")
    if s.lower() in _MISSING_LOWER:
        return None
    return s


def norm_code(x):
    """norm_text 后，仅当整串匹配 ^[0-9]+\\.0$ 时去掉末尾 ".0"；不补、不删前导零。"""
    s = norm_text(x)
    if s is not None and _INT_DOT_ZERO_RE.match(s):
        return s[:-2]
    return s


# ============================================================
# 其余标量函数（任务 5.1）
# ============================================================

_SIX_DIGITS_RE = re.compile(r"[0-9]{6}")
_NUMBER_RE = re.compile(r"[0-9]+(\.[0-9]+)?")
_LEADING_ZERO_RE = re.compile(r"(?<![0-9])0+(?=[0-9])")

# 映射表的键/词条在加载时先经 norm_text，维护人员可以直接写原文
_SUBJECT_MAP_NORM = {
    k: v
    for k, v in ((norm_text(k0), v0) for k0, v0 in config.SUBJECT_MAP.items())
    if k is not None
}
_BATCH_NORMALIZE_TERMS = tuple(
    t for t in (norm_text(t0) for t0 in config.BATCH_NAME_NORMALIZE) if t is not None
)
_MAJOR_GROUP_STRIP_TOKENS = tuple(
    t for t in (norm_text(t0) for t0 in config.MAJOR_GROUP_STRIP_TOKENS) if t is not None
)


def norm_province_id(x):
    """A 的 province_id：norm_code 后为 6 位数字时取前 2 位查国标前缀表，否则 None。"""
    s = norm_code(x)
    if s is None or not _SIX_DIGITS_RE.fullmatch(s):
        return None
    return config.PROVINCE_PREFIX_MAP.get(s[:2])


def norm_province_name(x):
    """B 的省份：按 PROVINCE_SUFFIXES 顺序只删除第一个匹配的结尾后缀一次；
    结果必须属于 PROVINCE_NAMES（31 个简称），否则 None。"""
    s = norm_text(x)
    if s is None:
        return None
    for suffix in config.PROVINCE_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s if s in config.PROVINCE_NAMES else None


def norm_subject_a(x):
    """选科：norm_text 后整体查 SUBJECT_MAP；表外值与缺失值返回 None。"""
    s = norm_text(x)
    if s is None:
        return None
    return _SUBJECT_MAP_NORM.get(s)


def norm_subject_b(x, province):
    """B 的选科：先 norm_subject_a，再按省一致性规则（Req 3.4）。

    province（标准省份简称或 None）属于 COMPREHENSIVE_PROVINCES 时只保留“综合”；
    否则（含 None）“综合”置为 None，物理类/历史类保留。
    """
    s = norm_subject_a(x)
    if s is None:
        return None
    is_comprehensive_subject = s == config.SUBJECT_COMPREHENSIVE
    if province in config.COMPREHENSIVE_PROVINCES:
        return s if is_comprehensive_subject else None
    return None if is_comprehensive_subject else s


def norm_major(x):
    """专业名称：只做 norm_text，不做同义词替换，也不删除括号内容。"""
    return norm_text(x)


def split_major_name(x):
    """专业名称 -> (去掉括号内容的主体, 括号片段元组)。

    例：``法学(中外合作办学)(合作方...)`` -> ``("法学", ("中外合作办学", "合作方..."))``。
    括号内容只作为关键词来源，不进入候选召回键；主体为空时回退为整个名称。
    """
    s = norm_text(x)
    if s is None:
        return None, ()
    base, fragments = _extract_bracket_fragments(s)
    fragments = tuple(f for f in fragments if not _is_empty(f))
    if _is_empty(base):
        base = s
    return base, fragments


def major_name_parts(x):
    """专业名称 -> (主体, 括号片段元组, 括号片段的备注标签掩码)。"""
    base, fragments = split_major_name(x)
    return base, fragments, tag_mask(fragments)


def _strip_enclosing_brackets(s):
    """反复删除整体包围文本的一对配对括号，如 "((01))" -> "01"；"(1)(2)" 不动。"""
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        enclosed = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    enclosed = False
                    break
        if not enclosed or depth != 0:
            break
        s = s[1:-1]
    return s


def norm_major_group(x):
    """专业组：norm_text -> 依次删除全部“第”“专业组”“组” -> 去外围配对括号
    -> 去每段数字的前导零（全零保留一个 0）-> 再做一次缺失判断。"""
    s = norm_text(x)
    if s is None:
        return None
    for token in _MAJOR_GROUP_STRIP_TOKENS:
        s = s.replace(token, "")
    s = _strip_enclosing_brackets(s)
    s = _LEADING_ZERO_RE.sub("", s)
    if s.lower() in _MISSING_LOWER:
        return None
    return s


def norm_number(x):
    """数值字段：数值单元格取其值；文本经 norm_text 后整串匹配
    ^[0-9]+(\\.[0-9]+)?$ 才转 float。0、NaN、非有限值、无法解析一律返回 None。"""
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        value = float(x)
    else:
        s = norm_text(x)
        if s is None or not _NUMBER_RE.fullmatch(s):
            return None
        value = float(s)
    if not math.isfinite(value) or value == 0:
        return None
    return value


def layer_from_column(x):
    """层次/层级列：norm_text -> 批次名称归一替换为“专科” ->
    含“本科”不含“专科”为本科，含“专科”不含“本科”为专科，其余 None。"""
    s = norm_text(x)
    if s is None:
        return None
    for term in _BATCH_NORMALIZE_TERMS:
        s = s.replace(term, config.BATCH_NORMALIZE_TARGET)
    has_undergrad = "本科" in s
    has_college = "专科" in s
    if has_undergrad and not has_college:
        return "本科"
    if has_college and not has_undergrad:
        return "专科"
    return None


# ============================================================
# 批次清洗链（任务 5.5，D11：唯一实现，匹配流程与预览工具共用）
# ============================================================

_LATIN_RE = re.compile(r"[A-Za-z]")
_SINGLE_LATIN_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
_TRAILING_DIGITS_RE = re.compile(r"[0-9]+$")

_BATCH_KEYWORDS = tuple(
    k for k in (norm_text(k0) for k0 in config.BATCH_KEYWORDS) if k is not None
)
_BATCH_NORMALIZE_TARGET = norm_text(config.BATCH_NORMALIZE_TARGET) or config.BATCH_NORMALIZE_TARGET
_ORDER_ITEMS = tuple(config.BATCH_ORDERS.items())
_SEGMENT_CN_DIGITS = frozenset(config.SEGMENT_CN_DIGITS)

EARLY_YES = "是"
EARLY_NO = "否"

BATCH_CHECK_EMPTY = "空"
BATCH_CHECK_UNRECOGNIZED = "未识别"
BATCH_CHECK_LEGAL = "合法"
BATCH_CHECK_NOT_IN_PROVINCE = "不在本省清单"


def _is_empty(t):
    """空串、None 或命中缺失值清单（不区分大小写）都视为空。"""
    return t is None or t == "" or t.lower() in _MISSING_LOWER


def _extract_bracket_fragments(s):
    """步骤③：栈配对括号，返回 (剩余文本, 片段列表)。

    - 每对最外层括号连同内容移除，内容（保留内层括号）作为片段；
    - 栈空时遇到的 ")" 直接删除；
    - 扫描结束栈仍非空时，最左一个未配对 "("（栈底）及其后全部内容移除，
      其后内容原样作为一个片段。
    """
    kept = []
    fragments = []
    stack = []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                start = stack.pop()
                if not stack:
                    fragments.append(s[start + 1:i])
            # 栈空时的 ")" 是未配对右括号，直接删除
        elif not stack:
            kept.append(ch)
    if stack:
        fragments.append(s[stack[0] + 1:])
    return "".join(kept), fragments


def _is_batch_piece(piece):
    """含批次关键字，或整段为单个拉丁字母。"""
    if len(piece) == 1 and _LATIN_RE.fullmatch(piece):
        return True
    return any(k in piece for k in _BATCH_KEYWORDS)


def clean_batch_steps_1_4(x):
    """批次清洗步骤①–④（不含别名替换），返回 (body, fragments)。

    body 为 None 表示空；fragments 为元组，步骤③片段在前、步骤④片段在后，空片段已丢弃。
    """
    s = norm_text(x)
    if s is None:
        return None, ()
    for term in _BATCH_NORMALIZE_TERMS:
        s = s.replace(term, _BATCH_NORMALIZE_TARGET)

    rest, fragments = _extract_bracket_fragments(s)
    fragments = [f for f in fragments if not _is_empty(f)]

    pieces = [p for p in rest.split("-") if not _is_empty(p)]
    if len(pieces) == 1:
        body = pieces[0]
    elif len(pieces) >= 2:
        body_parts = []
        for p in pieces:
            if _is_batch_piece(p):
                body_parts.append(p)
            else:
                fragments.append(p)
        body = "".join(body_parts)
    else:
        body = None
    if _is_empty(body):
        body = None
    return body, tuple(fragments)


# ---- 别名表 ----

_alias_cache = {"snapshot": None, "table": None}


def build_alias_table(raw=None):
    """把 BATCH_ALIAS 的键按步骤①–④清洗后建字典 {(省份或"*", 清洗后批次主体): 规范批次名}。"""
    if raw is None:
        raw = config.BATCH_ALIAS
    table = {}
    for (province, text), value in raw.items():
        body, _ = clean_batch_steps_1_4(text)
        if body is None:
            continue
        table[(province, body)] = value
    return table


def get_alias_table():
    """带缓存的别名表；config.BATCH_ALIAS 内容变化时自动重建。"""
    snapshot = tuple(config.BATCH_ALIAS.items())
    if _alias_cache["snapshot"] != snapshot or _alias_cache["table"] is None:
        _alias_cache["table"] = build_alias_table(config.BATCH_ALIAS)
        _alias_cache["snapshot"] = snapshot
    return _alias_cache["table"]


def apply_alias(body, province, alias):
    """步骤⑤：省份键优先于 "*"；province 为 None 时只查 "*"；都没有时返回 body。"""
    if body is None:
        return None
    if province is not None:
        hit = alias.get((province, body))
        if hit is not None:
            return hit
    return alias.get(("*", body), body)


def clean_batch(x, province, alias=None):
    """批次清洗五步，返回 (body, fragments)。

    ① norm_text ② 归一表替换为“专科” ③ 栈配对括号抽片段
    ④ 按“-”切分 ⑤ 按省别名替换。x 缺失返回 (None, ())。
    """
    body, fragments = clean_batch_steps_1_4(x)
    if body is None:
        return None, fragments
    if alias is None:
        alias = get_alias_table()
    return apply_alias(body, province, alias), fragments


def join_batch_remark(fragments):
    """批次备注文本：片段以备注分隔符拼接（仅用于输出，不参与匹配）。"""
    return config.REMARK_SEPARATOR.join(fragments)


# ---- 批次成分 ----

def extract_components(body):
    """返回 (layer, early, segment, order)；body 为空时全为 None。"""
    if _is_empty(body):
        return None, None, None, None

    has_undergrad = "本科" in body
    has_college = "专科" in body
    if has_undergrad and not has_college:
        layer = "本科"
    elif has_college and not has_undergrad:
        layer = "专科"
    else:
        layer = None

    # 段
    segments = set()
    if "段" in body:
        start = 0
        while True:
            i = body.find("段", start)
            if i < 0:
                break
            start = i + 1
            if i == 0:
                continue
            prev = body[i - 1]
            if _LATIN_RE.fullmatch(prev):
                if i < 2 or not _LATIN_RE.fullmatch(body[i - 2]):
                    segments.add(prev.upper())
            elif "0" <= prev <= "9":
                m = _TRAILING_DIGITS_RE.search(body[:i])
                if m:
                    segments.add(m.group(0))
            elif prev in _SEGMENT_CN_DIGITS:
                if i < 2 or body[i - 2] not in _SEGMENT_CN_DIGITS:
                    segments.add(prev)
    else:
        segments = {m.group(0).upper() for m in _SINGLE_LATIN_RE.finditer(body)}
    segment = next(iter(segments)) if len(segments) == 1 else None

    # 批序
    orders = {value for key, value in _ORDER_ITEMS if key in body}
    order = next(iter(orders)) if len(orders) == 1 else None

    # 提前标记
    if "提前" in body:
        early = EARLY_YES
    elif "普通" in body or layer is not None or segment is not None or order is not None:
        early = EARLY_NO
    else:
        early = None

    return layer, early, segment, order


def compose_batch_name(components):
    """按 BATCH_NAME_TEMPLATE 合成规范批次名；4 个成分全为空时返回 None。"""
    layer, early, segment, order = components
    if layer is None and early is None and segment is None and order is None:
        return None
    if layer:
        prefix = layer
    elif early == EARLY_NO:
        prefix = "普通"
    else:
        prefix = ""
    template = config.BATCH_NAME_TEMPLATE
    if not segment:
        template = template.replace("{segment}段", "")
    return template.format(
        prefix=prefix,
        early="提前" if early == EARLY_YES else "",
        order=order or "",
        segment=segment or "",
    )


def canonical_batch_name(name, province, alias=None):
    """批次名称 -> 规范批次名（清洗 -> 抽取成分 -> 合成），结果可为 None。"""
    body, _ = clean_batch(name, province, alias)
    return compose_batch_name(extract_components(body))


# ---- 省合法批次集合与核查 ----

def build_province_legal_sets(a_batch_table, province_list):
    """返回 dict[省份 -> frozenset[规范批次名]]。

    来源：A 批次对照表 exact 的值与 ambiguous 的全部候选名称（按 province_id 归省），
    以及各省批次清单（为 None 时跳过）。省份无法转换的行丢弃。
    """
    alias = get_alias_table()
    legal = {}

    def add(province_id, name):
        province = norm_province_id(province_id)
        if province is None:
            return
        canon = canonical_batch_name(name, province, alias)
        if canon is not None:
            legal.setdefault(province, set()).add(canon)

    for (pid, _bt), name in a_batch_table.exact.items():
        add(pid, name)
    for (pid, _bt), names in a_batch_table.ambiguous.items():
        for name in names:
            add(pid, name)
    if province_list is not None:
        for pid, name in zip(province_list["province_id"], province_list["batch_name"]):
            add(pid, name)

    return {p: frozenset(s) for p, s in legal.items()}


def batch_check(name_raw, name_std, province, legal):
    """按顺序取第一个成立的核查结果：空 / 未识别 / 合法 / 不在本省清单。只做诊断。"""
    if norm_text(name_raw) is None:
        return BATCH_CHECK_EMPTY
    if name_std is None:
        return BATCH_CHECK_UNRECOGNIZED
    if province is not None and legal is not None and name_std in legal.get(province, ()):
        return BATCH_CHECK_LEGAL
    return BATCH_CHECK_NOT_IN_PROVINCE


def merge_batch_remark(raw_remark, fragments):
    """合并批次备注：原“批次备注”列 norm_text 后（非缺失）整体作为第一个片段，其后接 fragments。
    全部为空时返回 ()，表示缺失。"""
    merged = []
    first = norm_text(raw_remark)
    if not _is_empty(first):
        merged.append(first)
    merged.extend(f for f in fragments if not _is_empty(f))
    return tuple(merged)


# ---- A 批次名称映射 ----

def map_a_batch_name(province_id, batch_type, table):
    """A 的 (province_id, batch_type) -> 批次名称：exact -> by_type -> None。
    batch_type 缺失返回 None；ambiguous 组合不在 exact/by_type 中，结果为 None。"""
    t = norm_code(batch_type)
    if t is None:
        return None
    p = norm_code(province_id)
    hit = table.exact.get((p, t))
    if hit is not None:
        return hit
    return table.by_type.get(t)


# ============================================================
# 备注标签提取（任务 5.11，Req 3.10、3.11、3.16）
# ============================================================

_tag_cache = {"snapshot": None, "data": None}


def _tag_snapshot():
    """config.REMARK_TAGS / NEGATION_PREFIXES 的内容快照，内容变化时缓存失效。"""
    return (
        tuple((str(tag), tuple(str(w) for w in words)) for tag, words in config.REMARK_TAGS.items()),
        tuple(str(p) for p in config.NEGATION_PREFIXES),
    )


def _build_tag_data(snapshot):
    """返回 (pattern, keyword_bits, prefixes)。

    关键词与否定前缀先经 norm_text；pattern 为按长度降序的关键词交替式
    （正则交替取第一个可匹配项，因此同一起点自然取最长关键词）；
    keyword_bits 为 关键词 -> 该关键词所属标签的位掩码。
    """
    tags, negations = snapshot
    keyword_bits = {}
    for index, (_tag, words) in enumerate(tags):
        for word in words:
            norm = norm_text(word)
            if norm is None:
                continue
            keyword_bits[norm] = keyword_bits.get(norm, 0) | (1 << index)
    prefixes = tuple(sorted(
        {p for p in (norm_text(p0) for p0 in negations) if p is not None},
        key=len, reverse=True,
    ))
    if keyword_bits:
        ordered = sorted(keyword_bits, key=lambda k: (-len(k), k))
        pattern = re.compile("|".join(re.escape(k) for k in ordered))
    else:
        pattern = None
    return pattern, keyword_bits, prefixes


def _get_tag_data():
    snapshot = _tag_snapshot()
    if _tag_cache["snapshot"] != snapshot or _tag_cache["data"] is None:
        _tag_cache["data"] = _build_tag_data(snapshot)
        _tag_cache["snapshot"] = snapshot
    return _tag_cache["data"]


def _fragment_mask(fragment, pattern, keyword_bits, prefixes):
    mask = 0
    for m in pattern.finditer(fragment):
        bits = keyword_bits[m.group(0)]
        if mask & bits == bits:
            continue
        before = fragment[:m.start()]
        if any(before.endswith(p) for p in prefixes):
            continue
        mask |= bits
    return mask


def tag_mask(text_or_fragments):
    """备注标签位掩码（Python int，标签数超过 64 也能表示）。

    输入为已 norm_text 的文本，或片段元组（每个片段独立匹配）；None / 空元组返回 0。
    每个片段内从左到右、同一起点取最长关键词、已匹配部分不再参与匹配；
    某标签只要存在一次未紧邻否定前缀的出现就置位（位序号 = 标签在 config.REMARK_TAGS 中的顺序）。
    匹配与否定前缀判断都不跨片段、不跨来源。
    """
    if text_or_fragments is None:
        return 0
    if isinstance(text_or_fragments, str):
        fragments = (text_or_fragments,)
    else:
        fragments = text_or_fragments
    pattern, keyword_bits, prefixes = _get_tag_data()
    if pattern is None:
        return 0
    mask = 0
    for fragment in fragments:
        if fragment:
            mask |= _fragment_mask(fragment, pattern, keyword_bits, prefixes)
    return mask


def or_masks(*masks):
    """按位或多个标签掩码。"""
    result = 0
    for m in masks:
        result |= int(m)
    return result


def mask_to_tags(mask):
    """把掩码展开为标签列表，顺序与 config.REMARK_TAGS 一致。"""
    mask = int(mask)
    return [tag for index, tag in enumerate(config.REMARK_TAGS) if (mask >> index) & 1]


# ============================================================
# 唯一值广播（任务 5.15，Req 3.12）
# ============================================================

def _object_array(n):
    """长度为 n 的 object 数组；逐元素赋值，避免元组被 numpy 展开成二维。"""
    return np.empty(n, dtype=object)


def _str_keys(col):
    """逐元素 ``str(x)`` 的 object 数组（numpy C 层转换）。

    不用 ``Series.astype(str)``：pandas 3 下它会保留 None/NaN 为缺失值，
    使 factorize/groupby 把它们当缺失丢弃（codes 为 -1/NaN），而不是字符串 "None"/"nan"。
    """
    return col.to_numpy(dtype=object).astype(str).astype(object)


def vec_map(col, fn):
    """先转字符串再 factorize，只对唯一值调用标量函数，再 take 回全列。

    键为 ``str(x)``，避免 ``1``、``1.0``、``True`` 在对象哈希中被合并；
    所有标量函数都以 ``str(x)`` 开头，因此对字符串键调用与对原值调用结果相同。
    返回 Series，index 与 ``col`` 一致；fn 的返回值（含元组）原样保留。
    """
    codes, uniques = pd.factorize(_str_keys(col))
    mapped = _object_array(len(uniques))
    for g, u in enumerate(uniques):
        mapped[g] = fn(u)
    return pd.Series(mapped.take(codes), index=col.index, dtype=object)


def vec_map_pair(first_col, second_col, fn, n_out):
    """按 (str(first), str(second)) 去重，每组取第一行的原始对象调用 ``fn(first, second)``。

    ``fn`` 返回长度为 ``n_out`` 的元组；各输出逐元素放入 object 数组，
    元素本身可以是元组（片段）。返回 ``n_out`` 个 Series，index 与 ``second_col`` 一致。
    """
    keys = pd.DataFrame({
        "p": _str_keys(first_col),
        "v": _str_keys(second_col),
    })
    codes = keys.groupby(["p", "v"], sort=False).ngroup().to_numpy()
    first = np.unique(codes, return_index=True)[1]
    p_arr = first_col.to_numpy(dtype=object)
    v_arr = second_col.to_numpy(dtype=object)
    cols = [_object_array(len(first)) for _ in range(n_out)]
    for g, i in enumerate(first):
        res = fn(p_arr[i], v_arr[i])
        for k in range(n_out):
            cols[k][g] = res[k]
    return tuple(
        pd.Series(c.take(codes), index=second_col.index, dtype=object) for c in cols
    )


# ============================================================
# 表级接口与诊断报告（任务 5.16，Req 3.6、3.10、3.12、3.14、3.15、3.21、3.22、11.2、16.2）
# ============================================================
#
# 约定（design.md 未写死处的取舍，均在此登记）：
# - A_scope_raw 保留 A 全量表的整数位置作为索引（split_scope 设置），normalize_a 据此得到
#   a_row = 索引 + 2（Excel 行号）；若表里已有 ``a_row`` 列（loader.assign_a_rows）则优先用该列。
# - B_kept 若含 ``b_id`` 列（loader.assign_b_ids）直接使用，否则按索引 + 2 生成 ``B_`` + 8 位行号。
# - A_std / B_std 使用 0..n-1 的 RangeIndex；a_pos / b_pos 与之相同，按位置与 A_scope_raw / B_kept 对齐。
# - 所有标准列的缺失值为 None；备注掩码为 Python int（object 列），分数列为 float64。
# - 本节全部为按列的向量化操作（唯一值广播 + numpy/pandas 列运算），不逐行遍历。

EXAMPLE_LIMIT = 20

A_STD_COLUMNS = (
    "a_pos", "a_row", "school_id", "province", "major_name", "major_base", "major_code",
    "subject", "batch_name", "batch_body", "batch_remark", "batch_layer", "early", "segment",
    "batch_order", "batch_name_std", "batch_check", "layer", "major_group", "remark_mask",
    "major_name_mask",
)
B_STD_COLUMNS = (
    "b_pos", "b_id", "school_id", "province", "major_name", "major_base", "major_code", "subject",
    "batch_name", "batch_body", "batch_remark", "batch_remark_merged", "batch_layer",
    "early", "segment", "batch_order", "batch_name_std", "batch_check", "layer",
    "major_group", "remark_mask", "major_name_mask",
) + tuple(out for _src, out in config.SCORE_COLUMNS)

BATCH_CHECK_VALUES = (
    BATCH_CHECK_EMPTY, BATCH_CHECK_UNRECOGNIZED, BATCH_CHECK_LEGAL, BATCH_CHECK_NOT_IN_PROVINCE,
)


# ---- 批次链（A、B 各一个模块级函数；匹配流程与预览工具共用，D11、Req 16.2）----

def _batch_chain_core(province, name, legal):
    """批次链公共部分：clean_batch -> extract_components -> compose_batch_name -> batch_check。

    输出 10 项：(batch_body, batch_remark, batch_layer, early, segment, batch_order,
    batch_name_std, batch_check, 批次备注片段掩码, 批次主体掩码)。
    两个掩码在此顺带算出，因为它们只依赖片段与主体（Req 3.10 的 batch_remark / batch_body 来源）。
    """
    body, fragments = clean_batch(name, province)
    layer, early, segment, order = extract_components(body)
    name_std = compose_batch_name((layer, early, segment, order))
    check = batch_check(name, name_std, province, legal)
    return (body, fragments, layer, early, segment, order, name_std, check,
            tag_mask(fragments), tag_mask(body))


def b_batch_chain(province, name, legal=None):
    """B 批次链：province 为标准省份（或 None），name 为“批次”列原值，legal 为省合法批次集合。

    返回 10 项元组，见 ``_batch_chain_core``。供 ``vec_map_pair(省份列, 批次列, fn, 10)`` 使用；
    ``normalize_b`` 与预览工具都经 ``bind_b_batch_chain`` 传入 ``functools.partial(b_batch_chain, legal=…)``，
    其 ``.func`` 是同一个模块级函数对象。
    """
    return _batch_chain_core(province, name, legal)


def a_batch_chain(province_id, batch_type, table, legal=None):
    """A 批次链：province_id、batch_type 为原值；map_a_batch_name 得到批次名称后走公共链。

    返回 11 项元组：(batch_name,) + ``_batch_chain_core`` 的 10 项。
    """
    province = norm_province_id(province_id)
    name = map_a_batch_name(province_id, batch_type, table)
    return (name,) + _batch_chain_core(province, name, legal)


def b_merged_remark_chain(raw_remark, batch_name):
    """B 合并批次备注（Req 3.22）：返回 (batch_remark_merged 元组, 其标签掩码)。

    片段只由清洗步骤①–④产生，与省份无关，所以按 (批次备注原值, 批次原值) 去重。
    """
    _body, fragments = clean_batch_steps_1_4(batch_name)
    merged = merge_batch_remark(raw_remark, fragments)
    return merged, tag_mask(merged)


def bind_b_batch_chain(legal):
    """返回绑定了 legal 的 B 批次链：``functools.partial(b_batch_chain, legal=legal)``。

    每次调用时按模块全局名取 ``b_batch_chain``，因此 normalize_b 与预览工具得到同一个底层函数。
    """
    return functools.partial(b_batch_chain, legal=legal)


def bind_a_batch_chain(table, legal):
    """返回绑定了 A 批次对照表与 legal 的 A 批次链 partial。"""
    return functools.partial(a_batch_chain, table=table, legal=legal)


# ---- 列辅助 ----

def _get_col(df, real_name):
    """取真实列；列缺失时返回等长全 None 列（缺列已由 loader 警告，此处不再重复）。"""
    if real_name in df.columns:
        return df[real_name].reset_index(drop=True)
    return pd.Series(np.full(len(df), None, dtype=object), dtype=object)


def _text_tag_mask(u):
    """vec_map 用：对文本 norm_text 后提取标签掩码（缺失为 0）。"""
    return tag_mask(norm_text(u))


def _mask_column(col):
    """文本列 -> 标签掩码数组（Python int；标签数超过 64 也能表示）。"""
    return vec_map(col, _text_tag_mask).to_numpy(dtype=object)


def _present(col):
    """norm_text 后非缺失的布尔数组。"""
    return vec_map(col, norm_text).notna().to_numpy()


def _or_sources(sources, order, n):
    """按 order 取各来源掩码并按位或；n 为 0 时返回空数组（Python int 掩码）。"""
    result = [0] * n
    for key in order:
        for i, value in enumerate(sources[key]):
            result[i] |= int(value)
    return np.array(result, dtype=object)


def _first_distinct(columns, mask, limit=EXAMPLE_LIMIT):
    """按行顺序取 mask 为真的行里前 limit 个不同值。

    columns 为一个或多个等长 object 数组；单列返回值列表，多列返回元组列表。
    """
    if not mask.any():
        return []
    frame = pd.DataFrame({i: c[mask] for i, c in enumerate(columns)})
    frame = frame.drop_duplicates().head(limit)
    cols = [frame[i].tolist() for i in range(len(columns))]
    return cols[0] if len(cols) == 1 else list(zip(*cols))


def _count_checks(check_col, rows):
    """batch_check 四类计数（0 也保留），并断言四者之和等于行数。"""
    counts = pd.Series(check_col, dtype=object).value_counts()
    result = {k: int(counts.get(k, 0)) for k in BATCH_CHECK_VALUES}
    assert sum(result.values()) == rows, "batch_check 四类计数之和必须等于行数"
    return result


def _excel_rows(df):
    """A 的 Excel 行号：优先 a_row 列，其次整数索引 + 2，否则按位置 + 2。"""
    if "a_row" in df.columns:
        return pd.to_numeric(df["a_row"], errors="coerce").to_numpy().astype(np.int64)
    if pd.api.types.is_integer_dtype(df.index):
        return df.index.to_numpy().astype(np.int64) + 2
    return np.arange(len(df), dtype=np.int64) + 2


# ---- 报告 ----

def _fmt_examples(items):
    return "（无）" if not items else repr(list(items))


@dataclass
class ScopeReport:
    """输出范围预筛（Req 11.2）报告；A 省份映射失败统计在全量 A 上（Req 3.15）。"""
    total: int = 0                       # 全量 A 行数
    kept: int = 0                        # A_scope 行数
    excluded: int = 0                    # 范围外行数
    province_failed: int = 0             # 全量 A 中原始省份非缺失、标准省份为空的行数
    province_failed_examples: list = field(default_factory=list)   # 原始值（前 20 个不同值）

    def format_lines(self):
        return [
            f"[范围预筛] A 原始行数={self.total} | 输出范围内={self.kept} | 范围外排除={self.excluded}",
            f"  A 省份无法转换行数={self.province_failed}（全量 A）",
            f"    示例原始值（前{EXAMPLE_LIMIT}个不同值）: {_fmt_examples(self.province_failed_examples)}",
        ]

    def print(self):
        for line in self.format_lines():
            print(line)


@dataclass
class NormalizeReport:
    """标准化诊断报告（Req 3.15）。示例都按行顺序取前 20 个不同值；计数为 0 也打印。

    补充字段（design.md 未列出，由取舍而来）：
    - ``province_rows``：省份映射失败的统计行数。B 等于 rows；A 为全量 A 行数（Req 3.15 / design 预筛）。
    - ``remark_hits``：仅 B，``Counter[(原文, 是否命中标签)] -> 单位个数``。单位按 design 的 Req 11.14 口径：
      “专业备注”“计划性质”每个非缺失值一个单位（原文取单元格 ``str``），合并批次备注按片段各一个单位。
      “是否命中”即 ``tag_mask(norm_text(原文)) != 0``。Output_Writer 取 ``hit`` 为 False 的项即得未命中备注清单。
    """
    table: str                                    # "A" / "B"
    rows: int = 0                                 # 参与标准化的行数
    province_failed: int = 0
    province_failed_examples: list = field(default_factory=list)
    province_rows: int = 0
    a_batch_unmapped: int = 0                     # 仅 A
    a_batch_unmapped_examples: list = field(default_factory=list)   # (province_id, batch_type) 原值
    batch_check_counts: dict = field(default_factory=dict)          # 空 / 未识别 / 合法 / 不在本省清单
    not_in_list_examples: list = field(default_factory=list)        # (标准省份, 原批次名称)
    unrecognized_examples: list = field(default_factory=list)       # (标准省份, 原批次名称)
    subject_dropped: int = 0                      # 仅 B
    subject_dropped_examples: list = field(default_factory=list)    # (标准省份, “选科”原值)
    remark_hits: Counter = field(default_factory=Counter)           # 仅 B
    # 专业名称括号片段 -> (片段原文, 是否命中标签) 的计数；A/B 都填
    major_name_hits: Counter = field(default_factory=Counter)

    def format_lines(self):
        lines = [f"[标准化诊断] 表={self.table} | 参与标准化行数={self.rows}"]
        lines.append(
            f"  省份无法转换（原始非缺失、标准省份为空）行数={self.province_failed}"
            f"（统计行数={self.province_rows}）")
        lines.append(f"    示例原始值（前{EXAMPLE_LIMIT}个不同值）: "
                     f"{_fmt_examples(self.province_failed_examples)}")
        if self.table == "A":
            lines.append(f"  batch_type 查不到批次名称行数={self.a_batch_unmapped}")
            lines.append(f"    示例 (province_id, batch_type): "
                         f"{_fmt_examples(self.a_batch_unmapped_examples)}")
        counts = self.batch_check_counts
        parts = " ".join(f"{k}={counts.get(k, 0)}" for k in BATCH_CHECK_VALUES)
        lines.append(f"  批次核查: {parts} | 合计={sum(counts.get(k, 0) for k in BATCH_CHECK_VALUES)}")
        lines.append(f"    不在本省清单示例 (标准省份, 原批次名称): "
                     f"{_fmt_examples(self.not_in_list_examples)}")
        lines.append(f"    未识别示例 (标准省份, 原批次名称): "
                     f"{_fmt_examples(self.unrecognized_examples)}")
        if self.table == "B":
            lines.append(f"  选科置空（原值非缺失、标准选科为缺失）行数={self.subject_dropped}")
            lines.append(f"    示例 (标准省份, 选科原值): {_fmt_examples(self.subject_dropped_examples)}")
        return lines

    def print(self):
        for line in self.format_lines():
            print(line)


# ---- 范围预筛 ----

def scope_keys(B_std):
    """参与匹配 B 的 (school_id, province) 唯一键集合（两者都非缺失），返回 MultiIndex。"""
    school = B_std["school_id"].to_numpy(dtype=object)
    province = B_std["province"].to_numpy(dtype=object)
    valid = pd.notna(school) & pd.notna(province)
    return pd.MultiIndex.from_arrays(
        [school[valid], province[valid]], names=["school_id", "province"]
    ).unique()


def split_scope(A_raw, keys, verbose=True):
    """输出范围预筛（D1、Req 11.2）：只标准化全量 A 的 school_id 与 province 两列，
    切出 (school_id, province) 出现在 ``keys`` 中的行。

    返回 (A_scope_raw, ScopeReport)。A_scope_raw 原列、原值、原顺序；索引为该行在全量 A 中的
    整数位置（供 normalize_a 得到 a_row = 位置 + 2）。A 省份映射失败在全量 A 上统计。
    """
    n = len(A_raw)
    raw_school = _get_col(A_raw, config.A_COLUMNS["school_id"])
    raw_province = _get_col(A_raw, config.A_COLUMNS["province"])
    school = vec_map(raw_school, norm_code).to_numpy(dtype=object)
    province = vec_map(raw_province, norm_province_id).to_numpy(dtype=object)

    failed = _present(raw_province) & pd.isna(province)
    valid = pd.notna(school) & pd.notna(province)
    in_scope = np.zeros(n, dtype=bool)
    if valid.any() and len(keys) > 0:
        probe = pd.MultiIndex.from_arrays([school[valid], province[valid]])
        in_scope[valid] = probe.isin(keys)

    positions = np.flatnonzero(in_scope)
    A_scope = A_raw.iloc[positions]
    A_scope.index = pd.Index(positions, dtype="int64")

    report = ScopeReport(
        total=n,
        kept=int(len(positions)),
        excluded=int(n - len(positions)),
        province_failed=int(failed.sum()),
        province_failed_examples=_first_distinct(
            [_str_keys(raw_province)], failed),
    )
    if verbose:
        report.print()
    return A_scope, report


# ---- B ----

def _b_remark_hits(raw_cols, merged):
    """remark_hits：见 NormalizeReport 说明。raw_cols 为“专业备注”“计划性质”原值列，merged 为合并批次备注列。"""
    hits = Counter()
    for col in raw_cols:
        present = _present(col)
        if not present.any():
            continue
        counts = pd.Series(_str_keys(col)[present]).value_counts()
        for text, count in counts.items():
            hits[(text, tag_mask(norm_text(text)) != 0)] += int(count)
    fragments = merged.explode().dropna()
    if len(fragments):
        for text, count in fragments.value_counts().items():
            hits[(text, tag_mask(text) != 0)] += int(count)
    return hits


def _major_name_hits(fragments_col):
    """专业名称括号片段的标签命中计数：Counter[(片段原文, 是否命中)]。"""
    hits = Counter()
    for fragments in fragments_col:
        for text in fragments:
            if not text:
                continue
            hits[(text, tag_mask(text) != 0)] += 1
    return hits


def _remark_source_hits(raw_cols, fragments_col=None, body_col=None):
    """备注来源文本的标签命中计数（A 侧诊断用）。

    raw_cols 为备注原值列；fragments_col 为片段元组列；body_col 为批次主体列。
    同一文本按出现次数累计；是否命中只看标签掩码是否非零。
    """
    hits = Counter()
    for col in raw_cols:
        present = _present(col)
        if not present.any():
            continue
        counts = pd.Series(_str_keys(col)[present]).value_counts()
        for text, count in counts.items():
            hits[(text, tag_mask(norm_text(text)) != 0)] += int(count)
    if fragments_col is not None:
        for fragments in fragments_col:
            for text in fragments:
                if text:
                    hits[(text, tag_mask(text) != 0)] += 1
    if body_col is not None:
        for text in body_col:
            if text and not _is_empty(text):
                hits[(text, tag_mask(text) != 0)] += 1
    return hits


def normalize_b(B_kept, legal, verbose=True):
    """B 表标准化，返回 (B_std, NormalizeReport)。不修改 B_kept。

    B_std 列见 ``B_STD_COLUMNS``（design.md Data Models）。``legal`` 为 ``build_province_legal_sets`` 的结果。
    标准层次按“层次”列 -> “层级”列 -> 批次层次成分取第一个非空值（Req 3.6）；
    备注掩码为 ``config.B_REMARK_SOURCES`` 四个来源逐列提取后按位或（Req 3.10）。
    """
    cols = config.B_COLUMNS
    n = len(B_kept)

    raw_school = _get_col(B_kept, cols["school_id"])
    raw_province = _get_col(B_kept, cols["province"])
    raw_name = _get_col(B_kept, cols["major_name"])
    raw_code = _get_col(B_kept, cols["major_code"])
    raw_subject = _get_col(B_kept, cols["subject"])
    raw_batch = _get_col(B_kept, cols["batch"])
    raw_group = _get_col(B_kept, cols["major_group"])
    raw_layer1 = _get_col(B_kept, cols["layer_primary"])
    raw_layer2 = _get_col(B_kept, cols["layer_secondary"])
    raw_remark_major = _get_col(B_kept, cols["remark_major"])
    raw_remark_batch = _get_col(B_kept, cols["remark_batch"])
    raw_remark_plan = _get_col(B_kept, cols["remark_plan"])

    school = vec_map(raw_school, norm_code)
    province = vec_map(raw_province, norm_province_name)
    major_name = vec_map(raw_name, norm_major)
    major_parts = vec_map(raw_name, major_name_parts)
    major_base = major_parts.map(lambda p: p[0])
    major_fragments = major_parts.map(lambda p: p[1])
    major_name_mask = major_parts.map(lambda p: p[2])
    major_code = vec_map(raw_code, norm_code)
    major_group = vec_map(raw_group, norm_major_group)

    (subject,) = vec_map_pair(province, raw_subject,
                              lambda p, v: (norm_subject_b(v, p),), 1)

    chain = bind_b_batch_chain(legal)
    (body, fragments, b_layer, early, segment, order, name_std, check,
     mask_fragments, mask_body) = vec_map_pair(province, raw_batch, chain, 10)
    merged, mask_merged = vec_map_pair(raw_remark_batch, raw_batch, b_merged_remark_chain, 2)

    layer_arr = vec_map(raw_layer1, layer_from_column).to_numpy(dtype=object)
    layer_alt = vec_map(raw_layer2, layer_from_column).to_numpy(dtype=object)
    layer = np.where(pd.notna(layer_arr), layer_arr,
                     np.where(pd.notna(layer_alt), layer_alt, b_layer.to_numpy(dtype=object)))

    sources = {
        "remark_major": _mask_column(raw_remark_major),
        "batch_remark_merged": mask_merged.to_numpy(dtype=object),
        "remark_plan": _mask_column(raw_remark_plan),
        "batch_body": mask_body.to_numpy(dtype=object),
        "batch_remark": mask_fragments.to_numpy(dtype=object),
    }
    remark_mask = _or_sources(sources, config.B_REMARK_SOURCES, n)

    if "b_id" in B_kept.columns:
        b_id = B_kept["b_id"].reset_index(drop=True).to_numpy(dtype=object)
    else:
        b_id = _object_array(n)
        b_id[:] = [f"B_{r:08d}" for r in _excel_rows(B_kept).tolist()]

    data = {
        "b_pos": np.arange(n, dtype=np.int64),
        "b_id": b_id,
        "school_id": school, "province": province,
        "major_name": major_name, "major_base": major_base,
        "major_code": major_code, "subject": subject,
        "batch_name": raw_batch, "batch_body": body, "batch_remark": fragments,
        "batch_remark_merged": merged, "batch_layer": b_layer,
        "early": early, "segment": segment, "batch_order": order,
        "batch_name_std": name_std, "batch_check": check,
        "layer": pd.Series(layer, dtype=object),
        "major_group": major_group,
        "remark_mask": remark_mask,
        "major_name_mask": major_name_mask,
    }
    for src, out in config.SCORE_COLUMNS:
        data[out] = pd.to_numeric(
            vec_map(_get_col(B_kept, src), norm_number), errors="coerce"
        ).astype("float64")
    B_std = pd.DataFrame(data, columns=list(B_STD_COLUMNS))

    # ---- 诊断 ----
    prov_arr = province.to_numpy(dtype=object)
    batch_str = _str_keys(raw_batch)
    check_arr = check.to_numpy(dtype=object)
    p_failed = _present(raw_province) & pd.isna(prov_arr)
    subject_dropped = _present(raw_subject) & subject.isna().to_numpy()
    report = NormalizeReport(
        table="B",
        rows=n,
        province_failed=int(p_failed.sum()),
        province_failed_examples=_first_distinct([_str_keys(raw_province)], p_failed),
        province_rows=n,
        batch_check_counts=_count_checks(check_arr, n),
        not_in_list_examples=_first_distinct(
            [prov_arr, batch_str], check_arr == BATCH_CHECK_NOT_IN_PROVINCE),
        unrecognized_examples=_first_distinct(
            [prov_arr, batch_str], check_arr == BATCH_CHECK_UNRECOGNIZED),
        subject_dropped=int(subject_dropped.sum()),
        subject_dropped_examples=_first_distinct(
            [prov_arr, _str_keys(raw_subject)], subject_dropped),
        remark_hits=_b_remark_hits([raw_remark_major, raw_remark_plan], merged),
        major_name_hits=_major_name_hits(major_fragments),
    )
    if verbose:
        report.print()
    return B_std, report


# ---- A ----

def normalize_a(A_scope_raw, a_batch, legal, scope_report=None, verbose=True):
    """A_scope 标准化，返回 (A_std, NormalizeReport)。不修改 A_scope_raw（行数、顺序、列与值都不变）。

    标准列写入独立的 ``A_std``（列见 ``A_STD_COLUMNS``），按位置与 A_scope_raw 一一对应，``a_row``
    为 Excel 行号。批次链按 (province_id, batch_type) 经 ``vec_map_pair`` 去重计算；备注掩码为
    ``config.A_REMARK_SOURCES`` 六个来源逐列提取后按位或。

    ``scope_report`` 为 ``split_scope`` 返回的报告：给出时，A 的省份映射失败数与示例取自全量 A
    （design 预筛说明）；未给出时在 A_scope_raw 上统计。
    """
    cols = config.A_COLUMNS
    n = len(A_scope_raw)

    raw_school = _get_col(A_scope_raw, cols["school_id"])
    raw_province = _get_col(A_scope_raw, cols["province"])
    raw_name = _get_col(A_scope_raw, cols["major_name"])
    raw_code = _get_col(A_scope_raw, cols["major_code"])
    raw_subject = _get_col(A_scope_raw, cols["subject"])
    raw_batch_type = _get_col(A_scope_raw, cols["batch"])
    raw_group = _get_col(A_scope_raw, cols["major_group"])
    raw_remark_major = _get_col(A_scope_raw, cols["remark_major"])
    raw_remark_plan = _get_col(A_scope_raw, cols["remark_plan"])
    raw_remark_direction = _get_col(A_scope_raw, cols["remark_direction"])
    raw_remark_group = _get_col(A_scope_raw, cols["remark_group"])

    school = vec_map(raw_school, norm_code)
    province = vec_map(raw_province, norm_province_id)
    major_name = vec_map(raw_name, norm_major)
    major_parts = vec_map(raw_name, major_name_parts)
    major_base = major_parts.map(lambda p: p[0])
    major_fragments = major_parts.map(lambda p: p[1])
    major_name_mask = major_parts.map(lambda p: p[2])
    major_code = vec_map(raw_code, norm_code)
    subject = vec_map(raw_subject, norm_subject_a)
    major_group = vec_map(raw_group, norm_major_group)

    chain = bind_a_batch_chain(a_batch, legal)
    (batch_name, body, fragments, layer, early, segment, order, name_std, check,
     mask_fragments, mask_body) = vec_map_pair(raw_province, raw_batch_type, chain, 11)

    sources = {
        "remark_major": _mask_column(raw_remark_major),
        "remark_plan": _mask_column(raw_remark_plan),
        "remark_direction": _mask_column(raw_remark_direction),
        "remark_group": _mask_column(raw_remark_group),
    }
    sources["batch_remark"] = mask_fragments.to_numpy(dtype=object)
    sources["batch_body"] = mask_body.to_numpy(dtype=object)
    remark_mask = _or_sources(sources, config.A_REMARK_SOURCES, n)

    A_std = pd.DataFrame({
        "a_pos": np.arange(n, dtype=np.int64),
        "a_row": _excel_rows(A_scope_raw),
        "school_id": school, "province": province,
        "major_name": major_name, "major_base": major_base,
        "major_code": major_code, "subject": subject,
        "batch_name": batch_name, "batch_body": body, "batch_remark": fragments,
        "batch_layer": layer, "early": early, "segment": segment, "batch_order": order,
        "batch_name_std": name_std, "batch_check": check,
        "layer": layer,                       # A 的标准层次即批次层次成分（Req 3.6）
        "major_group": major_group,
        "remark_mask": remark_mask,
        "major_name_mask": major_name_mask,
    }, columns=list(A_STD_COLUMNS))

    # ---- 诊断 ----
    prov_arr = province.to_numpy(dtype=object)
    name_arr = batch_name.to_numpy(dtype=object)
    check_arr = check.to_numpy(dtype=object)
    name_str = _str_keys(batch_name)
    if scope_report is not None:
        p_failed_n = scope_report.province_failed
        p_failed_ex = list(scope_report.province_failed_examples)
        p_rows = scope_report.total
    else:
        p_failed = _present(raw_province) & pd.isna(prov_arr)
        p_failed_n = int(p_failed.sum())
        p_failed_ex = _first_distinct([_str_keys(raw_province)], p_failed)
        p_rows = n
    unmapped = _present(raw_batch_type) & pd.isna(name_arr)
    report = NormalizeReport(
        table="A",
        rows=n,
        province_failed=p_failed_n,
        province_failed_examples=p_failed_ex,
        province_rows=p_rows,
        a_batch_unmapped=int(unmapped.sum()),
        a_batch_unmapped_examples=_first_distinct(
            [_str_keys(raw_province), _str_keys(raw_batch_type)], unmapped),
        batch_check_counts=_count_checks(check_arr, n),
        not_in_list_examples=_first_distinct(
            [prov_arr, name_str], check_arr == BATCH_CHECK_NOT_IN_PROVINCE),
        unrecognized_examples=_first_distinct(
            [prov_arr, name_str], check_arr == BATCH_CHECK_UNRECOGNIZED),
        major_name_hits=_major_name_hits(major_fragments),
        remark_hits=_remark_source_hits(
            [raw_remark_major, raw_remark_plan, raw_remark_direction, raw_remark_group],
            fragments),
    )
    if verbose:
        report.print()
    return A_std, report
