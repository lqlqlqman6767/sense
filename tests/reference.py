# -*- coding: utf-8 -*-
"""测试用朴素参考实现（只用于比对，不进入生产代码）。后续任务按需追加。"""


# ---------------------------------------------------------------------------
# Property 1：B 过滤（Req 2.2、2.3、2.4）
# ---------------------------------------------------------------------------
def _ref_flag(x):
    """strip 后，整串形如 ``<数字>.0`` 则去掉 ``.0``；缺失视为空串。"""
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0") and len(s) > 2 and s[:-2].isascii() and s[:-2].isdigit():
        s = s[:-2]
    return s


def naive_filter_b(rows, soft_delete_col, current_col, granularity_col,
                   granularity_keep="专业分"):
    """逐行朴素过滤。rows 为 dict 列表。

    返回 (保留行下标列表, 排除计数 dict)；排除计数的键依次为
    ``soft_delete`` / ``is_current`` / ``granularity``，每条排除记录只计第一个未通过的条件。
    """
    kept = []
    excluded = {"soft_delete": 0, "is_current": 0, "granularity": 0}
    for i, row in enumerate(rows):
        if _ref_flag(row.get(soft_delete_col)) != "0":
            excluded["soft_delete"] += 1
        elif _ref_flag(row.get(current_col)) != "1":
            excluded["is_current"] += 1
        elif _ref_flag(row.get(granularity_col)) != granularity_keep:
            excluded["granularity"] += 1
        else:
            kept.append(i)
    return kept, excluded


# ---------------------------------------------------------------------------
# Property 8：备注标签提取（Req 3.10、3.11）
# ---------------------------------------------------------------------------
def naive_extract_tags(sources, tags, negation_prefixes):
    """朴素备注标签提取，不依赖 normalizer.tag_mask 的内部实现。

    sources：每个来源是 None、str，或片段元组（文本已 norm_text）。
    tags：有序 dict[标签, 关键词序列]；negation_prefixes：否定前缀序列。
    规则：关键词与否定前缀先经 norm_text；每个来源、每个片段独立处理；
    片段内从左到右扫描，每个起点取最长关键词，命中后跳过该关键词（被覆盖的较短关键词不计出现）；
    出现紧邻其前的文本（仅限同一片段）以任一否定前缀结尾则该出现无效；
    某标签只要有一次有效出现即被提取。返回按 tags 顺序排列的标签列表。
    """
    from normalizer import norm_text

    kw_tags = {}  # 规范化关键词 -> 含该关键词的标签集合
    for tag, words in tags.items():
        for w in words:
            nw = norm_text(w)
            if nw is not None:
                kw_tags.setdefault(nw, set()).add(tag)
    prefixes = [p for p in (norm_text(p0) for p0 in negation_prefixes) if p is not None]

    found = set()
    for source in sources:
        if source is None:
            continue
        fragments = [source] if isinstance(source, str) else list(source)
        for frag in fragments:
            if not frag:
                continue
            i = 0
            while i < len(frag):
                best = None
                for kw in kw_tags:
                    if frag.startswith(kw, i) and (best is None or len(kw) > len(best)):
                        best = kw
                if best is None:
                    i += 1
                    continue
                before = frag[:i]
                if not any(before.endswith(p) for p in prefixes):
                    found |= kw_tags[best]
                i += len(best)
    return [t for t in tags if t in found]


# ---------------------------------------------------------------------------
# Property 7：批次清洗五步与成分抽取（Req 3.5、3.6、3.18）
# 按需求文字逐步写出，不调用 normalizer.clean_batch / extract_components 的任何内部函数。
# ---------------------------------------------------------------------------
def _ref_is_empty(t):
    """空串或命中缺失值清单（不区分大小写，同 norm_text 的缺失判断）。"""
    import config

    return t is None or t == "" or t.lower() in {m.lower() for m in config.MISSING_TOKENS}


def _ref_is_latin(ch):
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def _ref_is_batch_piece(piece, keywords):
    """含批次关键字，或整段为单个拉丁字母。"""
    return any(k in piece for k in keywords) or (len(piece) == 1 and _ref_is_latin(piece))


def ref_clean_batch_steps_1_4(x):
    """步骤①–④。返回 (主体或 None, 步骤③片段列表, 步骤④片段列表)。"""
    import config
    from normalizer import norm_text

    # ① 格式标准化
    s = norm_text(x)
    if s is None:
        return None, [], []
    # ② 按归一表顺序，把每个词条的全部出现替换为“专科”
    for term in config.BATCH_NAME_NORMALIZE:
        nt = norm_text(term)
        if nt is not None:
            s = s.replace(nt, norm_text(config.BATCH_NORMALIZE_TARGET))
    # ③ 圆括号从左到右配对，用嵌套深度计数
    rest = []
    bracket_frags = []
    depth = 0
    outer_start = None
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                outer_start = i
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue  # 未配对右括号：直接删除
            depth -= 1
            if depth == 0:
                bracket_frags.append(s[outer_start + 1:i])  # 最外层一对：内容作为片段
        elif depth == 0:
            rest.append(ch)
    if depth > 0:
        # 最左未配对左括号起到末尾整体移出，其后内容原样为一个片段
        bracket_frags.append(s[outer_start + 1:])
    bracket_frags = [f for f in bracket_frags if not _ref_is_empty(f)]
    rest = "".join(rest)
    # ④ 按半角“-”切分，丢弃空片段
    keywords = [norm_text(k) for k in config.BATCH_KEYWORDS]
    keywords = [k for k in keywords if k is not None]
    pieces = [p for p in rest.split("-") if not _ref_is_empty(p)]
    dash_frags = []
    if len(pieces) == 0:
        body = None
    elif len(pieces) == 1:
        body = pieces[0]
    else:
        parts = []
        for p in pieces:
            if _ref_is_batch_piece(p, keywords):
                parts.append(p)
            else:
                dash_frags.append(p)
        body = "".join(parts)
    if _ref_is_empty(body):
        body = None
    return body, bracket_frags, dash_frags


def ref_build_alias(raw_alias):
    """别名表：键文本按步骤①–④清洗；清洗后主体为空的键丢弃。"""
    table = {}
    for (province, text), value in raw_alias.items():
        body, _b, _d = ref_clean_batch_steps_1_4(text)
        if body is not None:
            table[(province, body)] = value
    return table


def ref_clean_batch(x, province, raw_alias=None):
    """五步清洗，返回 (主体或 None, 片段元组)；片段顺序为步骤③在前、步骤④在后。"""
    import config

    body, bracket_frags, dash_frags = ref_clean_batch_steps_1_4(x)
    fragments = tuple(bracket_frags + dash_frags)
    if body is None:
        return None, fragments
    table = ref_build_alias(config.BATCH_ALIAS if raw_alias is None else raw_alias)
    # ⑤ 省份键优先于 "*"
    if province is not None and (province, body) in table:
        return table[(province, body)], fragments
    if ("*", body) in table:
        return table[("*", body)], fragments
    return body, fragments


def ref_extract_components(body):
    """返回 (层次, 提前, 段, 批序)，按 Req 3.18 字面抽取。"""
    import config

    if _ref_is_empty(body):
        return None, None, None, None
    # 层次
    if "本科" in body and "专科" not in body:
        layer = "本科"
    elif "专科" in body and "本科" not in body:
        layer = "专科"
    else:
        layer = None
    # 段
    values = set()
    if "段" in body:
        for i, ch in enumerate(body):
            if ch != "段" or i == 0:
                continue
            prev = body[i - 1]
            if _ref_is_latin(prev):
                if i == 1 or not _ref_is_latin(body[i - 2]):
                    values.add(prev.upper())
            elif prev in "0123456789":
                j = i - 1
                while j > 0 and body[j - 1] in "0123456789":
                    j -= 1
                values.add(body[j:i])
            elif prev in config.SEGMENT_CN_DIGITS:
                if i == 1 or body[i - 2] not in config.SEGMENT_CN_DIGITS:
                    values.add(prev)
    else:
        for i, ch in enumerate(body):
            if not _ref_is_latin(ch):
                continue
            if i > 0 and _ref_is_latin(body[i - 1]):
                continue
            if i + 1 < len(body) and _ref_is_latin(body[i + 1]):
                continue
            values.add(ch.upper())
    segment = next(iter(values)) if len(values) == 1 else None
    # 批序
    orders = {v for k, v in config.BATCH_ORDERS.items() if k in body}
    order = next(iter(orders)) if len(orders) == 1 else None
    # 提前标记
    if "提前" in body:
        early = "是"
    elif "普通" in body or layer is not None or segment is not None or order is not None:
        early = "否"
    else:
        early = None
    return layer, early, segment, order


def ref_layer_from_column(x):
    """“层次”“层级”列的判定（Req 3.6）：格式标准化、归一替换后看是否含本科/专科。"""
    import config
    from normalizer import norm_text

    s = norm_text(x)
    if s is None:
        return None
    for term in config.BATCH_NAME_NORMALIZE:
        nt = norm_text(term)
        if nt is not None:
            s = s.replace(nt, norm_text(config.BATCH_NORMALIZE_TARGET))
    if "本科" in s and "专科" not in s:
        return "本科"
    if "专科" in s and "本科" not in s:
        return "专科"
    return None


def ref_b_layer(layer_primary, layer_secondary, batch, province=None):
    """B 标准层次：“层次”列 → “层级”列 → 批次层次成分，取第一个非空值。"""
    first = ref_layer_from_column(layer_primary)
    if first is not None:
        return first
    second = ref_layer_from_column(layer_secondary)
    if second is not None:
        return second
    body, _frags = ref_clean_batch(batch, province)
    return ref_extract_components(body)[0]
