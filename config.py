# -*- coding: utf-8 -*-
"""招生计划匹配系统配置。

所有路径、列名映射、规则表、编码表、词表集中在本文件维护；
其它模块只 ``import config`` 读取，不定义同类常量。
备注标签表与否定前缀放在 ``remark_tags.json``，由 :func:`load_remark_tags` 加载。
"""

import json
from pathlib import Path

# ============================================================
# 1. 路径与读取
# ============================================================

BASE_DIR = Path(__file__).resolve().parent  # ABC/ 目录，与启动目录无关


def resolve_path(p):
    """相对路径以 BASE_DIR 为基准解析为绝对路径；绝对路径原样返回。"""
    path = Path(p)
    return path if path.is_absolute() else BASE_DIR / path


INPUT_A = resolve_path("data/26招生计划.xlsx")
INPUT_B = resolve_path("data/26年专业分数据.xlsx")
OUTPUT = resolve_path("output/匹配结果.xlsx")
KEYWORD_REPORT = resolve_path("output/备注关键词候选.csv")
SHEET_A = 0
SHEET_B = 0

EXCEL_READ_ENGINE = "calamine"  # 未安装 python-calamine 时由 loader 回退 openpyxl 并警告
EXCEL_MAX_DATA_ROWS = 1_048_575

# A 批次对照表（GBK，读 province_id / batch_name / batch_type，忽略 year）
A_BATCH_TABLE = resolve_path("data/批次.csv")
A_BATCH_TABLE_ENCODING = "gbk"
A_BATCH_TABLE_COLUMNS = ("province_id", "batch_name", "batch_type")

# 各省批次清单（只用于按省核查）
PROVINCE_BATCH_LIST = resolve_path("data/各省批次映射.xlsx")
PROVINCE_BATCH_SHEET = "各省批次映射"
PROVINCE_BATCH_COLUMNS = ("province_id", "batch_name")

# B 批次清洗预览工具输出目录
PREVIEW_OUTPUT_DIR = resolve_path("output")

# ============================================================
# 2. 列映射
# ============================================================

# 标准字段 -> A 真实列名（11 项）
A_COLUMNS = {
    "school_id": "school_id",
    "province": "province_id",
    "major_name": "special_name",
    "major_code": "special_code",
    "subject": "subject",
    "batch": "batch_type",
    "major_group": "major_group",
    "remark_major": "major_remark",
    "remark_plan": "remark",
    "remark_direction": "major_direction",
    "remark_group": "major_group_remark",
}

# A 备注来源（顺序即提取顺序）；batch_remark / batch_body 为批次清洗派生字段
A_REMARK_SOURCES = (
    "remark_major",
    "remark_plan",
    "remark_direction",
    "remark_group",
    "batch_remark",
    "batch_body",
)

# 标准字段 -> B 真实列名（15 项，表头全文以真实文件为准）
B_COLUMNS = {
    "school_id": "院校ID",
    "province": "省份",
    "major_name": "专业名称",
    "major_code": "专业代码",
    "subject": "选科",  # 只精确对应“选科”列，“选科组合（如 物理+化学）”不参与
    "batch": "批次",
    "major_group": "专业组名称",
    "layer_primary": "层次：本科/专科",
    "layer_secondary": "层级",
    "granularity": "数据颗粒度：专业分/专业组分/院校分",
    "soft_delete": "软删除标记：0正常 1已删除",
    "is_current": "当前有效标记：0历史 1当前有效",
    "remark_major": "专业备注",
    "remark_batch": "批次备注",
    "remark_plan": "计划性质（如 非定向）",
}

# B 备注来源；batch_remark_merged = “批次备注”列 + 批次括号/“-后缀”片段
B_REMARK_SOURCES = (
    "remark_major",
    "batch_remark_merged",
    "remark_plan",
    "batch_body",
)

# B 源列 -> 输出列（有序，7 项）
SCORE_COLUMNS = [
    ("最低分", "B_最低分"),
    ("最高分", "B_最高分"),
    ("平均分", "B_平均分"),
    ("最低分位次", "B_最低分位次"),
    ("平均分位次", "B_平均分位次"),
    ("最高分位次", "B_最高分位次"),
    ("录取人数", "B_录取人数"),
]

# B 过滤：(标准字段, 保留值)，按此顺序执行
B_FLAG_FILTERS = [("soft_delete", "0"), ("is_current", "1")]
GRANULARITY_FIELD = "granularity"
GRANULARITY_MAJOR = "专业分"
# 颗粒度扩展入口，默认不启用
GRANULARITY_EXTENSIONS = {"专业组分": False, "院校分": False}

# ============================================================
# 3. 映射表与词表
# ============================================================

MISSING_TOKENS = frozenset({"", "nan", "none", "null", "无", "未知", "不详", "-", "--", "/"})

# 国标省级 2 位码 -> 省份简称（31 项）
PROVINCE_PREFIX_MAP = {
    "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
    "21": "辽宁", "22": "吉林", "23": "黑龙江",
    "31": "上海", "32": "江苏", "33": "浙江", "34": "安徽", "35": "福建",
    "36": "江西", "37": "山东",
    "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
    "46": "海南",
    "50": "重庆", "51": "四川", "52": "贵州", "53": "云南", "54": "西藏",
    "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏", "65": "新疆",
}
# 长后缀在前，只删除第一个匹配到的后缀一次
PROVINCE_SUFFIXES = ("维吾尔自治区", "壮族自治区", "回族自治区", "自治区", "省", "市")
PROVINCE_NAMES = frozenset(PROVINCE_PREFIX_MAP.values())

# 选科：A、B 共用，整体相等查表
SUBJECT_MAP = {
    "物理": "物理类", "物理类": "物理类", "理科": "物理类",
    "历史": "历史类", "历史类": "历史类", "文科": "历史类",
    "综合": "综合", "不限": "综合",
}
SUBJECT_COMPREHENSIVE = "综合"
COMPREHENSIVE_PROVINCES = frozenset({"北京", "天津", "上海", "浙江", "山东", "海南"})

# 专业组标准化：按此顺序删除全部出现（Req 3.7）
MAJOR_GROUP_STRIP_TOKENS = ("第", "专业组", "组")
# A 专业名称 -> B 专业名称（标准化后），默认空
MAJOR_NAME_ALIAS = {}

# ============================================================
# 4. 批次
# ============================================================

BATCH_NAME_NORMALIZE = ("高职(专科)", "专科(高职)", "高职高专", "高职专科", "高职")
BATCH_NORMALIZE_TARGET = "专科"
BATCH_KEYWORDS = ("本科", "专科", "提前", "普通", "批", "段", "一批", "二批", "三批")
BATCH_ORDERS = {"一批": "一", "二批": "二", "三批": "三"}
SEGMENT_CN_DIGITS = "一二三四五六七八九十"
BATCH_NAME_TEMPLATE = "{prefix}{early}{order}批{segment}段"
# (标准省份或 "*", 批次主体) -> 规范批次名
BATCH_ALIAS = {
    ("*", "本科普通批"): "本科批",
    ("*", "普通本科批"): "本科批",
    ("*", "专科普通批"): "专科批",
    ("*", "高职高专普通批"): "专科批",
    # 特殊类型批次没有显式“本科/专科”字样；这些批次按本科批归并，
    # 保证层次可识别，同时不丢失“本科批”这一层信息。
    ("*", "地方专项"): "本科批",
    ("*", "地方专项计划"): "本科批",
    ("*", "国家专项计划"): "本科批",
    ("*", "高校专项"): "本科批",
    ("*", "高校专项计划"): "本科批",
    ("*", "高水平运动队"): "本科批",
    ("*", "高水平运动队招生"): "本科批",
    ("*", "高水平运动队志愿"): "本科批",
    ("*", "特殊类型招生"): "本科批",
    ("*", "特殊类型志愿"): "本科批",
}

# ============================================================
# 5. 备注标签
# ============================================================

REMARK_TAGS_FILE = resolve_path("remark_tags.json")
REMARK_SEPARATOR = "|"


def load_remark_tags(path=REMARK_TAGS_FILE):
    """加载备注标签表。

    返回 (tags, negation_prefixes)：tags 为保持文件顺序的 dict[标签 -> 关键词元组]，
    negation_prefixes 为否定前缀元组。
    """
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = {str(k): tuple(str(w) for w in v) for k, v in data.get("tags", {}).items()}
    prefixes = tuple(str(p) for p in data.get("negation_prefixes", []))
    return tags, prefixes


REMARK_TAGS, NEGATION_PREFIXES = load_remark_tags(REMARK_TAGS_FILE)

# ============================================================
# 6. 规则表与编码
# ============================================================

# 用列表而非 dict，便于检测重复编号
RULE_TABLE = [
    ("R101", "院校+省份+专业", "NORMAL"),
    ("R201", "院校+省份+专业+选科", "STRONG"),
    ("R301", "院校+省份+专业+专业组", "STRONG"),
    ("R302", "院校+省份+专业+选科+专业组", "STRONG"),
    ("R401", "院校+省份+专业代码（含其它任意核心证据）", "STRONG"),
    ("R501", "键1–4与下一组并列、由键5–7决出，只有备注更优", "WEAK"),
    ("R601", "键1–4与下一组并列、由键5–7决出，只有标准批次更优", "WEAK"),
    ("R602", "键1–4与下一组并列、由键5–7决出，备注与标准批次都更优", "WEAK"),
]

# 核心证据 (code, subject, major_group) -> 规则
CORE_RULE_TABLE = {
    (True, True, True): "R401",
    (True, True, False): "R401",
    (True, False, True): "R401",
    (True, False, False): "R401",
    (False, True, True): "R302",
    (False, True, False): "R201",
    (False, False, True): "R301",
    (False, False, False): "R101",
}
# (备注更优, 标准批次更优) -> 规则
SOFT_RULE_TABLE = {
    (True, False): "R501",
    (False, True): "R601",
    (True, True): "R602",
}

LEVEL_CN = {"STRONG": "强匹配", "NORMAL": "普通匹配", "WEAK": "弱匹配"}
NOT_MATCH_CN = "不能匹配"
ALLOW_ASSIGN_LEVELS = frozenset({"STRONG", "NORMAL", "WEAK"})

# 参与三态比较的核心字段；顺序即 U04 冲突字段名的输出顺序
CONFLICT_FIELDS = ("layer", "subject", "major_group", "code")

# 哪些核心字段是硬冲突：True 时双方都有值且不同 -> 直接淘汰（U04）；
# False 时只作为未命中（不提高 key_code）并记录“xxx不一致”风险，不淘汰候选。
# 当前四个字段默认都不是硬冲突；需要在 config 中显式改 True 才淘汰。
HARD_CONFLICT = {
    "layer": False,
    "subject": False,
    "major_group": False,
    "code": False,
}

REASON_CODES = {
    "U01": "无候选B",
    "U02": "多候选无法区分",
    "U03": "B资源冲突",
    "U04": "硬字段冲突",
    "U05": "证据不足",
}

# 顺序即输出顺序
RISK_TAGS = (
    "选科缺失",
    "专业组缺失",
    "批次缺失",
    "批次不一致",
    "专业代码缺失",
    "备注不一致",
    "层次缺失",
    "辅助证据消歧",
    "使用次优候选",
    "层次不一致",
    "选科不一致",
    "专业组不一致",
    "专业代码不一致",
    "备注关键词多余",
    "备注关键词不一致",
    "弱匹配补救",
)
RISK_BY_STATE = {
    ("layer", "UNKNOWN"): "层次缺失",
    ("subject", "UNKNOWN"): "选科缺失",
    ("major_group", "UNKNOWN"): "专业组缺失",
    ("code", "UNKNOWN"): "专业代码缺失",
    ("batch", "UNKNOWN"): "批次缺失",
    ("batch", "CONFLICT"): "批次不一致",
    ("remark", "CONFLICT"): "备注不一致",
    ("layer", "CONFLICT"): "层次不一致",
    ("subject", "CONFLICT"): "选科不一致",
    ("major_group", "CONFLICT"): "专业组不一致",
    ("code", "CONFLICT"): "专业代码不一致",
    ("major_tag", "UNKNOWN"): "备注关键词多余",
    ("major_tag", "CONFLICT"): "备注关键词不一致",
}
RISK_WEAK = "辅助证据消歧"
RISK_FALLBACK = "使用次优候选"

# 位 0、位 1；字符串输出按此顺序用“|”连接
RECALL_SOURCES = ("基础召回", "映射召回")

FIELD_LABELS = {
    "layer": "层次",
    "subject": "选科",
    "major_group": "专业组",
    "code": "专业代码",
    "remark": "备注",
    "batch": "标准批次",
    "school_id": "院校ID",
    "province": "省份",
    "major_name": "专业名称",
    "major_base": "专业名称主体",
    "major_tag": "备注关键词",
}

U03_MARK = "B_MULTI_USED"

# ============================================================
# 7. 配置校验
# ============================================================

_GLOSSARY_RISK_TAGS = (
    "选科缺失", "专业组缺失", "批次缺失", "批次不一致", "专业代码缺失",
    "备注不一致", "层次缺失", "辅助证据消歧", "使用次优候选",
    "层次不一致", "选科不一致", "专业组不一致", "专业代码不一致",
    "备注关键词多余", "备注关键词不一致", "弱匹配补救",
)
_MAX_REMARK_TAGS = 256  # 掩码用 Python int，不受 int64 位数限制
_FORBIDDEN_SEPARATORS = frozenset({"(", ")", "（", "）", "-"})


def validate_config():
    """基础配置校验；失败抛 ConfigError，消息指明出错项。读取模块全局变量。"""
    from errors import ConfigError

    g = globals()

    expected = {"U01", "U02", "U03", "U04", "U05"}
    if set(g["REASON_CODES"]) != expected:
        raise ConfigError(
            f"REASON_CODES 键必须恰为 U01–U05，实际为 {sorted(g['REASON_CODES'])}")

    risk_tags = list(g["RISK_TAGS"])
    dup = sorted({t for t in risk_tags if risk_tags.count(t) > 1})
    if dup:
        raise ConfigError(f"RISK_TAGS 存在重复项: {dup}")
    missing = [t for t in _GLOSSARY_RISK_TAGS if t not in risk_tags]
    if missing:
        raise ConfigError(f"RISK_TAGS 缺少术语表标签: {missing}")

    for state, tag in g["RISK_BY_STATE"].items():
        if tag not in risk_tags:
            raise ConfigError(f"RISK_BY_STATE[{state!r}] 的值 {tag!r} 不在 RISK_TAGS 中")

    bad_levels = set(g["ALLOW_ASSIGN_LEVELS"]) - {"STRONG", "NORMAL", "WEAK"}
    if bad_levels:
        raise ConfigError(f"ALLOW_ASSIGN_LEVELS 含非法等级: {sorted(bad_levels)}")

    conflict_fields = tuple(g["CONFLICT_FIELDS"])
    if len(conflict_fields) != len(set(conflict_fields)):
        raise ConfigError(f"CONFLICT_FIELDS 存在重复项: {conflict_fields}")
    if set(conflict_fields) != {"layer", "subject", "major_group", "code"}:
        raise ConfigError(
            f"CONFLICT_FIELDS 必须恰为 layer/subject/major_group/code，实际为 {conflict_fields}")

    hard_conflict = g["HARD_CONFLICT"]
    if set(hard_conflict) != set(conflict_fields):
        raise ConfigError(
            f"HARD_CONFLICT 的键必须与 CONFLICT_FIELDS 一致，实际为 {sorted(hard_conflict)}")
    for field, is_hard in hard_conflict.items():
        if not isinstance(is_hard, bool):
            raise ConfigError(
                f"HARD_CONFLICT[{field!r}] 必须是 bool，实际为 {type(is_hard).__name__}")

    subject_map = g["SUBJECT_MAP"]
    for src, target in subject_map.items():
        if subject_map.get(target) != target:
            raise ConfigError(
                f"SUBJECT_MAP 不幂等: {src!r} -> {target!r}，"
                f"但 {target!r} 再映射为 {subject_map.get(target)!r}")

    tags = g["REMARK_TAGS"]
    if len(tags) > _MAX_REMARK_TAGS:
        raise ConfigError(f"备注标签数 {len(tags)} 超过上限 {_MAX_REMARK_TAGS}")

    sep = g["REMARK_SEPARATOR"]
    if not isinstance(sep, str) or not sep:
        raise ConfigError("REMARK_SEPARATOR 不能为空")
    if sep.strip() == "":
        raise ConfigError(f"REMARK_SEPARATOR 不能是空白字符: {sep!r}")
    if sep in _FORBIDDEN_SEPARATORS:
        raise ConfigError(f"REMARK_SEPARATOR 不能是括号或 '-': {sep!r}")
    for tag, words in tags.items():
        for w in words:
            if sep in w:
                raise ConfigError(
                    f"REMARK_SEPARATOR {sep!r} 出现在标签 {tag!r} 的关键词 {w!r} 中")
    for p in g["NEGATION_PREFIXES"]:
        if sep in p:
            raise ConfigError(f"REMARK_SEPARATOR {sep!r} 出现在否定前缀 {p!r} 中")

    # ---- 依赖 normalizer 的校验（函数内导入，避免循环导入）----
    import normalizer

    _validate_remark_words_normalized(normalizer, g)
    _validate_batch_alias(normalizer, g["BATCH_ALIAS"])

    # 规则表校验（延迟导入，避免 config 与 rule_engine 循环导入）
    import rule_engine

    rule_engine.validate_rules()


def _validate_remark_words_normalized(normalizer, g):
    """备注关键词与否定前缀经 norm_text 后必须非空，且不含备注分隔符（Req 3.10、3.16）。"""
    from errors import ConfigError

    sep = g["REMARK_SEPARATOR"]
    seps = {sep, normalizer.norm_text(sep)} - {None}

    def check(kind, owner, word):
        norm = normalizer.norm_text(word)
        if norm is None:
            raise ConfigError(f"{kind} {word!r}{owner} 经标准化后为空")
        for s in seps:
            if s in norm:
                raise ConfigError(
                    f"REMARK_SEPARATOR {sep!r} 出现在{kind} {word!r}{owner} 标准化后的 {norm!r} 中")

    for tag, words in g["REMARK_TAGS"].items():
        for w in words:
            check("备注关键词", f"（标签 {tag!r}）", w)
    for p in g["NEGATION_PREFIXES"]:
        check("否定前缀", "", p)


def _validate_batch_alias(normalizer, raw):
    """批次别名表校验（Req 3.20）：

    1. 同省份内键清洗（步骤①–④）后相同而值不同 -> ConfigError；
    2. 每个值在其键的省份下（"*" 键在 "*" 及别名表出现的每个省份下），
       经步骤①–⑤、成分抽取、合成后必须等于自身 -> ConfigError。
    """
    from errors import ConfigError

    # 1) 冲突检测
    seen = {}  # (省份, 清洗后主体) -> (首个原键文本, 值)
    entries = []  # (省份, 原键文本, 值)
    for (province, text), value in raw.items():
        entries.append((province, text, value))
        body, _ = normalizer.clean_batch_steps_1_4(text)
        if body is None:
            continue
        key = (province, body)
        if key in seen and seen[key][1] != value:
            first_text, first_value = seen[key]
            raise ConfigError(
                f"BATCH_ALIAS 冲突: 省份 {province!r} 下键 {first_text!r} 与 {text!r} "
                f"清洗后同为 {body!r}，但值不同: {first_value!r} vs {value!r}")
        seen.setdefault(key, (text, value))

    # 2) 值的自洽检测
    table = normalizer.build_alias_table(raw)
    named_provinces = sorted({p for p, _, _ in entries if p != "*"})
    for province, text, value in entries:
        targets = ["*"] + named_provinces if province == "*" else [province]
        for target in targets:
            body, _ = normalizer.clean_batch(value, target, table)
            name = normalizer.compose_batch_name(normalizer.extract_components(body))
            if name != value:
                raise ConfigError(
                    f"BATCH_ALIAS 值不自洽: 键 ({province!r}, {text!r}) 的值 {value!r} "
                    f"在省份 {target!r} 下处理后为 {name!r}，不等于自身")
