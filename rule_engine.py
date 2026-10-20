# -*- coding: utf-8 -*-
"""Rule_Engine：规则表校验、核心规则/软规则判定、等级查询、风险掩码。

规则表、核心/软规则映射、风险标签全部来自 config，本模块不定义同类常量。
等级只用于输出与赋分资格，排序与择优只使用 key_code（Req 7.8）。
"""
import re

import numpy as np

import config
from errors import ConfigError

_RULE_ID_RE = re.compile(r"^R\d{3}$")
_LEVELS = frozenset({"STRONG", "NORMAL", "WEAK"})

# 初始 8 个编号及其等级（Req 7.4、7.10）
_INITIAL_RULES = {
    "R101": "NORMAL",
    "R201": "STRONG",
    "R301": "STRONG",
    "R302": "STRONG",
    "R401": "STRONG",
    "R501": "WEAK",
    "R601": "WEAK",
    "R602": "WEAK",
}


def _bool_combos(n):
    return {
        tuple(bool((i >> (n - 1 - k)) & 1) for k in range(n))
        for i in range(2 ** n)
    }


def validate_rules():
    """校验规则表；失败抛 ConfigError，消息含问题编号与问题类型。"""
    table = list(config.RULE_TABLE)

    seen = {}
    for entry in table:
        rid, level = entry[0], entry[2]
        if not isinstance(rid, str) or not _RULE_ID_RE.match(rid):
            raise ConfigError(f"规则编号 {rid!r} 格式错误：须符合 ^R\\d{{3}}$")
        if rid in seen:
            raise ConfigError(f"规则编号 {rid} 重复")
        if level not in _LEVELS:
            raise ConfigError(f"规则编号 {rid} 等级非法: {level!r}")
        seen[rid] = level

    for rid, level in _INITIAL_RULES.items():
        if rid not in seen:
            raise ConfigError(f"规则编号 {rid} 缺失：初始编号必须齐全")
        if seen[rid] != level:
            raise ConfigError(
                f"规则编号 {rid} 等级不符：应为 {level}，实际为 {seen[rid]}")

    core = config.CORE_RULE_TABLE
    if set(core) != _bool_combos(3):
        raise ConfigError(
            "CORE_RULE_TABLE 覆盖不完整：必须恰好覆盖 (code, subject, major_group) 的 8 种组合")
    for combo, rid in core.items():
        if rid not in seen:
            raise ConfigError(f"规则编号 {rid} 未在规则表中（CORE_RULE_TABLE{combo}）")

    soft = config.SOFT_RULE_TABLE
    if set(soft) != {(True, False), (False, True), (True, True)}:
        raise ConfigError(
            "SOFT_RULE_TABLE 覆盖不完整：必须恰好覆盖 (备注更优, 标准批次更优) 的 3 种组合")
    for combo, rid in soft.items():
        if rid not in seen:
            raise ConfigError(f"规则编号 {rid} 未在规则表中（SOFT_RULE_TABLE{combo}）")


def _levels():
    return {e[0]: e[2] for e in config.RULE_TABLE}


def level_of(rule):
    """规则编号 -> 等级；编号不在规则表时抛 ConfigError（Req 7.6、14.9）。"""
    level = _levels().get(rule)
    if level is None:
        raise ConfigError(f"规则编号 {rule!r} 未在规则表中")
    return level


def core_rule(evidence):
    """evidence: {"code","subject","major_group"} -> 核心规则编号。"""
    key = (bool(evidence.get("code")), bool(evidence.get("subject")),
           bool(evidence.get("major_group")))
    return config.CORE_RULE_TABLE[key]


def core_rule_vec(code, subject, group):
    """向量化核心规则：8 元素查表，下标 4*code+2*subject+group。返回 object 数组。"""
    lut = np.empty(8, dtype=object)
    for (c, s, g), rid in config.CORE_RULE_TABLE.items():
        lut[4 * int(c) + 2 * int(s) + int(g)] = rid
    idx = (4 * np.asarray(code, dtype=bool).astype(np.int64)
           + 2 * np.asarray(subject, dtype=bool).astype(np.int64)
           + np.asarray(group, dtype=bool).astype(np.int64))
    return lut[idx]


def soft_rule(remark_better, batch_better):
    """软规则编号；两者都不更优时没有规则，抛 KeyError 由调用方避免。"""
    return config.SOFT_RULE_TABLE[(bool(remark_better), bool(batch_better))]


def risk_bit(tag):
    """风险标签 -> 位掩码（位序 = RISK_TAGS 下标）。"""
    return 1 << tuple(config.RISK_TAGS).index(tag)


def risk_mask_vec(states):
    """states: 字段名 -> int8 三态数组（1 MATCH / 0 UNKNOWN / -1 CONFLICT）。

    按 RISK_BY_STATE 生成 int64 风险掩码；缺少的字段忽略。
    """
    code_of = {"UNKNOWN": 0, "CONFLICT": -1, "MATCH": 1}
    n = None
    for arr in states.values():
        n = len(arr)
        break
    mask = np.zeros(n or 0, dtype=np.int64)
    for (field, state), tag in config.RISK_BY_STATE.items():
        arr = states.get(field)
        if arr is None:
            continue
        mask |= np.where(np.asarray(arr) == code_of[state], risk_bit(tag), 0).astype(np.int64)
    return mask


def risk_tags(mask):
    """掩码 -> 按 RISK_TAGS 顺序展开的标签元组（去重、顺序固定）。"""
    mask = int(mask)
    return tuple(t for i, t in enumerate(config.RISK_TAGS) if mask >> i & 1)
