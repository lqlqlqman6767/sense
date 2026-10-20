# -*- coding: utf-8 -*-
"""REVIEW U03 的保守补救：只在未用同主体 B 且竞争 A 有唯一字段时，转 WEAK。"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import normalizer as nz

B_ID_RE = re.compile(r"B_\d{8}")
A_ROWS_RE = re.compile(r"A_ROWS=([0-9,]+)")
NOISE_RE = re.compile(
    r"(元|年|校区|办学地点|地址|电话|网址|邮编|语种|口试|选科要求|色盲|色弱|体检|面试|"
    r"政治考察|公办|民办|普通类|普通组|校本部|计划|人数|学费|住宿|非定向|定向|只招|"
    r"不招|男|女|身高|视力|科类|批次)"
)


def _parts(x):
    base_name, ps = nz.split_major_name(x)
    return [p for p in ps if p and len(p) >= 2 and not NOISE_RE.search(p)
            and not re.fullmatch(r"[0-9.]+", p)]


def _txt(x):
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return str(x).strip()


def _unique_a(competitors, b, b_remark=""):
    """返回 (A_std 行, 字段名) 或 (None, "")。"""
    if not competitors:
        return None, ""
    b_group = _txt(b.get("major_group"))
    b_code = _txt(b.get("major_code"))
    b_subject = _txt(b.get("subject"))
    b_layer = _txt(b.get("layer"))
    b_tag = int(b.get("remark_mask") or 0)
    b_name_tag = int(b.get("major_name_mask") or 0)
    b_parts = _parts(b_remark)

    def hits(field):
        out = []
        for a in competitors:
            if field == "group" and b_group and _txt(a.get("major_group")) == b_group:
                out.append(a)
            elif field == "code" and b_code and _txt(a.get("major_code")) == b_code:
                out.append(a)
            elif field == "subject" and b_subject and _txt(a.get("subject")) == b_subject:
                out.append(a)
            elif field == "layer" and b_layer and _txt(a.get("layer")) == b_layer:
                out.append(a)
            elif field == "tag" and b_tag and (int(a.get("remark_mask") or 0) & b_tag):
                out.append(a)
            elif field == "name_tag" and b_name_tag and (int(a.get("major_name_mask") or 0) & b_name_tag):
                out.append(a)
            elif field == "remark" and b_parts:
                ap = _parts(a.get("_remark", ""))
                if any(any(p in q or q in p for q in b_parts) for p in ap):
                    out.append(a)
        return out

    for field in ("group", "code", "subject", "layer", "tag", "name_tag", "remark"):
        hs = hits(field)
        if len(hs) == 1:
            return hs[0], field
    return None, ""


def apply_review_fallback(final, A_std, B_std, A_raw=None, B_raw=None):
    """把可唯一的 U03 改为 WEAK；不满足条件的保持 REVIEW。"""
    if final.empty or B_std.empty:
        return final, 0

    out = final.copy()
    A_std = A_std.copy()
    B_std = B_std.copy()
    if A_raw is not None and "major_remark" in A_raw.columns:
        if len(A_raw) == len(A_std):
            A_std["_remark"] = A_raw["major_remark"].reset_index(drop=True)
        elif "a_row" in A_raw.columns:
            rem = {int(r["a_row"]): r["major_remark"] for _, r in A_raw.iterrows()}
            A_std["_remark"] = A_std["a_row"].map(lambda x: rem.get(int(x), "") if pd.notna(x) else "")
        else:
            A_std["_remark"] = ""
    else:
        A_std["_remark"] = ""
    if B_raw is not None and "b_id" in B_raw.columns and "专业备注" in B_raw.columns:
        rem_b = {str(r["b_id"]): r["专业备注"] for _, r in B_raw.iterrows()}
        B_std["_remark"] = B_std["b_id"].map(lambda x: rem_b.get(str(x), ""))
    else:
        B_std["_remark"] = ""

    used = set(out.loc[out["status"].eq("MATCH"), "bi"].dropna().astype(str))
    b_by_id = {str(r["b_id"]): r for _, r in B_std.iterrows()}
    a_by_row = {int(r["a_row"]): r for _, r in A_std.iterrows()}
    a_pos_by_row = {int(r["a_row"]): int(r["a_pos"]) for _, r in A_std.iterrows()}

    # 同一 school/province/base 的 B
    b_key = defaultdict(list)
    for _, r in B_std.iterrows():
        b_key[(r["school_id"], r["province"], r["major_base"])].append(r)

    changed = 0
    for idx, row in out.iterrows():
        if row.get("status") != "REVIEW" or row.get("reason") != "U03":
            continue
        a = A_std.iloc[int(row["a_pos"])]
        key = (a["school_id"], a["province"], a["major_base"])
        candidates = [b for b in b_key.get(key, []) if str(b["b_id"]) not in used]
        if not candidates:
            continue

        detail = _txt(row.get("reason_detail"))
        ref_ids = set(B_ID_RE.findall(detail))
        ref_used = any(str(x) in used for x in ref_ids)
        # B 已占用且没有剩余同主体：直接跳过（用户原则）
        if ref_used and not candidates:
            continue

        competitors = []
        for r in A_ROWS_RE.findall(detail):
            for x in r.split(","):
                if not x:
                    continue
                pos = int(x)
                if pos in a_by_row:
                    competitors.append(a_by_row[pos])
        if not competitors:
            competitors = [a]

        assigned = False
        for b in candidates:
            target_a, field = _unique_a(competitors, b, b.get("_remark", ""))
            if target_a is None:
                continue
            target_pos = a_pos_by_row.get(int(target_a["a_row"]))
            if target_pos is None:
                continue
            target = out[out["a_pos"].eq(target_pos)]
            if target.empty or target.iloc[0]["status"] == "MATCH":
                continue
            i = target.index[0]
            out.at[i, "status"] = "MATCH"
            out.at[i, "bi"] = b["b_id"]
            out.at[i, "rule"] = "R501"
            out.at[i, "level"] = "WEAK"
            out.at[i, "reason"] = None
            out.at[i, "reason_detail"] = f"U03补救：{field}唯一选中A"
            out.at[i, "risk"] = ("辅助证据消歧", "弱匹配补救")
            out.at[i, "fallback"] = True
            used.add(str(b["b_id"]))
            changed += 1
            assigned = True
            break
        if assigned:
            continue

    return out, changed
