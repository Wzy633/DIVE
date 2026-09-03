"""DIOR 检测类别英文名 → 题干中「英文（中文）」并列展示，减轻 VLM 对类名的歧义。"""

from __future__ import annotations

# 与 marker_meta 中 label_name 一致（val 全量中出现的 20 类）
DIOR_CLASS_ZH: dict[str, str] = {
    "airplane": "飞机",
    "airport": "机场",
    "baseballfield": "棒球场",
    "basketballcourt": "篮球场",
    "bridge": "桥梁",
    "chimney": "烟囱",
    "dam": "水坝",
    "expressway service area": "高速公路服务区",
    "expressway toll station": "高速公路收费站",
    "golffield": "高尔夫球场",
    "groundtrackfield": "田径场",
    "harbor": "港口",
    "overpass": "立交桥",
    "ship": "舰船",
    "stadium": "体育场",
    "storagetank": "储罐",
    "tenniscourt": "网球场",
    "trainstation": "火车站",
    "vehicle": "车辆",
    "windmill": "风车",
}


def dior_class_bilingual(cls_name: str) -> str:
    """返回「英文名（中文）」；未知类名则原样返回。"""
    zh = DIOR_CLASS_ZH.get(cls_name.strip())
    if not zh:
        return cls_name
    return f"{cls_name}（{zh}）"


def migrate_question_bilingual(q: str) -> str:
    """将题干中的英文类名替换为「英文（中文）」；已带「（」的片段不重复包裹。"""
    out = q
    for cls in sorted(DIOR_CLASS_ZH.keys(), key=lambda k: (len(k), k), reverse=True):
        rep = dior_class_bilingual(cls)
        if rep == cls:
            continue
        i = 0
        parts: list[str] = []
        while i < len(out):
            if out.startswith(cls, i) and (
                i + len(cls) == len(out) or out[i + len(cls)] != "（"
            ):
                parts.append(rep)
                i += len(cls)
            else:
                parts.append(out[i])
                i += 1
        out = "".join(parts)
    return out
