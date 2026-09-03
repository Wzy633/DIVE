#!/usr/bin/env python3
"""Generate STVG-style Chinese text pairs (FIXED format) from marker metadata.

This script is the "fixed.jsonl-first" generator that matches the current
`data/DIOR_marked/ours/epoch_26/val/text_pairs_stvg_style.fixed.jsonl` style:

- `expected_answer`: "<answer>k</answer>" (numeric-only)
- `expected_response_template`: ends with "<answer>k</answer>"
- Question templates wording matches the existing fixed file:
  - single:   "请定位图中唯一的{cls}，并给出其目标ID。"
  - lr rank:  "图中存在多个{cls}。请返回从左到右第N个{cls}的目标ID。"
  - tb rank:  "请在所有{cls}中，找出从上到下第N个目标，并输出其ID。"
  - area:     "在图中所有{cls}里，面积最大的/最小的目标是哪一个？请输出目标ID。"

Input:
  - marker_meta_lae_stvg.json (list of image records)
Output:
  - JSONL with fields:
    image_file, question, target_marker_id, target_class, bbox_xyxy, source,
    expected_answer, expected_response_template
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# Allow importing shared helpers from parent scripts dir (dior_label_zh, text_pair_cn_ordinal).
_PARENT_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_PARENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_SCRIPTS_DIR))

from dior_label_zh import dior_class_bilingual  # noqa: E402
from text_pair_cn_ordinal import cn_ordinal_rank  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed-format text pairs from marker meta.")
    p.add_argument("--marker-meta", type=str, required=True, help="marker_meta_lae_stvg.json")
    p.add_argument("--out-jsonl", type=str, required=True, help="Output .fixed.jsonl path")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument(
        "--min-rank-rel-diff",
        type=float,
        default=0.03,
        help="Minimum relative difference to generate lr/tb rank questions (default: 0.03).",
    )
    p.add_argument(
        "--min-area-rel-diff",
        type=float,
        default=0.10,
        help="Minimum relative area gap to generate area max/min questions (default: 0.10).",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _area_xyxy(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _sorted_lr(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: (x["centroid_xy"][0], x["centroid_xy"][1]))


def _sorted_tb(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: (x["centroid_xy"][1], x["centroid_xy"][0]))


def _rank_by_area(group: List[Dict[str, Any]], reverse: bool) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: _area_xyxy(x["bbox_xyxy"]), reverse=reverse)


def _min_adjacent_rel_gap(values: List[float], denom: float) -> float:
    if len(values) < 2 or denom <= 0:
        return 0.0
    vals = sorted(values)
    gaps = [abs(vals[i + 1] - vals[i]) / denom for i in range(len(vals) - 1)]
    return min(gaps) if gaps else 0.0


def _expected(mid: int) -> str:
    return f"<answer>{mid}</answer>"


def build_samples_for_class_fixed(
    file_name: str,
    cls_name: str,
    group: List[Dict[str, Any]],
    min_rank_rel_diff: float,
    min_area_rel_diff: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(group)
    if n == 0:
        return out

    ql = dior_class_bilingual(cls_name)

    # 1) 单实例
    if n == 1:
        ins = group[0]
        mid = int(ins["marker_id"])
        out.append(
            {
                "image_file": file_name,
                "question": f"请定位图中唯一的{ql}，并给出其目标ID。",
                "target_marker_id": mid,
                "target_class": cls_name,
                "bbox_xyxy": ins["bbox_xyxy"],
                "expected_answer": _expected(mid),
                "expected_response_template": (
                    "<think>根据图中目标的空间位置与描述约束进行匹配。</think>" + _expected(mid)
                ),
                "source": "stvg_style_single",
            }
        )
        return out

    # 2) 左到右排序（相邻中心 x 相对差过小则跳过）
    lr = _sorted_lr(group)
    xs = [ins["centroid_xy"][0] for ins in group]
    lr_gap = _min_adjacent_rel_gap(xs, 800.0)
    if lr_gap > min_rank_rel_diff:
        for rank, ins in enumerate(lr, start=1):
            cn = cn_ordinal_rank(rank)  # 第一个/第二个...
            mid = int(ins["marker_id"])
            out.append(
                {
                    "image_file": file_name,
                    "question": f"图中存在多个{ql}。请返回从左到右{cn}{ql}的目标ID。",
                    "target_marker_id": mid,
                    "target_class": cls_name,
                    "bbox_xyxy": ins["bbox_xyxy"],
                    "expected_answer": _expected(mid),
                    "expected_response_template": (
                        f"<think>先对同类目标按左右顺序排序，再选择{cn}。</think>" + _expected(mid)
                    ),
                    "source": "stvg_style_lr_rank",
                }
            )

    # 3) 上到下排序（相邻中心 y 相对差过小则跳过）
    tb = _sorted_tb(group)
    ys = [ins["centroid_xy"][1] for ins in group]
    tb_gap = _min_adjacent_rel_gap(ys, 800.0)
    if tb_gap > min_rank_rel_diff:
        for rank, ins in enumerate(tb, start=1):
            cn = cn_ordinal_rank(rank)
            mid = int(ins["marker_id"])
            out.append(
                {
                    "image_file": file_name,
                    "question": f"请在所有{ql}中，找出从上到下{cn}目标，并输出其ID。",
                    "target_marker_id": mid,
                    "target_class": cls_name,
                    "bbox_xyxy": ins["bbox_xyxy"],
                    "expected_answer": _expected(mid),
                    "expected_response_template": (
                        f"<think>先对同类目标按上下顺序排序，再选择{cn}。</think>" + _expected(mid)
                    ),
                    "source": "stvg_style_tb_rank",
                }
            )

    # 4) 面积最大/最小（相对差<=阈值则跳过）
    max_ins = _rank_by_area(group, reverse=True)[0]
    min_ins = _rank_by_area(group, reverse=False)[0]
    max_area = _area_xyxy(max_ins["bbox_xyxy"])
    min_area = _area_xyxy(min_ins["bbox_xyxy"])
    area_rel_diff = (max_area - min_area) / max_area if max_area > 0 else 0.0
    if area_rel_diff > min_area_rel_diff:
        max_mid = int(max_ins["marker_id"])
        min_mid = int(min_ins["marker_id"])
        out.append(
            {
                "image_file": file_name,
                "question": f"在图中所有{ql}里，面积最大的目标是哪一个？请输出目标ID。",
                "target_marker_id": max_mid,
                "target_class": cls_name,
                "bbox_xyxy": max_ins["bbox_xyxy"],
                "expected_answer": _expected(max_mid),
                "expected_response_template": (
                    "<think>比较同类目标的面积，选择面积最大的实例。</think>" + _expected(max_mid)
                ),
                "source": "stvg_style_area_max",
            }
        )
        out.append(
            {
                "image_file": file_name,
                "question": f"在图中所有{ql}里，面积最小的目标是哪一个？请输出目标ID。",
                "target_marker_id": min_mid,
                "target_class": cls_name,
                "bbox_xyxy": min_ins["bbox_xyxy"],
                "expected_answer": _expected(min_mid),
                "expected_response_template": (
                    "<think>比较同类目标的面积，选择面积最小的实例。</think>" + _expected(min_mid)
                ),
                "source": "stvg_style_area_min",
            }
        )

    return out


def main() -> None:
    args = parse_args()
    marker_path = Path(args.marker_meta)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(marker_path)
    if args.max_images and args.max_images > 0:
        data = data[: args.max_images]

    n_in_images = 0
    n_out = 0
    with out_path.open("w", encoding="utf-8") as w:
        for rec in data:
            file_name = str(rec.get("file_name", "") or "")
            instances = rec.get("instances", []) or []
            if not file_name or not isinstance(instances, list):
                continue
            n_in_images += 1

            # Keep insertion order of classes as they first appear in instances list
            by_cls: Dict[str, List[Dict[str, Any]]] = {}
            for ins in instances:
                if not isinstance(ins, dict):
                    continue
                cls = str(ins.get("label_name", "") or "")
                if not cls:
                    continue
                by_cls.setdefault(cls, []).append(ins)

            for cls_name, group in by_cls.items():
                samples = build_samples_for_class_fixed(
                    file_name=file_name,
                    cls_name=cls_name,
                    group=group,
                    min_rank_rel_diff=float(args.min_rank_rel_diff),
                    min_area_rel_diff=float(args.min_area_rel_diff),
                )
                for s in samples:
                    w.write(json.dumps(s, ensure_ascii=False) + "\n")
                    n_out += 1

    print(f"[DONE] images={n_in_images} lines={n_out} -> {out_path}")


if __name__ == "__main__":
    main()

