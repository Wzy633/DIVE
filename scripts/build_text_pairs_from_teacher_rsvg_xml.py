#!/usr/bin/env python3
"""
Build text-pairs JSONL for DIOR-RSVG using XML descriptions, aligned to teacher marker IDs.

Inputs:
  - --xml-dir: DIOR_RSVG/_xml_{split} directory (each xml contains filename, object{name,bndbox,description})
  - --teacher-marker-dir: teacher marker output directory that contains per-image marker jsons
        e.g. LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/val/vis_pred_marker
        where each <stem>.json contains {"image_name": "<stem>", "markers":[{marker_id,bbox,class,score,...}, ...]}

Output (JSONL, one line per XML object):
  - image_file, question, target_marker_id, target_class, bbox_xyxy, source,
    expected_answer, expected_response_template,
    gt_bbox_xyxy, match_iou, match_policy

Matching policy:
  - Prefer matching teacher markers by class equality (marker.class == xml.name).
  - Choose marker with highest IoU to the XML GT bbox.
  - If best IoU < --min-match-iou, mark as unmatched and skip unless --keep-unmatched is set.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml-dir", type=str, required=True)
    p.add_argument("--teacher-marker-dir", type=str, required=True)
    p.add_argument("--out-jsonl", type=str, required=True)
    p.add_argument("--min-match-iou", type=float, default=0.5)
    p.add_argument("--keep-unmatched", action="store_true", help="Keep unmatched objects with target_marker_id=-1.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N xml files (sorted).")
    return p.parse_args()


def _xyxy_from_xml_box(b: ET.Element) -> Tuple[float, float, float, float]:
    def _get(tag: str) -> float:
        x = b.find(tag)
        if x is None or x.text is None:
            raise ValueError(f"missing <{tag}> in bndbox")
        return float(x.text.strip())

    x1 = _get("xmin")
    y1 = _get("ymin")
    x2 = _get("xmax")
    y2 = _get("ymax")
    return (x1, y1, x2, y2)


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def load_teacher_markers(marker_json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(marker_json_path.read_text(encoding="utf-8"))
    markers = data.get("markers", [])
    if not isinstance(markers, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in markers:
        if not isinstance(m, dict):
            continue
        if "marker_id" not in m or "bbox" not in m:
            continue
        out.append(m)
    return out


def match_marker_id(
    gt_box: Tuple[float, float, float, float],
    gt_class: str,
    markers: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Tuple[float, float, float, float]], float, str]:
    """
    Returns: (marker_id, marker_bbox, best_iou, policy)
    policy:
      - "class_filtered": candidates restricted to same class
      - "no_class_match": fallback to all markers
      - "no_markers": no candidates
    """
    gt_class_norm = gt_class.strip().lower()
    same_cls = [m for m in markers if str(m.get("class", "")).strip().lower() == gt_class_norm]
    policy = "class_filtered"
    cand = same_cls
    if not cand:
        policy = "no_class_match"
        cand = markers
    if not cand:
        return None, None, 0.0, "no_markers"
    best = (-1.0, None, None)  # (iou, id, bbox)
    for m in cand:
        try:
            mid = int(m["marker_id"])
            b = m["bbox"]
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            biou = iou_xyxy(gt_box, (bx1, by1, bx2, by2))
            if biou > best[0]:
                best = (biou, mid, (bx1, by1, bx2, by2))
        except Exception:
            continue
    if best[1] is None or best[2] is None:
        return None, None, 0.0, policy
    return int(best[1]), best[2], float(best[0]), policy


def build_question(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        d = "(no description)"
    return f"请根据描述定位目标：{d}。请输出该目标的ID。"


def main() -> None:
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    marker_dir = Path(args.teacher_marker_dir)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xmls = sorted(xml_dir.glob("*.xml"))
    if args.limit and args.limit > 0:
        xmls = xmls[: int(args.limit)]

    n_total_obj = 0
    n_matched = 0
    n_unmatched = 0
    n_missing_marker_json = 0

    with out_path.open("w", encoding="utf-8") as w:
        for xp in xmls:
            tree = ET.parse(str(xp))
            root = tree.getroot()
            fn_elem = root.find("filename")
            image_file = fn_elem.text.strip() if fn_elem is not None and fn_elem.text else ""
            if not image_file:
                continue
            stem = Path(image_file).stem
            marker_json = marker_dir / f"{stem}.json"
            markers: List[Dict[str, Any]] = []
            if marker_json.is_file():
                markers = load_teacher_markers(marker_json)
            else:
                n_missing_marker_json += 1

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                box_elem = obj.find("bndbox")
                desc_elem = obj.find("description")
                if name_elem is None or name_elem.text is None or box_elem is None:
                    continue
                gt_class = name_elem.text.strip()
                if not gt_class:
                    continue
                try:
                    gt_box = _xyxy_from_xml_box(box_elem)
                except Exception:
                    continue
                desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""

                n_total_obj += 1
                mid, mbox, biou, policy = match_marker_id(gt_box, gt_class, markers)
                matched = mid is not None and mbox is not None and biou >= float(args.min_match_iou)
                if matched:
                    n_matched += 1
                    target_mid = int(mid)
                    bbox_xyxy = [float(x) for x in mbox]
                else:
                    n_unmatched += 1
                    if not args.keep_unmatched:
                        continue
                    target_mid = -1
                    bbox_xyxy = [math.nan, math.nan, math.nan, math.nan]

                rec: Dict[str, Any] = {
                    "image_file": image_file,
                    "question": build_question(desc),
                    "target_marker_id": target_mid,
                    "target_class": gt_class,
                    "bbox_xyxy": bbox_xyxy,
                    "gt_bbox_xyxy": [float(x) for x in gt_box],
                    "match_iou": float(biou),
                    "match_policy": policy,
                    "source": "rsvg_xml_description_teacher_marker",
                    "expected_answer": f"<answer>{target_mid}</answer>",
                    "expected_response_template": (
                        "<think>根据描述定位目标，并在图中找到对应编号。</think>"
                        f"<answer>{target_mid}</answer>"
                    ),
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("Build text pairs from DIOR-RSVG XML (description) + teacher markers")
    print("=" * 70)
    print(f"xml_files={len(xmls)}")
    print(f"objects_total={n_total_obj}")
    print(f"matched_iou>={float(args.min_match_iou):.2f}: {n_matched}")
    print(f"unmatched: {n_unmatched}")
    print(f"missing_marker_json: {n_missing_marker_json}")
    print(f"out_jsonl={out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

