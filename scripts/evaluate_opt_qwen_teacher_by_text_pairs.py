#!/usr/bin/env python3
"""Evaluate Qwen predictions on OPT-RSVG teacher markers by text pairs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SEP = "-" * 70
ANSWER_TAG_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)
ANSWER_ALT_RE = re.compile(r"answer[:\s]+(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OPT-RSVG Qwen teacher-marker predictions by text-pairs.")
    p.add_argument("--pred-file", type=str, required=True)
    p.add_argument("--text-pairs-jsonl", type=str, required=True)
    p.add_argument("--marker-json-dir", type=str, required=True)
    p.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--exclude-pred-ids", type=str, default="0")
    return p.parse_args()


def parse_exclude_pred_ids(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def extract_predicted_id(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = ANSWER_TAG_RE.search(text)
    if m:
        return int(m.group(1))
    m = ANSWER_ALT_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    inter, union = intersection_union_xyxy(a, b)
    return float(inter / union) if union > 0 else 0.0


def intersection_union_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        inter = 0.0
    else:
        inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter), float(max(0.0, union))


def box_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_pred_file(pred_path: Path) -> List[Tuple[str, str]]:
    txt = pred_path.read_text(encoding="utf-8", errors="ignore")
    parts = txt.split(SEP)
    out: List[Tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
        if not lines:
            continue
        out.append((lines[0], "\n".join(lines[1:]).strip()))
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_marker_bbox_map(marker_json_dir: Path, stem: str) -> Dict[int, Tuple[float, float, float, float]]:
    p = marker_json_dir / f"{stem}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    markers = data.get("markers", []) or []
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for it in markers:
        try:
            mid = int(it["marker_id"])
            b = it["bbox"]
            out[mid] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            continue
    return out


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_file)
    tp_path = Path(args.text_pairs_jsonl)
    marker_dir = Path(args.marker_json_dir)
    records = load_jsonl(tp_path)
    pred_pairs = parse_pred_file(pred_path)
    n = min(len(records), len(pred_pairs))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    exclude_pred_ids = parse_exclude_pred_ids(args.exclude_pred_ids)

    id_correct = 0
    total = 0
    invalid = 0
    excluded = 0
    ious: List[float] = []
    sum_intersection = 0.0
    sum_union = 0.0
    marker_cache: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}

    for i in range(n):
        rec = records[i]
        img_rec = str(rec.get("image_file", "")).strip()
        img_pred, resp = pred_pairs[i]
        try:
            gt_mid = int(rec.get("target_marker_id"))
        except Exception:
            gt_mid = -1
        gt_box = rec.get("gt_bbox_xyxy")
        if not isinstance(gt_box, list) or len(gt_box) != 4:
            invalid += 1
            ious.append(0.0)
            continue
        gt_box_t = (float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3]))
        pred_id = extract_predicted_id(resp)
        if pred_id is None:
            invalid += 1
            ious.append(0.0)
            sum_union += box_area_xyxy(gt_box_t)
            continue
        if pred_id in exclude_pred_ids:
            excluded += 1
            continue
        total += 1
        if pred_id == gt_mid:
            id_correct += 1
        stem = Path(img_rec or img_pred).stem
        if stem not in marker_cache:
            marker_cache[stem] = load_marker_bbox_map(marker_dir, stem)
        pred_box = marker_cache[stem].get(int(pred_id))
        if pred_box is None:
            invalid += 1
            ious.append(0.0)
            sum_union += box_area_xyxy(gt_box_t)
            continue
        inter, union = intersection_union_xyxy(pred_box, gt_box_t)
        sum_intersection += inter
        sum_union += union
        ious.append(float(inter / union) if union > 0 else 0.0)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    cum_iou = sum_intersection / sum_union if sum_union > 0 else 0.0
    id_acc = id_correct / max(1, total)
    print("=" * 70)
    print("OPT-RSVG Qwen Teacher-Marker Evaluation")
    print("=" * 70)
    print(
        f"evaluated_n={len(ious)} invalid_or_missing={invalid} "
        f"excluded_by_pred_id={excluded} exclude_pred_ids={sorted(exclude_pred_ids) if exclude_pred_ids else '(disabled)'}"
    )
    print(f"ID accuracy = {id_acc:.4f} ({id_correct}/{total})")
    print(f"Mean IoU = {mean_iou:.4f}")
    print(f"cumIoU = {cum_iou:.4f}")
    for thr in thresholds:
        v = sum(1 for x in ious if x >= thr) / max(1, len(ious))
        print(f"precision@{thr}: {v:.4f}")
        print(f"PR{thr}: {v:.4f}")
        print(f"AP@{thr}: {v:.4f}")
        print(f"AP{thr}: {v:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
