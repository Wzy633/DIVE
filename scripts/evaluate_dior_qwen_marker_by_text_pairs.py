#!/usr/bin/env python3
"""
Evaluate DIOR Qwen marker predictions using STVG-style text-pairs JSONL.

This is the DIOR (COCO-derived) counterpart of `evaluate_rsvg_marker(1).py`.
It supports two JSONL styles:
  - GT-backed records: prefer `gt_bbox_xyxy` as the true target bbox.
  - Legacy marker-only records: fallback to `bbox_xyxy`.

For a predicted `marker_id`, we look up the corresponding bbox from other JSONL lines
sharing the same (image_file, marker_id) key.

Inputs:
  - --pred-file: results.txt produced by `qwen_marker_infer_openai_compat.py`
    Format per sample:
      image_file.jpg
      <model response ... containing <answer>k</answer>>
  - --text-pairs-jsonl: *fixed.jsonl* where `expected_answer` already contains <answer>k</answer>.

Metrics:
  - ID accuracy
  - mean IoU
  - precision@{0.5..0.9} computed with invalid/lookup failures treated as IoU=0.
  - --exclude-pred-ids: omit samples whose predicted id is in this set from metrics (default: 0 placeholder).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DIOR Qwen marker preds using text-pairs.")
    p.add_argument("--pred-file", type=str, required=True, help="results_qwen_marker.txt")
    p.add_argument("--text-pairs-jsonl", type=str, required=True, help="text_pairs_stvg_style.fixed.jsonl")
    p.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N samples.")
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated IoU thresholds.",
    )
    p.add_argument(
        "--exclude-pred-ids",
        type=str,
        default="0",
        help="Comma-separated pred_ids to exclude from metrics (not counted). Default '0'; empty disables.",
    )
    return p.parse_args()


def parse_exclude_pred_ids(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise SystemExit(f"invalid --exclude-pred-ids token: {part!r}")
    return out


ANSWER_TAG_RE = re.compile(r"<answer>(\d+)</answer>")
ANSWER_ALT_RE = re.compile(r"answer[:\s]+(\d+)", re.IGNORECASE)


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iou_xyxy(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    inter, union = intersection_union_xyxy(box1, box2)
    return float(inter / union) if union > 0 else 0.0


def intersection_union_xyxy(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        inter_area = 0.0
    else:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area1 + area2 - inter_area
    return float(inter_area), float(max(0.0, union))


def box_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_pred_file(pred_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (image_file, response_text) aligned with qwen_marker_infer_openai_compat.py output.
    """
    txt = pred_path.read_text(encoding="utf-8")
    parts = txt.split("-" * 70)
    out: List[Tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
        if not lines:
            continue
        image_file = lines[0]
        resp = "\n".join(lines[1:]).strip()
        out.append((image_file, resp))
    return out


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_file)
    tp_path = Path(args.text_pairs_jsonl)
    if not pred_path.is_file():
        raise SystemExit(f"pred-file not found: {pred_path}")
    if not tp_path.is_file():
        raise SystemExit(f"text-pairs-jsonl not found: {tp_path}")

    records = load_jsonl(tp_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    pred_pairs = parse_pred_file(pred_path)
    # 以 pred 文件中的顺序为准：qwen 推理是逐行 jsonl 产生样本，因此通常两者长度一致。
    if len(pred_pairs) != len(records):
        # 尽量按 image_file 对齐（fallback）
        # 如果仍无法对齐，就以最短长度截断。
        rec_by_key: Dict[str, List[int]] = {}
        for idx, r in enumerate(records):
            img = str(r.get("image_file", "")).strip()
            rec_by_key.setdefault(img, []).append(idx)
        aligned_preds: List[Tuple[int, str]] = []
        used = set()
        for img, resp in pred_pairs:
            if img in rec_by_key and rec_by_key[img]:
                # use first unused
                for ridx in rec_by_key[img]:
                    if ridx not in used:
                        aligned_preds.append((ridx, resp))
                        used.add(ridx)
                        break
        # Build sequential list by record index
        if len(aligned_preds) == 0:
            n = min(len(pred_pairs), len(records))
            pred_pairs = pred_pairs[:n]
            records = records[:n]
        else:
            # convert to dict of record index -> resp
            resp_by_ridx = {ridx: resp for ridx, resp in aligned_preds}
            pred_pairs = [(str(records[i].get("image_file", "")), resp_by_ridx.get(i, "")) for i in range(len(records))]

    # Build bbox lookup by (image_file, marker_id)
    bbox_by_key: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
    for r in records:
        img = str(r.get("image_file", "")).strip()
        mid = r.get("target_marker_id", None)
        bbox = r.get("bbox_xyxy", None)
        if not img or mid is None or bbox is None:
            continue
        try:
            mid_i = int(mid)
            box = tuple(float(x) for x in bbox)
            if len(box) != 4:
                continue
            bbox_by_key[(img, mid_i)] = (box[0], box[1], box[2], box[3])
        except Exception:
            continue

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    exclude_pred_ids = parse_exclude_pred_ids(args.exclude_pred_ids)

    id_correct = 0
    total = 0
    ious: List[float] = []
    sum_intersection = 0.0
    sum_union = 0.0
    invalid = 0
    excluded = 0

    n = min(len(pred_pairs), len(records))
    for idx in range(n):
        img, resp = pred_pairs[idx]
        gt_rec = records[idx]
        gt_mid = gt_rec.get("target_marker_id", None)
        img_rec = str(gt_rec.get("image_file", "")).strip()
        if not img:
            invalid += 1
            ious.append(0.0)
            continue
        if str(img_rec) != str(img):
            # mismatch: still proceed with gt bbox from record
            pass
        try:
            gt_mid_i = int(gt_mid)
        except Exception:
            invalid += 1
            ious.append(0.0)
            continue

        pred_id = extract_predicted_id(resp)
        if pred_id is None:
            invalid += 1
            ious.append(0.0)
            raw_gt_box = gt_rec.get("gt_bbox_xyxy", gt_rec.get("bbox_xyxy"))
            if isinstance(raw_gt_box, (list, tuple)) and len(raw_gt_box) == 4:
                sum_union += box_area_xyxy(tuple(float(x) for x in raw_gt_box))
            continue
        if pred_id in exclude_pred_ids:
            excluded += 1
            continue

        total += 1
        if pred_id == gt_mid_i:
            id_correct += 1

        gt_box = gt_rec.get("gt_bbox_xyxy", None)
        if gt_box is None:
            gt_box = gt_rec.get("bbox_xyxy", None)
        if gt_box is None:
            invalid += 1
            ious.append(0.0)
            continue
        gt_box_t = tuple(float(x) for x in gt_box)
        gt_box_t = (gt_box_t[0], gt_box_t[1], gt_box_t[2], gt_box_t[3])

        pred_box = bbox_by_key.get((img, pred_id), None)
        if pred_box is None:
            invalid += 1
            ious.append(0.0)
            sum_union += box_area_xyxy(gt_box_t)
            continue

        inter, union = intersection_union_xyxy(pred_box, gt_box_t)
        sum_intersection += inter
        sum_union += union
        ious.append(float(inter / union) if union > 0 else 0.0)

    if not ious:
        print("[WARN] no samples evaluated.")
        return

    mean_iou = sum(ious) / len(ious)
    cum_iou = sum_intersection / sum_union if sum_union > 0 else 0.0
    precisions: Dict[str, float] = {}
    for thr in thresholds:
        precisions[f"precision@{thr}"] = sum(1 for x in ious if x >= thr) / len(ious)

    id_acc = id_correct / max(1, total)
    print("=" * 70)
    print("DIOR Qwen Marker Evaluation (val-style, IoU by text-pairs)")
    print("=" * 70)
    ex_rule = sorted(exclude_pred_ids) if exclude_pred_ids else "(disabled)"
    print(
        f"evaluated_n={len(ious)} invalid_or_missing={invalid} "
        f"excluded_by_pred_id={excluded} exclude_pred_ids={ex_rule}"
    )
    print(f"ID accuracy = {id_acc:.4f} ({id_correct}/{total})")
    print(f"Mean IoU = {mean_iou:.4f}")
    print(f"cumIoU = {cum_iou:.4f}")
    for k in sorted(precisions.keys(), key=lambda s: float(s.split('@')[-1])):
        thr = k.split("@")[-1]
        # Keep multiple aliases for easy alignment:
        # - precision@0.5
        # - PR0.5 (your DIOR-RSVG naming)
        # - AP@0.5 / AP0.5 (DIOR paper sometimes uses AP/PR shorthand for this protocol)
        v = precisions[k]
        print(f"{k}: {v:.4f}")
        print(f"PR{thr}: {v:.4f}")
        print(f"AP@{thr}: {v:.4f}")
        print(f"AP{thr}: {v:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
