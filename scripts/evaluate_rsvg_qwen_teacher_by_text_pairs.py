#!/usr/bin/env python3
"""
Evaluate Qwen marker predictions on DIOR-RSVG teacher markers using XML-description-aligned text pairs.

Inputs:
  - --pred-file: results_qwen_marker.txt produced by qwen_marker_infer_openai_compat.py
      Format per sample (block separated by 70 dashes):
        image_file.jpg
        <model response containing <answer>k</answer>>
  - --text-pairs-jsonl: text_pairs_rsvg_teacher.fixed.jsonl
      Each line contains:
        image_file, target_marker_id, gt_bbox_xyxy
  - --marker-json-dir: directory containing teacher marker jsons (<stem>.json)
      Each json contains:
        {"image_name": "<stem>", "markers":[{"marker_id":1,"bbox":[x1,y1,x2,y2], ...}, ...]}

Metrics:
  - ID accuracy (pred_id == target_marker_id)
  - Mean IoU (IoU between predicted marker bbox and GT bbox)
  - precision@{0.5..0.9} where IoU>=thr counts as correct (unresolved/missing treated as IoU=0)
  - invalid_or_missing: pred_id missing OR bbox lookup failed OR GT bbox missing
  - --exclude-pred-ids: 从上述指标中排除指定 pred_id（默认 0，对应推理占位/缺图/API 失败时的 <answer>0</answer>）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SEP = "-" * 70
ANSWER_TAG_RE = re.compile(r"<answer>(\d+)</answer>")
ANSWER_ALT_RE = re.compile(r"answer[:\s]+(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DIOR-RSVG Qwen teacher-marker predictions by text-pairs.")
    p.add_argument("--pred-file", type=str, required=True)
    p.add_argument("--text-pairs-jsonl", type=str, required=True)
    p.add_argument("--marker-json-dir", type=str, required=True)
    p.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--exclude-pred-ids",
        type=str,
        default="0",
        help="逗号分隔的 pred_id，从指标中完全排除（不计入分母）。默认 '0' 排除占位预测；传空字符串关闭。",
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
        image_file = lines[0]
        resp = "\n".join(lines[1:]).strip()
        out.append((image_file, resp))
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_marker_bbox_map(marker_json_dir: Path, stem: str) -> Dict[int, Tuple[float, float, float, float]]:
    p = marker_json_dir / f"{stem}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    markers = data.get("markers", []) or []
    m: Dict[int, Tuple[float, float, float, float]] = {}
    for it in markers:
        try:
            mid = int(it["marker_id"])
            b = it["bbox"]
            m[mid] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            continue
    return m


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_file)
    tp_path = Path(args.text_pairs_jsonl)
    marker_dir = Path(args.marker_json_dir)
    if not pred_path.is_file():
        raise SystemExit(f"pred-file not found: {pred_path}")
    if not tp_path.is_file():
        raise SystemExit(f"text-pairs-jsonl not found: {tp_path}")
    if not marker_dir.is_dir():
        raise SystemExit(f"marker-json-dir not found: {marker_dir}")

    records = load_jsonl(tp_path)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]

    pred_pairs = parse_pred_file(pred_path)
    n = min(len(records), len(pred_pairs))
    if n == 0:
        raise SystemExit("no samples to evaluate")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    exclude_pred_ids = parse_exclude_pred_ids(args.exclude_pred_ids)

    id_correct = 0
    total = 0
    invalid = 0
    excluded = 0
    ious: List[float] = []
    sum_intersection = 0.0
    sum_union = 0.0

    # cache marker maps per image stem
    marker_cache: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}

    for i in range(n):
        rec = records[i]
        img_rec = str(rec.get("image_file", "")).strip()
        img_pred, resp = pred_pairs[i]
        if img_rec and img_pred and img_rec != img_pred:
            # still evaluate by record order; image mismatch indicates upstream alignment issue
            pass

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
            try:
                marker_cache[stem] = load_marker_bbox_map(marker_dir, stem)
            except Exception:
                marker_cache[stem] = {}
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

    if not ious:
        print("[WARN] no samples in metrics after filtering (check preds / --exclude-pred-ids).")
        print(
            f"invalid_or_missing={invalid} excluded_by_pred_id={excluded} "
            f"exclude_pred_ids={sorted(exclude_pred_ids) if exclude_pred_ids else '(disabled)'}"
        )
        return

    mean_iou = sum(ious) / len(ious)
    cum_iou = sum_intersection / sum_union if sum_union > 0 else 0.0
    id_acc = id_correct / max(1, total)
    precisions: Dict[str, float] = {}
    for thr in thresholds:
        precisions[f"precision@{thr}"] = sum(1 for x in ious if x >= thr) / len(ious)

    print("=" * 70)
    print("DIOR-RSVG Qwen Teacher-Marker Evaluation (XML description)")
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
        v = precisions[k]
        print(f"{k}: {v:.4f}")
        print(f"PR{thr}: {v:.4f}")
        print(f"AP@{thr}: {v:.4f}")
        print(f"AP{thr}: {v:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
