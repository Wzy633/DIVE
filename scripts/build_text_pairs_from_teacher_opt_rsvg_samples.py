#!/usr/bin/env python3
"""Build text pairs for OPT-RSVG from valid sample manifests and teacher marker jsons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OPT-RSVG text pairs from valid samples + teacher markers.")
    p.add_argument("--samples-jsonl", type=str, required=True)
    p.add_argument("--teacher-marker-dir", type=str, required=True, help="vis_pred_marker directory")
    p.add_argument("--out-jsonl", type=str, required=True)
    p.add_argument("--min-match-iou", type=float, default=0.5)
    p.add_argument("--keep-unmatched", action="store_true")
    return p.parse_args()


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


def norm_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def load_teacher_markers(marker_json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(marker_json_path.read_text(encoding="utf-8"))
    markers = data.get("markers", []) or []
    return [m for m in markers if isinstance(m, dict) and "marker_id" in m and "bbox" in m]


def match_marker_id(
    gt_box: Tuple[float, float, float, float],
    gt_class: str,
    markers: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Tuple[float, float, float, float]], float, str]:
    gt_n = norm_name(gt_class)
    same = [m for m in markers if norm_name(str(m.get("class", "") or "")) == gt_n]
    policy = "class_filtered"
    cand = same
    if not cand:
        cand = markers
        policy = "no_class_match"
    if not cand:
        return None, None, 0.0, "no_markers"
    best_iou = -1.0
    best_id: Optional[int] = None
    best_box: Optional[Tuple[float, float, float, float]] = None
    for m in cand:
        try:
            mid = int(m["marker_id"])
            b = m["bbox"]
            box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            continue
        biou = iou_xyxy(gt_box, box)
        if biou > best_iou:
            best_iou = biou
            best_id = mid
            best_box = box
    if best_id is None or best_box is None:
        return None, None, 0.0, policy
    return best_id, best_box, float(best_iou), policy


def build_question(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        d = "(no description)"
    return f"Please locate the target according to the description: {d} Output only the target ID."


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples_jsonl)
    marker_dir = Path(args.teacher_marker_dir)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    n_total = 0
    n_matched = 0
    n_unmatched = 0
    n_missing_marker_json = 0

    with out_path.open("w", encoding="utf-8") as w:
        for s in samples:
            n_total += 1
            image_file = str(s["image_file"])
            stem = str(s.get("image_stem", Path(image_file).stem))
            gt_class = str(s["target_class"])
            gt_box = tuple(float(x) for x in s["gt_bbox_xyxy"])
            desc = str(s.get("description", "") or "")
            marker_json = marker_dir / f"{stem}.json"
            if marker_json.is_file():
                markers = load_teacher_markers(marker_json)
            else:
                markers = []
                n_missing_marker_json += 1

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
                "sample_index": int(s["sample_index"]),
                "image_file": image_file,
                "question": build_question(desc),
                "target_marker_id": target_mid,
                "target_class": gt_class,
                "bbox_xyxy": bbox_xyxy,
                "gt_bbox_xyxy": [float(x) for x in gt_box],
                "match_iou": float(biou),
                "match_policy": policy,
                "source": "opt_rsvg_description_teacher_marker",
                "expected_answer": f"<answer>{target_mid}</answer>",
                "expected_response_template": f"<think>Locate the referred target.</think><answer>{target_mid}</answer>",
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("Build text pairs from OPT-RSVG valid samples + teacher markers")
    print("=" * 70)
    print(f"samples_total={n_total}")
    print(f"matched_iou>={float(args.min_match_iou):.2f}: {n_matched}")
    print(f"unmatched: {n_unmatched}")
    print(f"missing_marker_json: {n_missing_marker_json}")
    print(f"out_jsonl={out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
