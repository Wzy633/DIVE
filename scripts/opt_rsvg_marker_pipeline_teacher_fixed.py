#!/usr/bin/env python3
"""Teacher-style marker pipeline for OPT-RSVG with fixed-class detectors.

This script is intended for detectors such as LAE-DINO or MTP that output a
fixed class vocabulary. The downstream teacher protocol is kept consistent with
the DIOR-RSVG teacher flow; only the detector front-end is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

_MMDET_ROOT = Path(__file__).resolve().parents[1] / "mmdetection_lae"
if str(_MMDET_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMDET_ROOT))

if hasattr(torch.optim, "Adafactor"):
    try:
        delattr(torch.optim, "Adafactor")
    except Exception:
        pass

from mmdet.apis import inference_detector, init_detector  # noqa: E402


OPT_SUPPORTED_10CLS = [
    "airplane",
    "baseball diamond",
    "basketball court",
    "bridge",
    "ground track field",
    "harbor",
    "ship",
    "storage tank",
    "tennis court",
    "vehicle",
]

# Canonical OPT-RSVG class -> detector-side DIOR class name
OPT_TO_DIOR_CLASS = {
    "airplane": "airplane",
    "baseball diamond": "baseballfield",
    "basketball court": "basketballcourt",
    "bridge": "bridge",
    "ground track field": "groundtrackfield",
    "harbor": "harbor",
    "ship": "ship",
    "storage tank": "storagetank",
    "tennis court": "tenniscourt",
    "vehicle": "vehicle",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-style OPT-RSVG marker pipeline for fixed-class detectors.")
    p.add_argument("--opt-root", type=str, required=True)
    p.add_argument("--samples-jsonl", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--keep-score-threshold", type=float, default=0.4)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--pred-score-thr", type=float, default=0.0)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--limit-images", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--font-scale", type=float, default=0.6)
    p.add_argument("--text-thickness", type=int, default=2)
    p.add_argument("--line-thickness", type=int, default=4)
    return p.parse_args()


def norm_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def find_names_in_description(description: str, class_names: Iterable[str]) -> List[str]:
    if not description:
        return []
    desc_n = description.lower().replace(" ", "")
    out: List[str] = []
    for cls in class_names:
        cls_n = cls.lower().replace(" ", "")
        if cls_n in desc_n and cls not in out:
            out.append(cls)
    return out


def load_samples_by_image(samples_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    samples_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_desc: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    kept = 0
    dropped_ambiguous = 0

    with samples_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        image_file = str(row["image_file"])
        desc = str(row.get("description", "") or "").strip()
        idx = int(row["sample_index"])
        seen_desc[(image_file, desc)].add(idx)

    ambiguous_keys = {k for k, v in seen_desc.items() if len(v) > 1 and k[1]}

    for row in rows:
        image_file = str(row["image_file"])
        desc = str(row.get("description", "") or "").strip()
        if (image_file, desc) in ambiguous_keys:
            dropped_ambiguous += 1
            continue
        samples_by_image[image_file].append(row)
        kept += 1

    stats = {
        "samples_total": len(rows),
        "samples_kept": kept,
        "samples_dropped_ambiguous": dropped_ambiguous,
        "ambiguous_groups": len(ambiguous_keys),
        "images_with_ambiguous_groups": len({k[0] for k in ambiguous_keys}),
    }
    return samples_by_image, stats


def parse_opt_target_classes(samples: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for s in samples:
        cls = str(s.get("target_class", "") or "").strip()
        if cls and cls not in names:
            names.append(cls)
        desc = str(s.get("description", "") or "")
        for found in find_names_in_description(desc, OPT_SUPPORTED_10CLS):
            if found not in names:
                names.append(found)
    return names


def calculate_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
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


def suppress_overlapping_bboxes_by_iou_for_label(
    pred_bboxes: List[Tuple[float, float, float, float, float]],
    iou_threshold: float,
    keep_score_threshold: float,
) -> List[Tuple[float, float, float, float, float]]:
    if not pred_bboxes:
        return pred_bboxes
    high_score = [b for b in pred_bboxes if b[4] >= keep_score_threshold]
    low_score = [b for b in pred_bboxes if b[4] < keep_score_threshold]
    sorted_bboxes = sorted(low_score, key=lambda x: x[4], reverse=True)
    kept: List[Tuple[float, float, float, float, float]] = []
    for cand in sorted_bboxes:
        cand_coords = cand[:4]
        should_keep = True
        for kept_box in kept:
            if calculate_iou(cand_coords, kept_box[:4]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)
    merged = high_score + kept
    merged.sort(key=lambda x: x[4], reverse=True)
    return merged


def suppress_overlapping_bboxes_by_iou_with_fallback(
    all_pred_bboxes: List[Tuple[float, float, float, float, float, int]],
    score_threshold: float,
    iou_threshold: float,
    keep_score_threshold: float,
) -> List[Tuple[float, float, float, float, float, int]]:
    if not all_pred_bboxes:
        return []
    bboxes_by_label: Dict[int, List[Tuple[float, float, float, float, float, int]]] = defaultdict(list)
    for b in all_pred_bboxes:
        bboxes_by_label[int(b[5])].append(b)
    merged_all: List[Tuple[float, float, float, float, float, int]] = []
    for label, label_bboxes in bboxes_by_label.items():
        above = [b for b in label_bboxes if b[4] >= score_threshold]
        if above:
            coords = [b[:5] for b in above]
            merged = suppress_overlapping_bboxes_by_iou_for_label(coords, iou_threshold, keep_score_threshold)
            merged_all.extend([(*m, label) for m in merged])
        else:
            top2 = sorted(label_bboxes, key=lambda x: x[4], reverse=True)[:2]
            merged_all.extend(top2)
    merged_all.sort(key=lambda x: x[4], reverse=True)
    return merged_all


def _color_for_label(label: int) -> Tuple[int, int, int]:
    hue = (label * 37) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_bbox(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    thickness: int,
    label: Optional[str],
    font_scale: float,
    text_thickness: int,
) -> None:
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    text_bg_height = text_height + baseline + 5
    text_bg_width = text_width
    img_height, img_width = img.shape[:2]
    text_bg_x1 = xmin
    text_bg_y1 = ymin - text_bg_height
    text_bg_x2 = xmin + text_bg_width
    text_bg_y2 = ymin
    if text_bg_y1 < 0:
        text_bg_y1 = ymin
        text_bg_y2 = ymin + text_bg_height
        text_y = ymin + text_height + baseline - 2
    else:
        text_y = ymin - baseline - 2
    if text_bg_x2 > img_width:
        text_bg_x1 = xmax - text_bg_width
        text_bg_x2 = xmax
        text_x = text_bg_x1
    else:
        text_x = xmin
    if text_bg_x1 < 0:
        text_bg_x1 = xmin
        text_bg_x2 = xmin + text_bg_width
        text_x = xmin
    text_bg_x1 = max(0, min(text_bg_x1, img_width - 1))
    text_bg_y1 = max(0, min(text_bg_y1, img_height - 1))
    text_bg_x2 = max(0, min(text_bg_x2, img_width - 1))
    text_bg_y2 = max(0, min(text_bg_y2, img_height - 1))
    text_x = max(0, min(text_x, img_width - 1))
    text_y = max(text_height, min(text_y, img_height - 1))
    cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)


def save_preds_json(
    out_path: Path,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    label_to_class: Dict[int, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bboxes": [[float(v) for v in row] for row in boxes],
        "scores": [float(v) for v in scores],
        "labels": [int(v) for v in labels],
        "label_to_class": {str(k): str(v) for k, v in label_to_class.items()},
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def render_teacher_markers(
    image_path: Path,
    merged_pred_bboxes: List[Tuple[float, float, float, float, float, int]],
    label_to_class: Dict[int, str],
    out_img_path: Path,
    out_json_path: Path,
    *,
    line_thickness: int,
    font_scale: float,
    text_thickness: int,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"failed to load image: {image_path}")
    marker_info_list: List[Dict[str, Any]] = []
    for i, pred_bbox in enumerate(merged_pred_bboxes):
        xmin, ymin, xmax, ymax, score, lbl = pred_bbox
        marker_id = i + 1
        cls = label_to_class.get(int(lbl), f"class_{lbl}")
        color = _color_for_label(int(lbl))
        draw_bbox(
            img,
            (xmin, ymin, xmax, ymax),
            color,
            line_thickness,
            str(marker_id),
            font_scale,
            text_thickness,
        )
        marker_info_list.append(
            {
                "marker_id": marker_id,
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "class": cls,
                "label": int(lbl),
                "score": float(score),
            }
        )
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), img)
    out_json_path.write_text(
        json.dumps({"image_name": image_path.stem, "markers": marker_info_list}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_done_stems(vis_dir: Path) -> set[str]:
    if not vis_dir.is_dir():
        return set()
    return {p.stem for p in vis_dir.glob("*.json")}


def main() -> None:
    args = parse_args()
    opt_root = Path(args.opt_root)
    img_dir = opt_root / "Image"
    out_dir = Path(args.out_dir)
    preds_dir = out_dir / "outputs" / "preds"
    vis_dir = out_dir / "vis_pred_marker"
    samples_path = Path(args.samples_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_by_image, sample_stats = load_samples_by_image(samples_path)
    image_files = sorted(samples_by_image.keys())
    if args.limit_images and args.limit_images > 0:
        image_files = image_files[: int(args.limit_images)]

    kept_samples_jsonl = out_dir / "samples_kept_after_teacher_filter.jsonl"
    with kept_samples_jsonl.open("w", encoding="utf-8") as f:
        for image_file in image_files:
            for row in samples_by_image.get(image_file, []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    done_stems = load_done_stems(vis_dir) if args.resume else set()

    os.chdir(str(_MMDET_ROOT))
    # Explicitly pass a palette so init_detector does not try to build the
    # test dataset only to fetch metainfo/palette. Some local infer configs
    # carry dataset kwargs that are not compatible with that code path.
    model = init_detector(args.config, args.checkpoint, device=args.device, palette="random")

    n_processed = 0
    n_empty = 0

    for image_file in image_files:
        stem = Path(image_file).stem
        if args.resume and stem in done_stems:
            continue
        img_path = img_dir / image_file
        if not img_path.is_file():
            continue
        samples = samples_by_image.get(image_file, [])
        target_classes = parse_opt_target_classes(samples)
        target_classes = [c for c in target_classes if c in OPT_TO_DIOR_CLASS]
        if not target_classes:
            n_empty += 1
            (vis_dir / f"{stem}.json").write_text(
                json.dumps({"image_name": stem, "markers": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            n_processed += 1
            continue

        detector_prompt_classes: List[str] = []
        label_to_class: Dict[int, str] = {}
        for opt_cls in target_classes:
            det_cls = OPT_TO_DIOR_CLASS[opt_cls]
            if det_cls in detector_prompt_classes:
                continue
            label_to_class[len(detector_prompt_classes)] = opt_cls
            detector_prompt_classes.append(det_cls)

        result = inference_detector(
            model,
            str(img_path),
            text_prompt=tuple(detector_prompt_classes),
            custom_entities=True,
        )
        pred_instances = getattr(result, "pred_instances", None)
        if pred_instances is None:
            boxes = []
            scores = []
            labels = []
        else:
            bboxes_np = pred_instances.bboxes.detach().cpu().numpy() if hasattr(pred_instances, "bboxes") else np.zeros((0, 4), dtype=np.float32)
            scores_np = pred_instances.scores.detach().cpu().numpy() if hasattr(pred_instances, "scores") else np.zeros((0,), dtype=np.float32)
            labels_np = pred_instances.labels.detach().cpu().numpy() if hasattr(pred_instances, "labels") else np.zeros((0,), dtype=np.int64)
            keep = scores_np >= float(args.pred_score_thr)
            boxes = bboxes_np[keep].tolist()
            scores = scores_np[keep].tolist()
            labels = labels_np[keep].tolist()

        filtered_boxes: List[List[float]] = []
        filtered_scores: List[float] = []
        filtered_labels: List[int] = []
        for i in range(len(boxes)):
            lbl = int(labels[i])
            if lbl not in label_to_class:
                continue
            filtered_boxes.append([float(v) for v in boxes[i]])
            filtered_scores.append(float(scores[i]))
            filtered_labels.append(lbl)

        save_preds_json(preds_dir / f"{stem}.json", filtered_boxes, filtered_scores, filtered_labels, label_to_class)

        if len(filtered_boxes) == 0:
            (vis_dir / f"{stem}.json").write_text(
                json.dumps({"image_name": stem, "markers": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            n_empty += 1
            n_processed += 1
            continue

        all_pred_bboxes = []
        for i in range(len(filtered_boxes)):
            x1, y1, x2, y2 = [float(v) for v in filtered_boxes[i]]
            all_pred_bboxes.append((x1, y1, x2, y2, float(filtered_scores[i]), int(filtered_labels[i])))

        merged = suppress_overlapping_bboxes_by_iou_with_fallback(
            all_pred_bboxes,
            score_threshold=float(args.score_threshold),
            iou_threshold=float(args.iou_threshold),
            keep_score_threshold=float(args.keep_score_threshold),
        )

        if not merged:
            (vis_dir / f"{stem}.json").write_text(
                json.dumps({"image_name": stem, "markers": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            n_empty += 1
            n_processed += 1
            continue

        if args.save_images:
            render_teacher_markers(
                img_path,
                merged,
                label_to_class,
                vis_dir / f"{stem}.jpg",
                vis_dir / f"{stem}.json",
                line_thickness=int(args.line_thickness),
                font_scale=float(args.font_scale),
                text_thickness=int(args.text_thickness),
            )
        else:
            (vis_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "image_name": stem,
                        "markers": [
                            {
                                "marker_id": i + 1,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "class": label_to_class.get(int(lbl), f"class_{lbl}"),
                                "label": int(lbl),
                                "score": float(score),
                            }
                            for i, (x1, y1, x2, y2, score, lbl) in enumerate(merged)
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"[PROGRESS] processed={n_processed}/{len(image_files)} empty={n_empty}", flush=True)

    manifest = {
        "mode": "teacher_opt_rsvg_fixed_detector",
        "opt_root": str(opt_root),
        "samples_jsonl": str(samples_path),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "pred_score_thr": args.pred_score_thr,
        "score_threshold": args.score_threshold,
        "keep_score_threshold": args.keep_score_threshold,
        "iou_threshold": args.iou_threshold,
        "opt_to_dior_class": OPT_TO_DIOR_CLASS,
        "num_images": len(image_files),
        "processed": n_processed,
        "empty": n_empty,
        "sample_filtering": sample_stats,
        "samples_kept_jsonl": str(kept_samples_jsonl),
    }
    (out_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] teacher-style OPT-RSVG markers (fixed detector) -> {out_dir}")
    print(f"images_processed={n_processed} empty={n_empty}")


if __name__ == "__main__":
    main()
