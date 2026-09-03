#!/usr/bin/env python3
"""Your-style marker pipeline for DIOR_RSVG using LAE-DINO detections.

Goal: align DIOR_RSVG marker style with your DIOR pipeline (`scripts/ours/dior_stvg_marker_pipeline.py`):
  - detector bboxes (xyxy) + score_thr
  - class-agnostic NMS
  - stable ID assignment
  - draw red digits (and optionally bboxes) on images

Outputs:
  out_dir/
    images/<filename>.jpg                 (if --save-images)
    marker_meta_lae_stvg.json             (list of per-image records)
    qa_stvg_style.json                    (simple QA pairs; optional downstream)
    pipeline_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_MMDET_ROOT = Path(__file__).resolve().parents[1] / "mmdetection_lae"
if str(_MMDET_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMDET_ROOT))

from mmdet.apis import inference_detector, init_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIOR_RSVG marker (ours style, LAE-DINO detections).")
    p.add_argument("--dior-rsvg-root", type=str, default="DIOR_RSVG")
    p.add_argument("--xml-dir", type=str, default="", help="Optional xml directory override (default: <dior-rsvg-root>/Annotations)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--score-thr", type=float, default=0.35)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--max-dets", type=int, default=100)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--font", type=str, default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    p.add_argument("--font-size", type=int, default=22)
    p.add_argument("--stroke-width", type=int, default=2)
    p.add_argument("--color", type=str, default="#ff0000")
    p.add_argument("--draw-bboxes", action="store_true")
    p.add_argument("--box-width", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N XMLs.")
    return p.parse_args()


def _bbox_area_xyxy(b: torch.Tensor) -> float:
    x1, y1, x2, y2 = b.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centroid_xyxy(b: torch.Tensor) -> Tuple[float, float]:
    x1, y1, x2, y2 = b.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def nms_filter(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float, max_num: int) -> List[int]:
    if boxes.numel() == 0:
        return []
    from torchvision.ops import nms

    keep = nms(boxes, scores, iou_thr)
    keep = keep[:max_num].tolist()
    return keep


def sort_instances_for_stable_ids(boxes: torch.Tensor) -> List[int]:
    n = boxes.shape[0]
    keys = []
    for i in range(n):
        b = boxes[i]
        area = _bbox_area_xyxy(b)
        x1, y1, _, _ = b.tolist()
        keys.append((-area, -y1, -x1, i))
    keys.sort()
    return [k[-1] for k in keys]


def draw_markers_pil(
    img_path: Path,
    markers: List[Dict[str, Any]],
    out_path: Path,
    font_path: str,
    font_size: int,
    color: str,
    stroke_width: int,
    draw_bboxes: bool = False,
    box_width: int = 1,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        font = ImageFont.load_default()

    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        s = h.lstrip("#")
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

    rgb = _hex_to_rgb(color)

    for m in markers:
        x1, y1, x2, y2 = m["bbox_xyxy"]
        if draw_bboxes:
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=max(1, int(box_width)))
        mid = str(m["marker_id"])
        # Put digit center exactly at bbox top-right corner.
        tx, ty = float(x2), float(y1)
        try:
            draw.text(
                (tx, ty),
                mid,
                font=font,
                fill=color,
                stroke_width=stroke_width,
                stroke_fill="#000000",
                anchor="mm",
            )
        except TypeError:
            # older Pillow fallback: center text manually at (tx, ty)
            bbox = draw.textbbox((0, 0), mid, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (tx - w / 2.0, ty - h / 2.0),
                mid,
                font=font,
                fill=color,
                stroke_width=stroke_width,
                stroke_fill="#000000",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def parse_xml_filename(xml_path: Path) -> Optional[str]:
    try:
        root = ET.parse(xml_path).getroot()
        fe = root.find("filename")
        if fe is None or not fe.text:
            return None
        return fe.text.strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    rsvg_root = Path(args.dior_rsvg_root)
    xml_dir = Path(args.xml_dir) if args.xml_dir else (rsvg_root / "Annotations")
    img_dir = rsvg_root / "JPEGImages"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(str(_MMDET_ROOT))
    model = init_detector(args.config, args.checkpoint, device=args.device)
    classes = model.dataset_meta.get("classes")
    if classes is None:
        raise SystemExit("Missing model.dataset_meta['classes']; cannot build text_prompt.")

    xmls = sorted(xml_dir.glob("*.xml"))
    if args.limit and args.limit > 0:
        xmls = xmls[: args.limit]

    meta_all: List[Dict[str, Any]] = []
    qa_all: List[Dict[str, Any]] = []

    for xml_path in xmls:
        filename = parse_xml_filename(xml_path)
        if not filename:
            continue
        img_path = img_dir / filename
        if not img_path.is_file():
            continue

        # open-vocab prompt uses the fixed DIOR 20 classes
        result = inference_detector(model, str(img_path), text_prompt=classes, custom_entities=True)
        pi = result.pred_instances
        if len(pi) == 0:
            meta_all.append({"file_name": filename, "xml": xml_path.name, "instances": []})
            continue

        boxes = pi.bboxes
        scores = pi.scores
        labels = pi.labels

        keep_mask = scores >= float(args.score_thr)
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        if boxes.numel() == 0:
            meta_all.append({"file_name": filename, "xml": xml_path.name, "instances": []})
            continue

        idx = nms_filter(boxes, scores, float(args.nms_iou), int(args.max_dets))
        boxes = boxes[idx]
        scores = scores[idx]
        labels = labels[idx]

        order = sort_instances_for_stable_ids(boxes)

        markers: List[Dict[str, Any]] = []
        mid = int(args.start_id)
        for oi in order:
            b = boxes[oi]
            cx, cy = _centroid_xyxy(b)
            lab = int(labels[oi].item())
            name = classes[lab] if 0 <= lab < len(classes) else str(lab)
            markers.append(
                {
                    "marker_id": mid,
                    "bbox_xyxy": [float(x) for x in b.tolist()],
                    "score": float(scores[oi].item()),
                    "label": lab,
                    "label_name": name,
                    "centroid_xy": [cx, cy],
                }
            )
            qa_all.append(
                {
                    "image_file": filename,
                    "question": f"Where is the {name}?",
                    "target_marker_id": mid,
                    "expected_answer": f"<answer>{mid}</answer>",
                }
            )
            mid += 1

        meta_all.append({"file_name": filename, "xml": xml_path.name, "instances": markers})

        if args.save_images:
            draw_markers_pil(
                img_path,
                markers,
                out_dir / "images" / filename,
                args.font,
                int(args.font_size),
                args.color,
                int(args.stroke_width),
                draw_bboxes=bool(args.draw_bboxes),
                box_width=int(args.box_width),
            )

    with (out_dir / "marker_meta_lae_stvg.json").open("w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)
    with (out_dir / "qa_stvg_style.json").open("w", encoding="utf-8") as f:
        json.dump(qa_all, f, ensure_ascii=False, indent=2)
    with (out_dir / "pipeline_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "ours_rsvg",
                "dior_rsvg_root": str(rsvg_root),
                "config": args.config,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "score_thr": args.score_thr,
                "nms_iou": args.nms_iou,
                "max_dets": args.max_dets,
                "num_xml": len(xmls),
                "num_images_written": len(meta_all),
                "num_qa": len(qa_all),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] images={len(meta_all)} -> {out_dir}")


if __name__ == "__main__":
    main()

