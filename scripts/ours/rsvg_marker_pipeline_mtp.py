from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config

_MMDET_ROOT = Path(__file__).resolve().parents[2] / "mmdetection_lae"
if str(_MMDET_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMDET_ROOT))

_MTP_HD_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "MTP" / "RS_Tasks_Finetune" / "Horizontal_Detection"
if str(_MTP_HD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MTP_HD_ROOT))

import mmdet  # noqa: E402
import mmdet.datasets  # noqa: E402
import mmdet.models  # noqa: E402
import mmdet.models.backbones  # noqa: E402

_MTP_MMDET_EXT = _MTP_HD_ROOT / "mmdet"
if _MTP_MMDET_EXT.is_dir() and str(_MTP_MMDET_EXT) not in mmdet.__path__:
    mmdet.__path__.append(str(_MTP_MMDET_EXT))

_MTP_DATASETS_EXT = _MTP_MMDET_EXT / "datasets"
if _MTP_DATASETS_EXT.is_dir() and str(_MTP_DATASETS_EXT) not in mmdet.datasets.__path__:
    mmdet.datasets.__path__.append(str(_MTP_DATASETS_EXT))

_MTP_MODELS_EXT = _MTP_MMDET_EXT / "models"
if _MTP_MODELS_EXT.is_dir() and str(_MTP_MODELS_EXT) not in mmdet.models.__path__:
    mmdet.models.__path__.append(str(_MTP_MODELS_EXT))

_MTP_BACKBONES_EXT = _MTP_MODELS_EXT / "backbones"
if _MTP_BACKBONES_EXT.is_dir() and str(_MTP_BACKBONES_EXT) not in mmdet.models.backbones.__path__:
    mmdet.models.backbones.__path__.append(str(_MTP_BACKBONES_EXT))

for _module_name in (
    "mmdet.datasets.dior",
    "mmdet.models.backbones.vit_rvsa_mtp",
    "mmdet.models.backbones.vit_rvsa_mtp_branches",
):
    importlib.import_module(_module_name)

from mmdet.apis import inference_detector, init_detector  # noqa: E402


def load_mtp_inference_config(path: str) -> Config:
    cfg = Config.fromfile(path)
    cfg.model.backbone.pretrained = None
    cfg.model.backbone.use_checkpoint = False
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTP teacher-style marker pipeline for DIOR-RSVG.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--xml-dir", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True, help="Split output dir; images/json saved under vis_pred_marker")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--keep-score-threshold", type=float, default=0.4)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--box-thickness", type=int, default=4)
    return p.parse_args()


def _normalize_name(s: str) -> str:
    return str(s).strip().lower().replace(' ', '')


def find_names_in_description(description: str, class_names: Tuple[str, ...]) -> List[str]:
    if not description:
        return []
    desc_norm = _normalize_name(description)
    found: List[str] = []
    for class_name in class_names:
        if _normalize_name(class_name) in desc_norm:
            found.append(class_name)
    return found


def parse_xml_annotation(xml_path: Path, class_names: Tuple[str, ...]) -> Tuple[str, List[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext('filename', default='').strip()
    names: List[str] = []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='').strip()
        if name:
            names.append(name)
        desc = obj.findtext('description', default='').strip()
        if desc:
            for found_name in find_names_in_description(desc, class_names):
                if found_name not in names:
                    names.append(found_name)
    return filename, names


def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    return 0.0 if union <= 0 else inter_area / union


def suppress_overlapping_bboxes_by_iou_for_label(pred_bboxes, iou_threshold=0.5, keep_score_threshold=0.4):
    if not pred_bboxes:
        return pred_bboxes
    high_score = [b for b in pred_bboxes if b[4] >= keep_score_threshold]
    low_score = [b for b in pred_bboxes if b[4] < keep_score_threshold]
    sorted_bboxes = sorted(low_score, key=lambda x: x[4], reverse=True)
    kept = []
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
    return sorted(merged, key=lambda x: x[4], reverse=True)


def suppress_overlapping_bboxes_by_iou_with_fallback(all_pred_bboxes, score_threshold=0.25, iou_threshold=0.5, keep_score_threshold=0.4):
    if not all_pred_bboxes:
        return []
    bboxes_by_label = defaultdict(list)
    for b in all_pred_bboxes:
        bboxes_by_label[int(b[5])].append(b)
    merged_all = []
    for label, label_bboxes in bboxes_by_label.items():
        above = [b for b in label_bboxes if b[4] >= score_threshold]
        if above:
            coords = [b[:5] for b in above]
            merged = suppress_overlapping_bboxes_by_iou_for_label(coords, iou_threshold=iou_threshold, keep_score_threshold=keep_score_threshold)
            merged_all.extend([(*m, label) for m in merged])
        else:
            top2 = sorted(label_bboxes, key=lambda x: x[4], reverse=True)[:2]
            merged_all.extend(top2)
    return sorted(merged_all, key=lambda x: x[4], reverse=True)


def draw_bbox(img: np.ndarray, bbox: Tuple[float, float, float, float], color: Tuple[int, int, int], thickness: int = 3, label: str | None = None):
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    img_h, img_w = img.shape[:2]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_bg_height = text_height + baseline + 5
        text_bg_width = text_width
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
        if text_bg_x2 > img_w:
            text_bg_x1 = xmax - text_bg_width
            text_bg_x2 = xmax
            text_x = text_bg_x1
        else:
            text_x = xmin
        if text_bg_x1 < 0:
            text_bg_x1 = xmin
            text_bg_x2 = xmin + text_bg_width
            text_x = xmin
        text_bg_x1 = max(0, min(text_bg_x1, img_w - 1))
        text_bg_y1 = max(0, min(text_bg_y1, img_h - 1))
        text_bg_x2 = max(0, min(text_bg_x2, img_w - 1))
        text_bg_y2 = max(0, min(text_bg_y2, img_h - 1))
        text_x = max(0, min(text_x, img_w - 1))
        text_y = max(text_height, min(text_y, img_h - 1))
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)


def main() -> None:
    args = parse_args()
    os.chdir(str(_MMDET_ROOT))
    model = init_detector(load_mtp_inference_config(args.config), args.checkpoint, device=args.device)
    classes = tuple(model.dataset_meta.get('classes') or [])
    if not classes:
        raise SystemExit('model.dataset_meta.classes missing')

    xml_dir = Path(args.xml_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    preds_dir = out_dir / 'outputs' / 'preds'
    vis_dir = out_dir / 'vis_pred_marker'
    preds_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob('*.xml'))
    if args.limit and args.limit > 0:
        xml_files = xml_files[: int(args.limit)]

    processed = 0
    empty = 0
    for xp in xml_files:
        filename, gt_names = parse_xml_annotation(xp, classes)
        if not filename:
            continue
        image_path = images_dir / filename
        if not image_path.is_file():
            continue
        gt_name_set = {str(n).strip().lower() for n in gt_names if isinstance(n, str) and str(n).strip()}
        gt_name_norm_set = {_normalize_name(n) for n in gt_name_set}

        result = inference_detector(model, str(image_path))
        pi = result.pred_instances
        all_pred_bboxes = []
        raw_boxes: List[List[float]] = []
        raw_scores: List[float] = []
        raw_labels: List[int] = []
        if len(pi) > 0:
            boxes = pi.bboxes.tolist()
            scores = pi.scores.tolist()
            labels = [int(x) for x in pi.labels.tolist()]
            for bbox, score, label in zip(boxes, scores, labels):
                if len(bbox) != 4:
                    continue
                cls = classes[int(label)] if int(label) < len(classes) else f'class_{label}'
                if _normalize_name(cls) not in gt_name_norm_set:
                    continue
                raw_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                raw_scores.append(float(score))
                raw_labels.append(int(label))
                all_pred_bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(score), int(label)))

        pred_bboxes = suppress_overlapping_bboxes_by_iou_with_fallback(
            all_pred_bboxes,
            score_threshold=float(args.score_threshold),
            iou_threshold=float(args.iou_threshold),
            keep_score_threshold=float(args.keep_score_threshold),
        )

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        marker_info_list = []
        for i, pred_bbox in enumerate(pred_bboxes):
            xmin, ymin, xmax, ymax, score, lbl = pred_bbox
            cls = classes[int(lbl)] if int(lbl) < len(classes) else f'class_{lbl}'
            cls_norm = str(cls).strip().lower()
            color = (0, 0, 255) if cls_norm in gt_name_set else (0, 255, 0)
            marker_id = i + 1
            draw_bbox(img, (xmin, ymin, xmax, ymax), color, int(args.box_thickness), str(marker_id))
            marker_info_list.append({
                'marker_id': marker_id,
                'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
                'class': cls,
                'label': int(lbl),
                'score': float(score),
            })

        stem = Path(filename).stem
        with (preds_dir / f'{stem}.json').open('w', encoding='utf-8') as f:
            json.dump({
                'labels': raw_labels,
                'scores': raw_scores,
                'bboxes': raw_boxes,
                'class_names': list(classes),
                'label_to_class': {str(i): str(c) for i, c in enumerate(classes)},
            }, f, indent=2, ensure_ascii=False)
        cv2.imwrite(str(vis_dir / f'{stem}.jpg'), img)
        with (vis_dir / f'{stem}.json').open('w', encoding='utf-8') as f:
            json.dump({'image_name': stem, 'markers': marker_info_list}, f, indent=2, ensure_ascii=False)

        if not marker_info_list:
            empty += 1
        processed += 1

    manifest = {
        'detector': 'MTP',
        'config': args.config,
        'checkpoint': args.checkpoint,
        'xml_dir': str(xml_dir),
        'images_dir': str(images_dir),
        'score_threshold': float(args.score_threshold),
        'keep_score_threshold': float(args.keep_score_threshold),
        'iou_threshold': float(args.iou_threshold),
        'processed': processed,
        'empty': empty,
    }
    (out_dir / 'pipeline_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[DONE] xml_files={processed} empty={empty} vis_pred_marker -> {vis_dir}')


if __name__ == '__main__':
    main()
