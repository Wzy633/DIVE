
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

DIOR_CLASSES = (
    'airplane', 'airport', 'groundtrackfield', 'harbor', 'baseballfield',
    'overpass', 'basketballcourt', 'ship', 'bridge', 'stadium',
    'storagetank', 'tenniscourt', 'expressway service area', 'trainstation',
    'expressway toll station', 'vehicle', 'golffield', 'windmill', 'chimney',
    'dam')


def infer_detector_name(config_path: str, checkpoint_path: str) -> str:
    cfg = str(config_path).lower()
    ckpt = str(checkpoint_path).lower()
    if 'mtp' in cfg or 'mtp' in ckpt or 'rvsa' in ckpt:
        return 'MTP'
    if 'grounding_dino' in cfg or 'groundingdino' in cfg or 'groundingdino' in ckpt:
        return 'GroundingDINO'
    if 'lae_dino' in cfg or 'laedino' in cfg or 'laedino' in ckpt:
        return 'LAE-DINO'
    return 'UnknownDetector'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STVG-R1-style numeric markers on DIOR using MTP detections."
    )
    p.add_argument("--config", type=str, required=True, help="MTP/DIOR detector config .py")
    p.add_argument("--checkpoint", type=str, required=True, help="MTP DIOR finetuned .pth")
    p.add_argument(
        "--ann",
        type=str,
        default="",
        help="Optional COCO json (DIOR). If set, iterate images in file order.",
    )
    p.add_argument("--img-dir", type=str, required=True, help="Root for JPEGImages-trainval")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0, help="Global random seed for reproducibility.")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable PyTorch/CUDA deterministic inference as much as possible. "
            "May slow down and can raise errors if non-deterministic ops are used."
        ),
    )
    p.add_argument("--score-thr", type=float, default=0.45, help="Confidence filter before NMS")
    p.add_argument("--nms-iou", type=float, default=0.5, help="Class-agnostic NMS IoU threshold")
    p.add_argument("--max-dets", type=int, default=100, help="Cap instances per image after NMS")
    p.add_argument("--max-images", type=int, default=0, help="If >0, only first N images")
    p.add_argument("--start-id", type=int, default=1, help="First marker ID per image")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--font", type=str, default=r"C:\Windows\Fonts\bahnschrift.ttf")
    p.add_argument("--font-size", type=int, default=22)
    p.add_argument("--dynamic-font-size", action="store_true", help="Use bbox-aware dynamic font size.")
    p.add_argument("--font-scale", type=float, default=0.16, help="Dynamic font size scale by bbox short edge.")
    p.add_argument("--font-min-size", type=int, default=8, help="Minimum dynamic font size.")
    p.add_argument("--font-max-size", type=int, default=20, help="Maximum dynamic font size.")
    p.add_argument("--color", type=str, default="#ff0000")
    p.add_argument(
        "--color-mode",
        type=str,
        choices=["fixed", "by_label"],
        default="by_label",
        help=(
            "fixed: all markers use --color (default red). "
            "by_label: bbox outline and digit use the same color per label_name "
            "(stable hash -> palette; same class same color across images)."
        ),
    )
    p.add_argument(
        "--color-darken",
        type=float,
        default=0.20,
        help="Darken marker colors by this factor in [0,1]. 0=no change, 0.2=slightly darker.",
    )
    p.add_argument("--stroke-width", type=int, default=2)
    p.add_argument(
        "--min-box-short-edge",
        type=float,
        default=0.0,
        help="Filter out tiny detections: require min(width,height) >= this value (pixels). 0 disables.",
    )
    p.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Filter out tiny detections: require bbox area >= this value (pixel^2). 0 disables.",
    )
    p.add_argument(
        "--max-instances-per-class",
        type=int,
        default=5,
        help=(
            "Cap number of kept instances per class (by label id) for each image. "
            "After NMS and tiny-box filtering, keep at most K instances per class (deterministic stable order). "
            "Set to 0 to disable the cap."
        ),
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=15,
        help="Cap total kept instances per image after per-class filtering; 0 disables.",
    )
    p.add_argument(
        "--min-gt-match-iou",
        type=float,
        default=0.6,
        help="Keep marker only when same-class GT one-to-one IoU >= this threshold; <0 disables.",
    )
    p.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Also draw bbox rectangles (for human/debug; STVG-R1 paper overlays digits only).",
    )
    p.add_argument("--box-width", type=int, default=2, help="Outline width when --draw-bboxes")
    p.add_argument(
        "--id-corner-mode",
        type=str,
        choices=["top_right", "random", "greedy_avoid"],
        default="greedy_avoid",
        help=(
            "Where to draw ID relative to bbox. "
            "top_right: fixed top-right; "
            "random: deterministic random corner; "
            "greedy_avoid: still corners, but greedily switch corner to reduce label overlaps."
        ),
    )
    p.add_argument(
        "--per-class-n",
        type=int,
        default=0,
        help="If >0, require --ann: pick N images per category from GT (union of images, deterministic).",
    )
    p.add_argument(
        "--organize-by-class",
        action="store_true",
        help="After saving, copy each selected image into by_class/<id>_<name>/ for the classes that chose it.",
    )
    return p.parse_args()


def set_deterministic(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _safe_class_dirname(category_id: int, category_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", category_name.strip())
    return f"{category_id:02d}_{safe}"


def build_per_class_image_plan(
    coco: Dict[str, Any], per_class_n: int
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    id2name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    img_id_to_file = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
    cat_to_img_ids: Dict[int, set] = defaultdict(set)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        cat_to_img_ids[int(ann["category_id"])].add(int(ann["image_id"]))

    per_class_files: Dict[str, List[str]] = {}
    for cid in sorted(cat_to_img_ids.keys()):
        ids_sorted = sorted(cat_to_img_ids[cid])
        picked_ids = ids_sorted[:per_class_n]
        name = id2name.get(cid, str(cid))
        per_class_files[name] = [img_id_to_file[i] for i in picked_ids if i in img_id_to_file]

    union_ids: set = set()
    for cid in sorted(cat_to_img_ids.keys()):
        ids_sorted = sorted(cat_to_img_ids[cid])
        for i in ids_sorted[:per_class_n]:
            union_ids.add(i)

    image_list = sorted(
        [(img_id_to_file[i], str(i)) for i in sorted(union_ids) if i in img_id_to_file],
        key=lambda x: x[0],
    )
    plan = {
        "per_class_pick": per_class_files,
        "num_unique_images": len(image_list),
        "categories_covered": len(per_class_files),
    }
    return image_list, plan


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


_LABEL_COLOR_PALETTE: Tuple[str, ...] = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#dc143c",
    "#228b22",
    "#ff1493",
    "#00ced1",
    "#daa520",
    "#4b0082",
    "#2e8b57",
    "#ff4500",
    "#1e90ff",
    "#8b4513",
    "#20b2aa",
    "#c71585",
)


def color_for_label_name(label_name: str) -> str:
    """Map the same label_name to a stable color via MD5 hashing."""
    key = (label_name or "").strip() or "_empty_"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_LABEL_COLOR_PALETTE)
    return _LABEL_COLOR_PALETTE[idx]


def _bbox_area_xyxy(b: torch.Tensor) -> float:
    x1, y1, x2, y2 = b.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centroid_xyxy(b: torch.Tensor) -> Tuple[float, float]:
    x1, y1, x2, y2 = b.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def nms_filter(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float,
    max_num: int,
) -> List[int]:
    """Class-agnostic NMS; return indices to keep (sorted by score desc)."""
    if boxes.numel() == 0:
        return []
    from torchvision.ops import nms

    keep = nms(boxes, scores, iou_thr)
    keep = keep[:max_num].tolist()
    return keep


def sort_instances_for_stable_ids(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> List[int]:
    """Deterministic order: area desc, then y desc, then x desc (aligns STVG-style instance ordering)."""
    n = boxes.shape[0]
    keys = []
    for i in range(n):
        b = boxes[i]
        area = _bbox_area_xyxy(b)
        x1, y1, x2, y2 = b.tolist()
        keys.append((-area, -y1, -x1, i))
    keys.sort()
    return [k[-1] for k in keys]


def load_gt_by_file_from_coco(coco: Dict[str, Any]) -> Dict[str, Dict[str, List[List[float]]]]:
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    img_id_to_file = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
    out: Dict[str, Dict[str, List[List[float]]]] = {}
    for ann in coco.get("annotations", []) or []:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann.get("image_id", -1))
        file_name = img_id_to_file.get(img_id, "")
        if not file_name:
            continue
        cls = cat_id_to_name.get(int(ann.get("category_id", -1)), "")
        if not cls:
            continue
        b = ann.get("bbox")
        if not isinstance(b, list) or len(b) != 4:
            continue
        x, y, w, h = [float(v) for v in b]
        out.setdefault(file_name, {}).setdefault(cls, []).append([x, y, x + w, y + h])
    return out


def iou_xyxy(a: List[float], b: List[float]) -> float:
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


def _sort_key_score(m: Dict[str, Any]) -> Tuple[float, int]:
    return (-float(m.get("score", 0.0)), int(m.get("orig_marker_id", 0)))


def _sort_key_spatial(m: Dict[str, Any]) -> Tuple[float, float, float, int]:
    x1, y1, x2, y2 = m["bbox_xyxy"]
    area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    return (-area, -float(y1), -float(x1), int(m.get("orig_marker_id", 0)))


def filter_markers(
    markers: List[Dict[str, Any]],
    *,
    score_thr: float,
    max_per_class: int,
    max_total: int,
) -> List[Dict[str, Any]]:
    markers = [m for m in markers if float(m.get("score", 0.0)) >= float(score_thr)]
    if max_per_class and max_per_class > 0:
        kept: List[Dict[str, Any]] = []
        by_cls: Dict[str, List[Dict[str, Any]]] = {}
        for m in markers:
            by_cls.setdefault(str(m["label_name"]), []).append(m)
        for _, group in by_cls.items():
            group = sorted(group, key=_sort_key_score)
            kept.extend(group[: int(max_per_class)])
        markers = kept
    if max_total and max_total > 0:
        markers = sorted(markers, key=_sort_key_score)[: int(max_total)]
    return markers


def filter_markers_by_gt_iou_one_to_one(
    markers: List[Dict[str, Any]],
    gt_by_class: Dict[str, List[List[float]]],
    min_gt_match_iou: float,
) -> List[Dict[str, Any]]:
    if float(min_gt_match_iou) < 0:
        return markers
    by_cls: Dict[str, List[Dict[str, Any]]] = {}
    for m in markers:
        cls = str(m.get("label_name", "") or "")
        if not cls:
            continue
        by_cls.setdefault(cls, []).append(m)

    kept: List[Dict[str, Any]] = []
    for cls, marker_group in by_cls.items():
        gt_group = gt_by_class.get(cls, [])
        if not gt_group or not marker_group:
            continue
        pairs: List[Tuple[float, int, int]] = []
        for gi, gt_box in enumerate(gt_group):
            for mi, mk in enumerate(marker_group):
                iou = iou_xyxy(mk["bbox_xyxy"], gt_box)
                if iou >= float(min_gt_match_iou):
                    pairs.append((iou, gi, mi))
        pairs.sort(reverse=True)
        used_g = set()
        used_m = set()
        for _, gi, mi in pairs:
            if gi in used_g or mi in used_m:
                continue
            used_g.add(gi)
            used_m.add(mi)
            kept.append(marker_group[mi])
    return kept


def reorder_and_reindex(markers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    markers = sorted(markers, key=_sort_key_spatial)
    out: List[Dict[str, Any]] = []
    for idx, m in enumerate(markers, start=1):
        mm = dict(m)
        mm["marker_id"] = idx
        out.append(mm)
    return out


def draw_markers_pil(
    img_path: Path,
    markers: List[Dict[str, Any]],
    out_path: Path,
    font_path: str,
    font_size: int,
    color: str,
    stroke_width: int,
    draw_bboxes: bool = False,
    box_width: int = 2,
    dynamic_font_size: bool = False,
    font_scale: float = 0.20,
    font_min_size: int = 10,
    font_max_size: int = 24,
    *,
    color_mode: str = "fixed",
    id_corner_mode: str = "random",
    color_darken: float = 0.20,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(im)
    try:
        base_font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        base_font = ImageFont.load_default()

    def _marker_color(m: Dict[str, Any]) -> str:
        if color_mode == "by_label":
            name = str(m.get("label_name", "") or "")
            return color_for_label_name(name)
        return color

    def _darken_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        f = max(0.0, min(1.0, float(factor)))
        r, g, b = rgb
        return (int(round(r * (1.0 - f))), int(round(g * (1.0 - f))), int(round(b * (1.0 - f))))

    def _pick_corner_xyxy(image_key: str, mid: str, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
        if id_corner_mode not in ("random", "greedy_avoid"):
            return float(x2), float(y1)
        seed = int(hashlib.md5(f"{image_key}::{mid}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        corner = rng.choice(("tr", "tl", "br", "bl"))
        if corner == "tr":
            return float(x2), float(y1)
        if corner == "tl":
            return float(x1), float(y1)
        if corner == "br":
            return float(x2), float(y2)
        return float(x1), float(y2)

    def _rect_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def _label_rect(center_xy: tuple[float, float], text: str, font: "ImageFont.FreeTypeFont") -> tuple[float, float, float, float]:
        bb = draw.textbbox((0, 0), text, font=font, stroke_width=int(stroke_width))
        w = float(bb[2] - bb[0])
        h = float(bb[3] - bb[1])
        pad = 2.0
        cx, cy = center_xy
        return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)

    def _rect_fully_inside_box(rect: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> bool:
        rx1, ry1, rx2, ry2 = rect
        bx1, by1, bx2, by2 = box
        return rx1 >= bx1 and ry1 >= by1 and rx2 <= bx2 and ry2 <= by2

    def _corner_candidates(image_key: str, mid: str, x1: float, y1: float, x2: float, y2: float) -> list[tuple[str, tuple[float, float]]]:
        seed = int(hashlib.md5(f"{image_key}::{mid}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        corners = ["tr", "tl", "br", "bl"]
        rng.shuffle(corners)
        m = {
            "tr": (float(x2), float(y1)),
            "tl": (float(x1), float(y1)),
            "br": (float(x2), float(y2)),
            "bl": (float(x1), float(y2)),
        }
        return [(c, m[c]) for c in corners]

    placed_label_rects: list[tuple[float, float, float, float]] = []
    overlap_ratio_thr = 0.20
    same_class_boxes: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for mm in markers:
        cls = str(mm.get("label_name", "") or "")
        bx = mm.get("bbox_xyxy")
        if not cls or not isinstance(bx, list) or len(bx) != 4:
            continue
        same_class_boxes.setdefault(cls, []).append((int(mm.get("marker_id", -1)), (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))))

    def _luma(rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _label_bg_rgba(marker_rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
        if _luma(marker_rgb) >= 140.0:
            return (0, 0, 0, 170)
        return (255, 255, 255, 190)

    for m in markers:
        mc = _marker_color(m)
        rgb = _darken_rgb(_hex_to_rgb(mc), color_darken)
        bg = _label_bg_rgba(rgb)
        if draw_bboxes and "bbox_xyxy" in m:
            x1, y1, x2, y2 = m["bbox_xyxy"]
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=rgb,
                width=max(1, int(box_width)),
            )
        mid = str(m["marker_id"])
        font = base_font
        if dynamic_font_size and "bbox_xyxy" in m:
            x1, y1, x2, y2 = m["bbox_xyxy"]
            short_edge = max(1.0, min(float(x2) - float(x1), float(y2) - float(y1)))
            dyn_size = int(round(short_edge * float(font_scale)))
            dyn_size = max(int(font_min_size), min(int(font_max_size), dyn_size))
            try:
                font = ImageFont.truetype(font_path, size=dyn_size)
            except OSError:
                font = base_font
        if "bbox_xyxy" in m:
            x1, y1, x2, y2 = m["bbox_xyxy"]
            if id_corner_mode == "greedy_avoid":
                best_xy = (float(x2), float(y1))
                best_score = 1e18
                cls_name = str(m.get("label_name", "") or "")
                self_mid = int(m.get("marker_id", -1))
                for _, (cx, cy) in _corner_candidates(img_path.name, mid, float(x1), float(y1), float(x2), float(y2)):
                    r = _label_rect((cx, cy), mid, font)
                    same_cls_contain_cnt = 0
                    for omid, obox in same_class_boxes.get(cls_name, []):
                        if omid == self_mid:
                            continue
                        if _rect_fully_inside_box(r, obox):
                            same_cls_contain_cnt += 1
                    inter = sum(_rect_intersection_area(r, pr) for pr in placed_label_rects)
                    area = max(1.0, (r[2] - r[0]) * (r[3] - r[1]))
                    ratio = inter / area
                    score = same_cls_contain_cnt * 1_000_000.0 + ratio * 1000.0 + inter
                    if same_cls_contain_cnt == 0 and ratio <= overlap_ratio_thr:
                        score -= 10000.0
                    if score < best_score:
                        best_score = score
                        best_xy = (cx, cy)
                tx, ty = best_xy
            else:
                tx, ty = _pick_corner_xyxy(img_path.name, mid, float(x1), float(y1), float(x2), float(y2))
            anchor = "mm"
        else:
            cx, cy = m["centroid_xy"]
            tx, ty = float(cx), float(cy)
            anchor = "mm"

        label_r = _label_rect((tx, ty), mid, font)
        placed_label_rects.append(label_r)

        try:
            draw.rounded_rectangle(label_r, radius=3, fill=bg, outline=None)
        except Exception:
            draw.rectangle(label_r, fill=bg, outline=None)

        try:
            draw.text(
                (tx, ty),
                mid,
                font=font,
                fill=rgb,
                stroke_width=stroke_width,
                stroke_fill="#000000",
                anchor=anchor,
            )
        except TypeError:
            bbox = draw.textbbox((0, 0), mid, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw_x = tx - w / 2
            draw_y = ty - h / 2
            draw.text(
                (draw_x, draw_y),
                mid,
                font=font,
                fill=rgb,
                stroke_width=stroke_width,
                stroke_fill="#000000",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(out_path)


def main() -> None:
    args = parse_args()
    set_deterministic(seed=int(args.seed), deterministic=bool(args.deterministic))
    ann_path = Path(args.ann) if args.ann else None
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(str(_MMDET_ROOT))

    model = init_detector(load_mtp_inference_config(args.config), args.checkpoint, device=args.device)
    detector_name = infer_detector_name(args.config, args.checkpoint)
    classes = model.dataset_meta.get("classes")
    if not classes or len(classes) != len(DIOR_CLASSES):
        classes = DIOR_CLASSES
        model.dataset_meta = dict(model.dataset_meta or {})
        model.dataset_meta["classes"] = classes

    image_list: List[Tuple[str, str]] = []
    per_class_plan: Optional[Dict[str, Any]] = None
    coco_for_plan: Optional[Dict[str, Any]] = None
    gt_by_file: Dict[str, Dict[str, List[List[float]]]] = {}

    if args.per_class_n and args.per_class_n > 0:
        if not ann_path or not ann_path.is_file():
            raise SystemExit("--per-class-n requires a valid --ann COCO json.")
        with ann_path.open("r", encoding="utf-8") as f:
            coco_for_plan = json.load(f)
        gt_by_file = load_gt_by_file_from_coco(coco_for_plan)
        image_list, per_class_plan = build_per_class_image_plan(coco_for_plan, args.per_class_n)
        pick_path = out_dir / "per_class_pick.json"
        with pick_path.open("w", encoding="utf-8") as f:
            json.dump(per_class_plan, f, ensure_ascii=False, indent=2)
        print(
            f"[INFO] per-class-n={args.per_class_n}: {per_class_plan['num_unique_images']} unique images "
            f"-> {pick_path}",
            flush=True,
        )
    elif ann_path and ann_path.is_file():
        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        gt_by_file = load_gt_by_file_from_coco(coco)
        seen_names = set()
        dup_names = 0
        for im in coco.get("images", []):
            fn = str(im["file_name"])
            if fn in seen_names:
                dup_names += 1
                continue
            seen_names.add(fn)
            image_list.append((fn, str(im.get("id", ""))))
        if dup_names > 0:
            print(
                f"[INFO] COCO images contains duplicated file_name entries; "
                f"skip_duplicates={dup_names}, keep_unique={len(image_list)}",
                flush=True,
            )
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in exts:
                image_list.append((p.name, ""))

    if args.max_images and args.max_images > 0:
        image_list = image_list[: args.max_images]

    meta_all: List[Dict[str, Any]] = []
    qa_all: List[Dict[str, Any]] = []

    for file_name, img_id in image_list:
        img_path = img_dir / file_name
        if not img_path.is_file():
            print(f"[WARN] missing image: {img_path}", flush=True)
            continue

        result = inference_detector(
            model,
            str(img_path),
            text_prompt=classes,
            custom_entities=True,
        )
        pi = result.pred_instances
        if len(pi) == 0:
            meta_all.append(
                {
                    "file_name": file_name,
                    "image_id": img_id,
                    "detector": detector_name,
                    "instances": [],
                }
            )
            if args.save_images:
                out_img = out_dir / "images" / file_name
                out_img.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(img_path, out_img)
                except Exception as e:
                    print(f"[WARN] failed to copy empty-instance image {img_path} -> {out_img}: {e}", flush=True)
            continue

        boxes = pi.bboxes
        scores = pi.scores
        labels = pi.labels
        n_all = len(scores)
        if hasattr(pi, "label_names") and pi.label_names is not None:
            names = list(pi.label_names)
        else:
            names = [classes[int(labels[i].item())] for i in range(n_all)]

        keep_mask = scores >= args.score_thr
        keep_np = keep_mask.cpu().numpy().astype(bool)
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]
        names = [names[i] for i in range(n_all) if keep_np[i]]

        if boxes.numel() == 0:
            meta_all.append(
                {
                    "file_name": file_name,
                    "image_id": img_id,
                    "detector": detector_name,
                    "instances": [],
                }
            )
            continue

        idx = nms_filter(boxes, scores, args.nms_iou, args.max_dets)
        boxes = boxes[idx]
        scores = scores[idx]
        labels = labels[idx]
        names = [names[i] for i in idx]

        if float(args.min_box_short_edge) > 0.0 or float(args.min_box_area) > 0.0:
            keep = []
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i].tolist()
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                short_edge = min(w, h)
                area = w * h
                if float(args.min_box_short_edge) > 0.0 and short_edge < float(args.min_box_short_edge):
                    continue
                if float(args.min_box_area) > 0.0 and area < float(args.min_box_area):
                    continue
                keep.append(i)
            if keep:
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                names = [names[i] for i in keep]
            else:
                boxes = boxes[:0]
                scores = scores[:0]
                labels = labels[:0]
                names = []

        raw_markers: List[Dict[str, Any]] = []
        order = sort_instances_for_stable_ids(boxes, scores, labels)
        for oi in order:
            b = boxes[oi]
            cx, cy = _centroid_xyxy(b)
            lab = int(labels[oi].item())
            name = names[oi] if names else str(lab)
            raw_markers.append(
                {
                    "orig_marker_id": int(oi) + int(args.start_id),
                    "marker_id": int(oi) + int(args.start_id),
                    "bbox_xyxy": [float(x) for x in b.tolist()],
                    "score": float(scores[oi].item()),
                    "label": lab,
                    "label_name": name,
                    "centroid_xy": [cx, cy],
                }
            )

        markers = filter_markers(
            raw_markers,
            score_thr=float(args.score_thr),
            max_per_class=int(args.max_instances_per_class),
            max_total=int(args.max_total),
        )
        if gt_by_file and float(args.min_gt_match_iou) >= 0:
            markers = filter_markers_by_gt_iou_one_to_one(
                markers,
                gt_by_file.get(file_name, {}),
                float(args.min_gt_match_iou),
            )
        if markers:
            markers = reorder_and_reindex(markers)

        if not markers:
            meta_all.append(
                {
                    "file_name": file_name,
                    "image_id": img_id,
                    "detector": detector_name,
                    "instances": [],
                }
            )
            if args.save_images:
                out_img = out_dir / "images" / file_name
                out_img.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(img_path, out_img)
                except Exception as e:
                    print(f"[WARN] failed to copy empty-instance image {img_path} -> {out_img}: {e}", flush=True)
            continue

        for m in markers:
            mid = int(m["marker_id"])
            name = str(m["label_name"])
            if args.id_corner_mode == "random":
                pos_hint = "near one of its bbox corners (corner chosen deterministically per ID)"
            else:
                pos_hint = "near its bbox top-right corner"
            vis_prompt = (
                f"Each object is marked with a colored number {pos_hint}; "
                "box outline and digit share one color per class; same color means the same label_name."
                if args.color_mode == "by_label"
                else f"Each object in the image is marked with a red number {pos_hint}."
            )
            qa_all.append(
                {
                    "file_name": file_name,
                    "prompt": vis_prompt,
                    "question": f"Where is the {name}?",
                    "answer": f"Target ID: {mid}",
                    "marker_id": mid,
                }
            )

        rec = {
            "file_name": file_name,
            "image_id": img_id,
            "detector": detector_name,
            "stvg_note": "Static DIOR: LAE bbox center as visual prompt location (cf. STVG-R1 mask centroid).",
            "instances": markers,
        }
        meta_all.append(rec)

        if args.save_images:
            draw_markers_pil(
                img_path,
                markers,
                out_dir / "images" / file_name,
                args.font,
                args.font_size,
                args.color,
                args.stroke_width,
                draw_bboxes=args.draw_bboxes,
                box_width=args.box_width,
                dynamic_font_size=args.dynamic_font_size,
                font_scale=args.font_scale,
                font_min_size=args.font_min_size,
                font_max_size=args.font_max_size,
                color_mode=args.color_mode,
                id_corner_mode=args.id_corner_mode,
                color_darken=float(args.color_darken),
            )

    meta_path = out_dir / "marker_meta_lae_stvg.json"
    qa_path = out_dir / "qa_stvg_style.json"
    manifest_path = out_dir / "pipeline_manifest.json"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa_all, f, ensure_ascii=False, indent=2)

    manifest = {
        "stvg_r1_reference": "Paper 鎼?.2: detector -> SAM2+track -> re-detect; centroid numeric prompts.",
        "dior_implementation": (
            f"{detector_name} -> score_thr + NMS -> stable IDs -> digit near bbox corners; "
            f"color_mode={args.color_mode}, id_corner_mode={args.id_corner_mode}."
        ),
        "color_mode": args.color_mode,
        "label_color_palette_size": len(_LABEL_COLOR_PALETTE),
        "qwen_system_append_suggestion": (
            "Each class uses one stable color for both boxes and ID digits; focus on the class color named in the query."
            if args.color_mode == "by_label"
            else ""
        ),
        "draw_bboxes_on_image": args.draw_bboxes,
        "min_box_short_edge": float(args.min_box_short_edge),
        "min_box_area": float(args.min_box_area),
        "max_instances_per_class": int(args.max_instances_per_class),
        "max_total": int(args.max_total),
        "min_gt_match_iou": float(args.min_gt_match_iou),
        "dynamic_font_size": args.dynamic_font_size,
        "font_scale": args.font_scale,
        "font_min_size": args.font_min_size,
        "font_max_size": args.font_max_size,
        "note_stvg_visual_input": "STVG-R1 uses centroid digits on frames; instance extent is in mask DB M, not drawn as boxes on video input.",
        "per_class_n": args.per_class_n,
        "organize_by_class": args.organize_by_class,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "score_thr": args.score_thr,
        "nms_iou": args.nms_iou,
        "max_dets": args.max_dets,
        "num_images": len(meta_all),
        "num_qa": len(qa_all),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] images processed: {len(meta_all)} -> {meta_path}")
    print(f"[DONE] qa pairs: {len(qa_all)} -> {qa_path}")
    if args.save_images:
        print(f"[DONE] marked images -> {out_dir / 'images'}")

    if args.save_images and args.organize_by_class and per_class_plan and coco_for_plan:
        id2name = {int(c["id"]): str(c["name"]) for c in coco_for_plan.get("categories", [])}
        by_root = out_dir / "by_class"
        by_root.mkdir(parents=True, exist_ok=True)
        picks: Dict[str, List[str]] = per_class_plan.get("per_class_pick", {})
        for cat_name, files in picks.items():
            cid = next((k for k, v in id2name.items() if v == cat_name), 0)
            sub = by_root / _safe_class_dirname(cid, cat_name)
            sub.mkdir(parents=True, exist_ok=True)
            for fn in files:
                src = out_dir / "images" / fn
                if src.is_file():
                    shutil.copy2(src, sub / fn)
        print(f"[DONE] copies by class -> {by_root}")


if __name__ == "__main__":
    main()











