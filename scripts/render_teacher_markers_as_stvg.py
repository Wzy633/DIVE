#!/usr/bin/env python3
"""Render teacher DIOR marker JSONs into STVG-style images and marker meta.

Goal:
  - keep teacher detector boxes
  - redraw visuals so "same color => same class" holds
  - optionally filter noisy detections
  - optionally reindex IDs into a compact deterministic order

Input marker json format:
  {"image_name": "<stem>", "markers": [{"marker_id", "bbox", "class", "score", ...}, ...]}

Output:
  out_dir/
    images/<file_name>.jpg
    marker_meta_teacher_stvg.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render teacher markers as STVG-style images.")
    p.add_argument("--marker-json-dir", type=str, required=True, help="teacher vis_pred_marker dir")
    p.add_argument("--images-dir", type=str, required=True, help="original DIOR image dir")
    p.add_argument("--out-dir", type=str, required=True, help="output dir with images/ + marker meta")
    p.add_argument("--font-path", type=str, default="", help="optional .ttf font path")
    p.add_argument("--font-size", type=int, default=22)
    p.add_argument("--font-min-size", type=int, default=8)
    p.add_argument("--font-max-size", type=int, default=20)
    p.add_argument("--font-scale", type=float, default=0.16)
    p.add_argument("--stroke-width", type=int, default=2)
    p.add_argument("--box-width", type=int, default=2)
    p.add_argument("--score-thr", type=float, default=0.0)
    p.add_argument("--max-per-class", type=int, default=0, help="0 disables class-wise cap")
    p.add_argument("--max-total", type=int, default=0, help="0 disables image-wise cap")
    p.add_argument("--ann", type=str, default="", help="Optional DIOR COCO json for GT-aware filtering.")
    p.add_argument(
        "--min-gt-match-iou",
        type=float,
        default=-1.0,
        help="If >=0 and --ann is set, keep marker only when same-class GT max IoU >= this threshold.",
    )
    p.add_argument("--dynamic-font-size", action="store_true", help="Use bbox-aware dynamic font size.")
    p.add_argument("--draw-label-bg", action="store_true", help="Draw background box behind ID text.")
    p.add_argument(
        "--id-order",
        type=str,
        default="spatial",
        choices=("spatial", "score", "original"),
        help="How to assign compact marker IDs after filtering.",
    )
    p.add_argument(
        "--id-corner-mode",
        type=str,
        default="greedy_avoid",
        choices=("top_right", "random", "greedy_avoid"),
    )
    return p.parse_args()


def color_for_label_name(label_name: str) -> str:
    key = (label_name or "").strip() or "_empty_"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_LABEL_COLOR_PALETTE)
    return _LABEL_COLOR_PALETTE[idx]


def resolve_font_path(user_path: str) -> str:
    p = str(user_path or "").strip()
    if p and Path(p).is_file():
        return p
    # Prefer common Windows fonts first (local desktop usage), then Linux fallback.
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return ""


def _hex_to_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        return (255, 0, 0)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _darken_rgb(rgb: Tuple[int, int, int], factor: float = 0.20) -> Tuple[int, int, int]:
    f = max(0.0, min(1.0, float(factor)))
    r, g, b = rgb
    return (int(round(r * (1.0 - f))), int(round(g * (1.0 - f))), int(round(b * (1.0 - f))))


def _luma(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _label_bg_rgba(marker_rgb: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    # High-contrast black/white label background by marker color brightness.
    if _luma(marker_rgb) >= 140.0:
        return (0, 0, 0, 170)
    return (255, 255, 255, 190)


def _bbox_area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
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


def _centroid_xyxy(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]


def _sort_key_spatial(m: Dict[str, Any]) -> Tuple[float, float, float, int]:
    x1, y1, x2, y2 = m["bbox_xyxy"]
    area = _bbox_area_xyxy((x1, y1, x2, y2))
    return (-area, -float(y1), -float(x1), int(m.get("orig_marker_id", 0)))


def _sort_key_score(m: Dict[str, Any]) -> Tuple[float, int]:
    return (-float(m.get("score", 0.0)), int(m.get("orig_marker_id", 0)))


def load_markers(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stem = str(data.get("image_name", "") or path.stem).strip()
    out: List[Dict[str, Any]] = []
    for it in data.get("markers", []) or []:
        if not isinstance(it, dict):
            continue
        try:
            mid = int(it["marker_id"])
            b = it["bbox"]
            box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        except Exception:
            continue
        cls = str(it.get("class", "") or "").strip()
        if not cls:
            continue
        out.append(
            {
                "orig_marker_id": mid,
                "marker_id": mid,
                "bbox_xyxy": box,
                "label_name": cls,
                "score": float(it.get("score", 0.0) or 0.0),
                "centroid_xy": _centroid_xyxy(box),
            }
        )
    return stem, out


def load_gt_by_file_from_coco(ann_path: Path) -> Dict[str, Dict[str, List[List[float]]]]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    img_id_to_file = {int(im["id"]): str(im["file_name"]) for im in data.get("images", [])}
    out: Dict[str, Dict[str, List[List[float]]]] = {}
    for ann in data.get("annotations", []) or []:
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
        for cls, group in by_cls.items():
            del cls
            group = sorted(group, key=_sort_key_score)
            kept.extend(group[: int(max_per_class)])
        markers = kept
    if max_total and max_total > 0:
        markers = sorted(markers, key=_sort_key_score)[: int(max_total)]
    return markers


def filter_markers_by_gt_iou(
    markers: List[Dict[str, Any]],
    gt_by_class: Dict[str, List[List[float]]],
    min_gt_match_iou: float,
) -> List[Dict[str, Any]]:
    if float(min_gt_match_iou) < 0:
        return markers
    kept: List[Dict[str, Any]] = []
    for m in markers:
        cls = str(m.get("label_name", "") or "")
        gts = gt_by_class.get(cls, [])
        if not gts:
            continue
        best = 0.0
        for gb in gts:
            best = max(best, iou_xyxy(m["bbox_xyxy"], gb))
        if best >= float(min_gt_match_iou):
            kept.append(m)
    return kept


def filter_markers_by_gt_iou_one_to_one(
    markers: List[Dict[str, Any]],
    gt_by_class: Dict[str, List[List[float]]],
    min_gt_match_iou: float,
) -> List[Dict[str, Any]]:
    """Keep only markers selected by class-wise one-to-one matching with GT."""
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


def reorder_and_reindex(markers: List[Dict[str, Any]], id_order: str) -> List[Dict[str, Any]]:
    if id_order == "score":
        markers = sorted(markers, key=_sort_key_score)
    elif id_order == "spatial":
        markers = sorted(markers, key=_sort_key_spatial)
    else:
        markers = sorted(markers, key=lambda m: int(m.get("orig_marker_id", 0)))
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
    *,
    font_path: str,
    font_size: int,
    font_min_size: int,
    font_max_size: int,
    font_scale: float,
    stroke_width: int,
    box_width: int,
    dynamic_font_size: bool,
    id_corner_mode: str,
    draw_label_bg: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(im)
    resolved_font = resolve_font_path(font_path)
    try:
        if resolved_font:
            base_font = ImageFont.truetype(resolved_font, size=font_size)
        else:
            base_font = ImageFont.load_default()
    except OSError:
        base_font = ImageFont.load_default()

    placed_label_rects: List[Tuple[float, float, float, float]] = []
    overlap_ratio_thr = 0.20
    same_class_boxes: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for mm in markers:
        cls = str(mm.get("label_name", "") or "")
        if not cls:
            continue
        bx = mm.get("bbox_xyxy", None)
        if not isinstance(bx, list) or len(bx) != 4:
            continue
        try:
            same_class_boxes.setdefault(cls, []).append(
                (
                    int(mm.get("marker_id", -1)),
                    (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                )
            )
        except Exception:
            continue

    def _rect_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def _label_rect(center_xy: Tuple[float, float], text: str, font: Any) -> Tuple[float, float, float, float]:
        bb = draw.textbbox((0, 0), text, font=font, stroke_width=int(stroke_width))
        w = float(bb[2] - bb[0])
        h = float(bb[3] - bb[1])
        pad = 2.0
        cx, cy = center_xy
        return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)

    def _rect_fully_inside_box(rect: Tuple[float, float, float, float], box: Tuple[float, float, float, float]) -> bool:
        rx1, ry1, rx2, ry2 = rect
        bx1, by1, bx2, by2 = box
        return rx1 >= bx1 and ry1 >= by1 and rx2 <= bx2 and ry2 <= by2

    def _corner_candidates(image_key: str, mid: str, x1: float, y1: float, x2: float, y2: float) -> List[Tuple[str, Tuple[float, float]]]:
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

    for m in markers:
        box = m["bbox_xyxy"]
        x1, y1, x2, y2 = box
        rgb = _darken_rgb(_hex_to_rgb(color_for_label_name(str(m["label_name"]))))
        bg = _label_bg_rgba(rgb)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=rgb,
            width=max(1, int(box_width)),
        )

        mid = str(m["marker_id"])
        font = base_font
        if dynamic_font_size:
            short_edge = max(1.0, min(float(x2) - float(x1), float(y2) - float(y1)))
            dyn_size = int(round(short_edge * float(font_scale)))
            dyn_size = max(int(font_min_size), min(int(font_max_size), dyn_size))
            try:
                if resolved_font:
                    font = ImageFont.truetype(resolved_font, size=dyn_size)
                else:
                    font = base_font
            except OSError:
                font = base_font

        if id_corner_mode == "top_right":
            tx, ty = (float(x2), float(y1))
        elif id_corner_mode == "random":
            tx, ty = _corner_candidates(img_path.name, mid, float(x1), float(y1), float(x2), float(y2))[0][1]
        else:
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
                # Priority:
                # 1) avoid same-class "ID fully inside another bbox" (highest)
                # 2) then reduce ID-label overlap.
                score = same_cls_contain_cnt * 1_000_000.0 + ratio * 1000.0 + inter
                if same_cls_contain_cnt == 0 and ratio <= overlap_ratio_thr:
                    score -= 10000.0
                if score < best_score:
                    best_score = score
                    best_xy = (cx, cy)
            tx, ty = best_xy

        label_r = _label_rect((tx, ty), mid, font)
        placed_label_rects.append(label_r)
        if draw_label_bg:
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
                anchor="mm",
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


def draw_markers_cv2(
    img_path: Path,
    markers: List[Dict[str, Any]],
    out_path: Path,
    *,
    font_path: str,
    font_size: int,
    font_min_size: int,
    font_max_size: int,
    font_scale: float,
    stroke_width: int,
    box_width: int,
    dynamic_font_size: bool,
    id_corner_mode: str,
    draw_label_bg: bool,
) -> None:
    # Keep function name for backward compatibility; delegate to PIL renderer
    draw_markers_pil(
        img_path,
        markers,
        out_path,
        font_path=font_path,
        font_size=font_size,
        font_min_size=font_min_size,
        font_max_size=font_max_size,
        font_scale=font_scale,
        stroke_width=stroke_width,
        box_width=box_width,
        dynamic_font_size=dynamic_font_size,
        id_corner_mode=id_corner_mode,
        draw_label_bg=draw_label_bg,
    )


def main() -> None:
    args = parse_args()
    marker_dir = Path(args.marker_json_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    json_paths = sorted(marker_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"no marker json found in: {marker_dir}")
    gt_by_file: Dict[str, Dict[str, List[List[float]]]] = {}
    if str(args.ann or "").strip():
        gt_by_file = load_gt_by_file_from_coco(Path(args.ann))

    for jp in json_paths:
        stem, markers = load_markers(jp)
        if not markers:
            continue
        markers = filter_markers(
            markers,
            score_thr=float(args.score_thr),
            max_per_class=int(args.max_per_class),
            max_total=int(args.max_total),
        )
        if gt_by_file and float(args.min_gt_match_iou) >= 0:
            file_name = f"{stem}.jpg"
            markers = filter_markers_by_gt_iou_one_to_one(
                markers,
                gt_by_file.get(file_name, {}),
                float(args.min_gt_match_iou),
            )
        if not markers:
            continue
        markers = reorder_and_reindex(markers, args.id_order)

        image_file = f"{stem}.jpg"
        img_path = images_dir / image_file
        if not img_path.is_file():
            continue

        draw_markers_cv2(
            img_path,
            markers,
            out_images / image_file,
            font_path=str(args.font_path or ""),
            font_size=int(args.font_size),
            font_min_size=int(args.font_min_size),
            font_max_size=int(args.font_max_size),
            font_scale=float(args.font_scale),
            stroke_width=int(args.stroke_width),
            box_width=int(args.box_width),
            dynamic_font_size=bool(args.dynamic_font_size),
            id_corner_mode=str(args.id_corner_mode),
            draw_label_bg=bool(args.draw_label_bg),
        )

        instances = []
        for m in markers:
            instances.append(
                {
                    "marker_id": int(m["marker_id"]),
                    "bbox_xyxy": [float(x) for x in m["bbox_xyxy"]],
                    "label_name": str(m["label_name"]),
                    "score": float(m.get("score", 0.0)),
                    "centroid_xy": [float(x) for x in m["centroid_xy"]],
                    "orig_marker_id": int(m.get("orig_marker_id", m["marker_id"])),
                }
            )
        records.append({"file_name": image_file, "instances": instances})

    meta_path = out_dir / "marker_meta_teacher_stvg.json"
    meta_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] images={len(records)} -> {meta_path}")


if __name__ == "__main__":
    main()
