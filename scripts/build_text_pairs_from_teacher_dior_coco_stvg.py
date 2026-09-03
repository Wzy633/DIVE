#!/usr/bin/env python3
"""Build GT-backed STVG-style text pairs for DIOR using teacher marker meta.

This script is intended for a stronger DIOR migration of the teacher pipeline:
  teacher detector boxes -> STVG-style marker visuals -> GT-backed text pairs -> VLM

Unlike the original description-style builder, this generates simpler STVG-style
questions and stores both marker bbox (`bbox_xyxy`) and GT bbox (`gt_bbox_xyxy`)
so evaluation can use real GT IoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dior_label_zh import dior_class_bilingual
from text_pair_cn_ordinal import cn_ordinal_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GT-backed teacher STVG text pairs for DIOR.")
    p.add_argument("--ann", type=str, required=True, help="DIOR COCO json")
    p.add_argument("--marker-meta", type=str, required=True, help="marker_meta_teacher_stvg.json")
    p.add_argument("--out-jsonl", type=str, required=True)
    p.add_argument("--min-match-iou", type=float, default=0.6)
    p.add_argument("--min-rank-rel-diff", type=float, default=0.08)
    p.add_argument("--min-area-rel-diff", type=float, default=0.35)
    p.add_argument(
        "--enable-area-questions",
        action="store_true",
        help="Enable area max/min questions. Default off for higher DIOR robustness.",
    )
    p.add_argument(
        "--enable-fallback-cover",
        action="store_true",
        help="Enable fallback cover questions to force all IDs appear at least once.",
    )
    p.add_argument(
        "--dense-classes",
        type=str,
        default="ship,vehicle,harbor",
        help="Comma-separated class names treated as dense/ambiguous.",
    )
    p.add_argument(
        "--dense-rank-mode",
        type=str,
        default="extreme_only",
        choices=("extreme_only", "disable_rank", "full"),
        help="Rank question policy for dense classes.",
    )
    p.add_argument(
        "--dense-enable-tb",
        action="store_true",
        help="Allow top-bottom rank questions for dense classes.",
    )
    p.add_argument(
        "--max-same-class-for-rank",
        type=int,
        default=3,
        help="Only generate rank questions when same-class candidate count is <= this value.",
    )
    p.add_argument(
        "--rank-position-policy",
        type=str,
        default="extreme_when_ge3",
        choices=("full", "extreme_only", "extreme_when_ge3"),
        help="Whether to keep all rank positions or prefer only extreme positions.",
    )
    p.add_argument(
        "--append-candidate-ids",
        action="store_true",
        help="Append valid candidate IDs of the target class to the question.",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(x) for x in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _sorted_lr(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: (float(x["centroid_xy"][0]), float(x["centroid_xy"][1])))


def _sorted_tb(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: (float(x["centroid_xy"][1]), float(x["centroid_xy"][0])))


def _rank_by_area(group: List[Dict[str, Any]], reverse: bool) -> List[Dict[str, Any]]:
    return sorted(group, key=lambda x: _area_xyxy(x["bbox_xyxy"]), reverse=reverse)


def _min_adjacent_rel_gap(values: List[float], denom: float) -> float:
    if len(values) < 2 or denom <= 0:
        return 0.0
    vals = sorted(values)
    gaps = [abs(vals[i + 1] - vals[i]) / denom for i in range(len(vals) - 1)]
    return min(gaps) if gaps else 0.0


def _expected(mid: int) -> str:
    return f"<answer>{mid}</answer>"


def _candidate_ids_hint(matched_group: List[Dict[str, Any]], enabled: bool) -> str:
    if not enabled:
        return ""
    ids = [str(int(ins["marker_id"])) for ins in matched_group]
    if not ids:
        return ""
    return f" 候选ID：{', '.join(ids)}。"


def _rank_items_by_policy(
    ordered_group: List[Dict[str, Any]],
    policy: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    n = len(ordered_group)
    if n == 0:
        return []
    if policy == "full" or n == 1:
        return list(enumerate(ordered_group, start=1))
    if policy == "extreme_only":
        if n == 2:
            return [(1, ordered_group[0]), (2, ordered_group[1])]
        return [(1, ordered_group[0]), (n, ordered_group[-1])]
    if n >= 3:
        return [(1, ordered_group[0]), (n, ordered_group[-1])]
    return list(enumerate(ordered_group, start=1))


def match_gt_to_markers_one_to_one(
    gt_items: List[Dict[str, Any]],
    marker_items: List[Dict[str, Any]],
    min_match_iou: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_items):
        for mi, mk in enumerate(marker_items):
            iou = iou_xyxy(gt["gt_bbox_xyxy"], mk["bbox_xyxy"])
            if iou >= float(min_match_iou):
                pairs.append((iou, gi, mi))
    pairs.sort(reverse=True)

    used_g = set()
    used_m = set()
    out: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for iou, gi, mi in pairs:
        if gi in used_g or mi in used_m:
            continue
        used_g.add(gi)
        used_m.add(mi)
        out.append((gt_items[gi], marker_items[mi], float(iou)))
    return out


def build_samples_for_class(
    file_name: str,
    cls_name: str,
    matched_group: List[Dict[str, Any]],
    gt_count: int,
    marker_count: int,
    min_rank_rel_diff: float,
    min_area_rel_diff: float,
    enable_area_questions: bool,
    enable_fallback_cover: bool,
    dense_classes_norm: set[str],
    dense_rank_mode: str,
    dense_enable_tb: bool,
    max_same_class_for_rank: int,
    rank_position_policy: str,
    append_candidate_ids: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(matched_group)
    if n == 0:
        return out

    ql = dior_class_bilingual(cls_name)
    only_id_hint = "请仅输出目标ID数字。"
    candidate_hint = _candidate_ids_hint(matched_group, append_candidate_ids)
    is_dense = str(cls_name).strip().lower() in dense_classes_norm

    if n == 1 and int(gt_count) == 1 and int(marker_count) == 1:
        ins = matched_group[0]
        mid = int(ins["marker_id"])
        out.append(
            {
                "image_file": file_name,
                "question": f"请定位图中唯一的{ql}，并给出其目标ID。{only_id_hint}{candidate_hint}",
                "target_marker_id": mid,
                "target_class": cls_name,
                "bbox_xyxy": [float(x) for x in ins["bbox_xyxy"]],
                "gt_bbox_xyxy": [float(x) for x in ins["gt_bbox_xyxy"]],
                "match_iou": float(ins["match_iou"]),
                "source": "teacher_stvg_single",
                "expected_answer": _expected(mid),
                "expected_response_template": "<think>根据图中目标的空间位置与描述约束进行匹配。</think>"
                + _expected(mid),
            }
        )
        return out

    lr = _sorted_lr(matched_group)
    xs = [float(ins["centroid_xy"][0]) for ins in matched_group]
    if n <= max_same_class_for_rank and _min_adjacent_rel_gap(xs, 800.0) > min_rank_rel_diff:
        lr_items: List[Tuple[int, Dict[str, Any]]] = _rank_items_by_policy(lr, rank_position_policy)
        if is_dense and dense_rank_mode == "disable_rank":
            lr_items = []
        elif is_dense and dense_rank_mode == "extreme_only":
            lr_items = _rank_items_by_policy(lr, "extreme_only")
        for rank, ins in lr_items:
            cn = cn_ordinal_rank(rank)
            mid = int(ins["marker_id"])
            out.append(
                {
                    "image_file": file_name,
                    "question": f"图中存在多个{ql}。请返回从左到右{cn}{ql}的目标ID。{only_id_hint}{candidate_hint}",
                    "target_marker_id": mid,
                    "target_class": cls_name,
                    "bbox_xyxy": [float(x) for x in ins["bbox_xyxy"]],
                    "gt_bbox_xyxy": [float(x) for x in ins["gt_bbox_xyxy"]],
                    "match_iou": float(ins["match_iou"]),
                    "source": "teacher_stvg_lr_rank",
                    "expected_answer": _expected(mid),
                    "expected_response_template": f"<think>先对同类目标按左右顺序排序，再选择{cn}。</think>"
                    + _expected(mid),
                }
            )

    tb = _sorted_tb(matched_group)
    ys = [float(ins["centroid_xy"][1]) for ins in matched_group]
    if (
        n <= max_same_class_for_rank
        and (not is_dense or dense_enable_tb)
        and _min_adjacent_rel_gap(ys, 800.0) > min_rank_rel_diff
    ):
        tb_items: List[Tuple[int, Dict[str, Any]]] = _rank_items_by_policy(tb, rank_position_policy)
        if is_dense and dense_rank_mode == "disable_rank":
            tb_items = []
        elif is_dense:
            tb_items = _rank_items_by_policy(tb, "extreme_only")
        for rank, ins in tb_items:
            cn = cn_ordinal_rank(rank)
            mid = int(ins["marker_id"])
            out.append(
                {
                    "image_file": file_name,
                    "question": f"请在所有{ql}中，找出从上到下{cn}目标，并输出其ID。{only_id_hint}{candidate_hint}",
                    "target_marker_id": mid,
                    "target_class": cls_name,
                    "bbox_xyxy": [float(x) for x in ins["bbox_xyxy"]],
                    "gt_bbox_xyxy": [float(x) for x in ins["gt_bbox_xyxy"]],
                    "match_iou": float(ins["match_iou"]),
                    "source": "teacher_stvg_tb_rank",
                    "expected_answer": _expected(mid),
                    "expected_response_template": f"<think>先对同类目标按上下顺序排序，再选择{cn}。</think>"
                    + _expected(mid),
                }
            )

    max_ins = _rank_by_area(matched_group, reverse=True)[0]
    min_ins = _rank_by_area(matched_group, reverse=False)[0]
    max_area = _area_xyxy(max_ins["bbox_xyxy"])
    min_area = _area_xyxy(min_ins["bbox_xyxy"])
    area_rel_diff = (max_area - min_area) / max_area if max_area > 0 else 0.0
    if enable_area_questions and (not is_dense) and area_rel_diff > min_area_rel_diff:
        max_mid = int(max_ins["marker_id"])
        min_mid = int(min_ins["marker_id"])
        out.append(
            {
                "image_file": file_name,
                "question": f"在图中所有{ql}里，面积最大的目标是哪一个？请输出目标ID。{only_id_hint}{candidate_hint}",
                "target_marker_id": max_mid,
                "target_class": cls_name,
                "bbox_xyxy": [float(x) for x in max_ins["bbox_xyxy"]],
                "gt_bbox_xyxy": [float(x) for x in max_ins["gt_bbox_xyxy"]],
                "match_iou": float(max_ins["match_iou"]),
                "source": "teacher_stvg_area_max",
                "expected_answer": _expected(max_mid),
                "expected_response_template": "<think>比较同类目标的面积，选择面积最大的实例。</think>"
                + _expected(max_mid),
            }
        )
        out.append(
            {
                "image_file": file_name,
                "question": f"在图中所有{ql}里，面积最小的目标是哪一个？请输出目标ID。{only_id_hint}{candidate_hint}",
                "target_marker_id": min_mid,
                "target_class": cls_name,
                "bbox_xyxy": [float(x) for x in min_ins["bbox_xyxy"]],
                "gt_bbox_xyxy": [float(x) for x in min_ins["gt_bbox_xyxy"]],
                "match_iou": float(min_ins["match_iou"]),
                "source": "teacher_stvg_area_min",
                "expected_answer": _expected(min_mid),
                "expected_response_template": "<think>比较同类目标的面积，选择面积最小的实例。</think>"
                + _expected(min_mid),
            }
        )

    covered = {int(s["target_marker_id"]) for s in out}
    if enable_fallback_cover and len(covered) < n:
        rank_map = {int(ins["marker_id"]): idx + 1 for idx, ins in enumerate(lr)}
        for ins in lr:
            mid = int(ins["marker_id"])
            if mid in covered:
                continue
            rank = rank_map[mid]
            cn = cn_ordinal_rank(rank)
            out.append(
                {
                    "image_file": file_name,
                    "question": f"图中存在多个{ql}。请返回从左到右{cn}{ql}的目标ID。{only_id_hint}{candidate_hint}",
                    "target_marker_id": mid,
                    "target_class": cls_name,
                    "bbox_xyxy": [float(x) for x in ins["bbox_xyxy"]],
                    "gt_bbox_xyxy": [float(x) for x in ins["gt_bbox_xyxy"]],
                    "match_iou": float(ins["match_iou"]),
                    "source": "teacher_stvg_lr_rank_fallback_cover",
                    "expected_answer": _expected(mid),
                    "expected_response_template": f"<think>按同类目标从左到右排序，选择{cn}。</think>"
                    + _expected(mid),
                }
            )
            covered.add(mid)

    return out


def main() -> None:
    args = parse_args()
    coco = load_json(Path(args.ann))
    marker_meta = load_json(Path(args.marker_meta))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dense_classes_norm = {
        s.strip().lower() for s in str(args.dense_classes or "").split(",") if s.strip()
    }

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    img_id_to_file = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []) or []:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann["image_id"])
        if img_id not in img_id_to_file:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    meta_by_file = {str(r.get("file_name", "")): r for r in marker_meta if r.get("file_name")}

    n_images = 0
    n_samples = 0
    with out_path.open("w", encoding="utf-8") as w:
        for img_id, file_name in img_id_to_file.items():
            rec = meta_by_file.get(file_name)
            if not rec:
                continue
            n_images += 1
            instances = rec.get("instances", []) or []
            by_cls_marker: Dict[str, List[Dict[str, Any]]] = {}
            for ins in instances:
                cls = str(ins.get("label_name", "") or "")
                if not cls:
                    continue
                by_cls_marker.setdefault(cls, []).append(ins)

            by_cls_gt: Dict[str, List[Dict[str, Any]]] = {}
            for ann in anns_by_img.get(int(img_id), []):
                cid = int(ann["category_id"])
                cls = cat_id_to_name.get(cid, str(cid))
                b = ann.get("bbox")
                if not isinstance(b, list) or len(b) != 4:
                    continue
                x, y, bw, bh = [float(v) for v in b]
                by_cls_gt.setdefault(cls, []).append(
                    {
                        "gt_bbox_xyxy": [x, y, x + bw, y + bh],
                        "ann_id": int(ann.get("id", -1)),
                    }
                )

            for cls_name, gt_group in by_cls_gt.items():
                marker_group = by_cls_marker.get(cls_name, [])
                if not marker_group:
                    continue
                matched_pairs = match_gt_to_markers_one_to_one(
                    gt_group,
                    marker_group,
                    float(args.min_match_iou),
                )
                matched_group: List[Dict[str, Any]] = []
                for gt, mk, biou in matched_pairs:
                    matched_group.append(
                        {
                            "marker_id": int(mk["marker_id"]),
                            "bbox_xyxy": [float(x) for x in mk["bbox_xyxy"]],
                            "centroid_xy": [float(x) for x in mk["centroid_xy"]],
                            "gt_bbox_xyxy": [float(x) for x in gt["gt_bbox_xyxy"]],
                            "match_iou": float(biou),
                        }
                    )
                samples = build_samples_for_class(
                    file_name=file_name,
                    cls_name=cls_name,
                    matched_group=matched_group,
                    gt_count=len(gt_group),
                    marker_count=len(marker_group),
                    min_rank_rel_diff=float(args.min_rank_rel_diff),
                    min_area_rel_diff=float(args.min_area_rel_diff),
                    enable_area_questions=bool(args.enable_area_questions),
                    enable_fallback_cover=bool(args.enable_fallback_cover),
                    dense_classes_norm=dense_classes_norm,
                    dense_rank_mode=str(args.dense_rank_mode),
                    dense_enable_tb=bool(args.dense_enable_tb),
                    max_same_class_for_rank=int(args.max_same_class_for_rank),
                    rank_position_policy=str(args.rank_position_policy),
                    append_candidate_ids=bool(args.append_candidate_ids),
                )
                for s in samples:
                    w.write(json.dumps(s, ensure_ascii=False) + "\n")
                    n_samples += 1

    print(f"[DONE] images={n_images} samples={n_samples} -> {out_path}")


if __name__ == "__main__":
    main()
