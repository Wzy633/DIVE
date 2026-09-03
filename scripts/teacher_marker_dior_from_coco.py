#!/usr/bin/env python3
"""Create teacher-style prediction JSONs for DIOR using COCO annotations.

Teacher pipeline expects:
  image_demo_dior_multiple -> outputs/preds/<stem>.json with `label_to_class`
  visualize_marker.py   -> reads preds + XML (for filename) to draw markers

DIOR is COCO json (no XML). This script:
  - reads COCO annotations
  - for each image: builds text prompt from *GT categories in that image*
  - runs DetInferencer once per image (like teacher xml script)
  - writes prediction json under out_dir/preds/<stem>.json
  - writes an XML stub under out_dir/../xml_stub/<stem>.xml containing <filename>
so the same marker renderer can be reused without modification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from mmengine.logging import print_log
from mmdet.apis import DetInferencer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", type=str, required=True, help="DIOR COCO json (train or val)")
    p.add_argument("--img-dir", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, required=True, help="Will create preds/ and vis/")
    p.add_argument("--pred-score-thr", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_coco(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_maps(coco: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, List[Dict[str, Any]]]]:
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    img_id_to_file = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann["image_id"])
        anns_by_img.setdefault(img_id, []).append(ann)
    return cat_id_to_name, img_id_to_file, anns_by_img


def write_xml_stub(xml_path: Path, filename: str, class_names: List[str]) -> None:
    """Write a minimal XML compatible with ``visualize_marker.py``.

    The teacher visualization script only uses:
    - ``<filename>`` to locate the image
    - ``<object><name>...`` to build ``gt_name_set`` and decide red/green colors

    Therefore, for DIOR-from-COCO migration we must preserve the GT category names
    in the stub XML; otherwise all predicted boxes are treated as non-matching and
    become green.
    """
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    uniq_names = sorted({str(x).strip() for x in class_names if str(x).strip()})
    object_lines = []
    for name in uniq_names:
        object_lines.append(f"  <object>\n    <name>{name}</name>\n  </object>")
    objects_block = "\n".join(object_lines)
    if objects_block:
        objects_block = "\n" + objects_block + "\n"
    xml_path.write_text(
        (
            '<?xml version="1.0" ?>\n'
            "<annotation>\n"
            f"  <filename>{filename}</filename>"
            f"{objects_block}"
            "</annotation>\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ann_path = Path(args.ann)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(ann_path)
    cat_id_to_name, img_id_to_file, anns_by_img = build_maps(coco)

    # init inferencer
    inferencer = DetInferencer(model=args.model, weights=args.weights, device=args.device, palette="random")

    images = coco.get("images", [])
    # Some DIOR processed COCO files contain repeated file_name with different image ids.
    # Teacher outputs are keyed by image stem, so repeated file_name would overwrite outputs.
    # Deduplicate by file_name explicitly to avoid wasted inference and silent overwrite.
    uniq_images: List[Dict[str, Any]] = []
    seen_files: Set[str] = set()
    dup_count = 0
    for im in images:
        fn = str(im.get("file_name", ""))
        if not fn:
            continue
        if fn in seen_files:
            dup_count += 1
            continue
        seen_files.add(fn)
        uniq_images.append(im)
    if dup_count > 0:
        print_log(
            f"[INFO] COCO images has duplicated file_name; skip_duplicates={dup_count}, "
            f"keep_unique={len(uniq_images)}",
            level="INFO",
        )
    images = uniq_images
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    # XML stubs location (sibling of outputs)
    xml_stub_dir = out_dir.parent / "xml_stub"
    xml_stub_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for im in images:
        img_id = int(im["id"])
        file_name = img_id_to_file.get(img_id, str(im.get("file_name", "")))
        if not file_name:
            continue
        img_path = img_dir / file_name
        if not img_path.is_file():
            print_log(f"[WARN] missing image: {img_path}", level="WARNING")
            continue

        anns = anns_by_img.get(img_id, [])
        cat_names: Set[str] = set()
        for ann in anns:
            cid = int(ann["category_id"])
            n = cat_id_to_name.get(cid)
            if n:
                cat_names.add(n)
        if not cat_names:
            # fallback: still provide all 20 classes if no gt (rare)
            cat_names = set(cat_id_to_name.values())

        names = sorted(cat_names)
        texts = " . ".join(names) + " ."
        label_to_class = {i: n for i, n in enumerate(names)}

        # run
        inferencer(
            inputs=str(img_path),
            texts=texts,
            custom_entities=True,
            pred_score_thr=float(args.pred_score_thr),
            batch_size=int(args.batch_size),
            out_dir=str(out_dir),
            no_save_vis=True,
            no_save_pred=False,
        )

        # DetInferencer saves preds/<image_stem>.json; rename to stem we need (DIOR stem already)
        stem = Path(file_name).stem
        pred_json = out_dir / "preds" / f"{stem}.json"
        if pred_json.is_file():
            try:
                data = json.loads(pred_json.read_text(encoding="utf-8"))
                data["class_names"] = names
                data["label_to_class"] = label_to_class
                pred_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print_log(f"[WARN] failed to inject label_to_class: {pred_json}: {e}", level="WARNING")

        # xml stub for visualize_marker; keep GT category names so teacher
        # red/green color semantics remain consistent after migrating from XML to COCO.
        write_xml_stub(xml_stub_dir / f"{stem}.xml", file_name, names)

        processed += 1

    print(f"[DONE] processed={processed} outputs={out_dir} xml_stub={xml_stub_dir}")


if __name__ == "__main__":
    main()
