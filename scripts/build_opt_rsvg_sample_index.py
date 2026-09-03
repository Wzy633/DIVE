#!/usr/bin/env python3
"""Build a validated sample-level index for OPT-RSVG.

OPT-RSVG stores one XML per image, but split files are sample-level:
each line corresponds to one object/description instance rather than one image.

This script enumerates XML files in sorted filename order, then enumerates
objects inside each XML in document order to build a global sample index.
It validates the split files against that index and exports a JSONL manifest
that downstream marker / text-pair builders can use safely.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build validated OPT-RSVG sample index.")
    p.add_argument("--opt-root", type=str, required=True, help="OPT-RSVG root containing Annotations/, Image/, train.txt, val.txt, test.txt")
    p.add_argument("--out-jsonl", type=str, required=True, help="Output JSONL path for sample-level manifest")
    return p.parse_args()


def xyxy_from_box(box_elem: ET.Element) -> Tuple[float, float, float, float]:
    def _get(tag: str) -> float:
        node = box_elem.find(tag)
        if node is None or node.text is None:
            raise ValueError(f"missing <{tag}>")
        return float(node.text.strip())

    return (_get("xmin"), _get("ymin"), _get("xmax"), _get("ymax"))


def load_split_indices(path: Path) -> List[int]:
    vals: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def infer_split(sample_index: int, split_sets: Dict[str, set[int]]) -> Optional[str]:
    found = [name for name, idxs in split_sets.items() if sample_index in idxs]
    if len(found) == 1:
        return found[0]
    return None


def iter_xml_paths(xml_dir: Path) -> Iterable[Path]:
    return sorted(xml_dir.glob("*.xml"))


def main() -> None:
    args = parse_args()
    opt_root = Path(args.opt_root)
    xml_dir = opt_root / "Annotations"
    img_dir = opt_root / "Image"
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not xml_dir.is_dir():
        raise SystemExit(f"missing xml dir: {xml_dir}")
    if not img_dir.is_dir():
        raise SystemExit(f"missing image dir: {img_dir}")

    split_lists = {
        "train": load_split_indices(opt_root / "train.txt"),
        "val": load_split_indices(opt_root / "val.txt"),
        "test": load_split_indices(opt_root / "test.txt"),
    }
    split_sets = {k: set(v) for k, v in split_lists.items()}

    overlaps = {}
    keys = list(split_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlaps[f"{a}_{b}"] = len(split_sets[a].intersection(split_sets[b]))

    manifest: List[Dict] = []
    class_counts: Dict[str, int] = {}
    split_image_sets: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    sample_index = 0

    for xml_path in iter_xml_paths(xml_dir):
        root = ET.parse(xml_path).getroot()
        filename_elem = root.find("filename")
        if filename_elem is None or not filename_elem.text:
            raise SystemExit(f"missing filename in {xml_path}")
        image_file = filename_elem.text.strip()
        image_stem = Path(image_file).stem
        image_path = img_dir / image_file

        size_elem = root.find("size")
        width = height = depth = None
        if size_elem is not None:
            w = size_elem.find("width")
            h = size_elem.find("height")
            d = size_elem.find("depth")
            width = int(w.text.strip()) if w is not None and w.text else None
            height = int(h.text.strip()) if h is not None and h.text else None
            depth = int(d.text.strip()) if d is not None and d.text else None

        objects = root.findall("object")
        for obj_idx, obj in enumerate(objects):
            name_elem = obj.find("name")
            box_elem = obj.find("bndbox")
            desc_elem = obj.find("description")
            if name_elem is None or not name_elem.text or box_elem is None:
                continue

            cls = name_elem.text.strip()
            bbox = [float(x) for x in xyxy_from_box(box_elem)]
            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
            split = infer_split(sample_index, split_sets)
            if split is None:
                raise SystemExit(f"sample_index {sample_index} is missing from split files or appears in multiple splits")

            rec = {
                "sample_index": sample_index,
                "split": split,
                "image_file": image_file,
                "image_stem": image_stem,
                "image_path": str(image_path),
                "xml_file": xml_path.name,
                "object_index_in_xml": obj_idx,
                "target_class": cls,
                "gt_bbox_xyxy": bbox,
                "description": desc,
                "image_width": width,
                "image_height": height,
                "image_depth": depth,
            }
            manifest.append(rec)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            split_image_sets[split].add(image_file)
            sample_index += 1

    union = set().union(*split_sets.values())
    expected = set(range(sample_index))
    missing_from_union = sorted(expected - union)
    extra_in_union = sorted(union - expected)

    if missing_from_union:
        raise SystemExit(f"split files miss {len(missing_from_union)} sample indices; first few: {missing_from_union[:10]}")
    if extra_in_union:
        raise SystemExit(f"split files contain {len(extra_in_union)} out-of-range sample indices; first few: {extra_in_union[:10]}")
    if any(v != 0 for v in overlaps.values()):
        raise SystemExit(f"split overlap detected: {overlaps}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    split_counts = {k: len(v) for k, v in split_lists.items()}
    split_image_counts = {k: len(v) for k, v in split_image_sets.items()}

    print("=" * 72)
    print("OPT-RSVG sample index built successfully")
    print("=" * 72)
    print(f"xml_files={len(list(iter_xml_paths(xml_dir)))}")
    print(f"sample_total={len(manifest)}")
    print(f"images_total={len({m['image_file'] for m in manifest})}")
    for split in ("train", "val", "test"):
        print(f"{split}_samples={split_counts[split]}  {split}_images={split_image_counts[split]}")
    print(f"classes={len(class_counts)}")
    print("class_counts=" + json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False))
    print(f"out_jsonl={out_jsonl}")
    print("=" * 72)


if __name__ == "__main__":
    main()
