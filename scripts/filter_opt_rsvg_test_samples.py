#!/usr/bin/env python3
"""Filter OPT-RSVG sample manifest into valid / invalid test subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter OPT-RSVG test samples into valid and invalid subsets.")
    p.add_argument("--samples-jsonl", type=str, required=True)
    p.add_argument("--out-valid-jsonl", type=str, required=True)
    p.add_argument("--out-invalid-jsonl", type=str, required=True)
    p.add_argument("--out-summary-md", type=str, required=True)
    return p.parse_args()


def is_valid_bbox(b: List[float]) -> bool:
    return isinstance(b, list) and len(b) == 4 and b[0] < b[2] and b[1] < b[3]


def main() -> None:
    args = parse_args()
    src = Path(args.samples_jsonl)
    out_valid = Path(args.out_valid_jsonl)
    out_invalid = Path(args.out_invalid_jsonl)
    out_summary = Path(args.out_summary_md)
    out_valid.parent.mkdir(parents=True, exist_ok=True)
    out_invalid.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_recs = [r for r in recs if r.get("split") == "test"]
    valid = [r for r in test_recs if is_valid_bbox(r.get("gt_bbox_xyxy", []))]
    invalid = [r for r in test_recs if not is_valid_bbox(r.get("gt_bbox_xyxy", []))]

    with out_valid.open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_invalid.open("w", encoding="utf-8") as f:
        for r in invalid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    unique_test_images = len({r["image_file"] for r in test_recs})
    unique_valid_images = len({r["image_file"] for r in valid})
    unique_invalid_images = len({r["image_file"] for r in invalid})

    lines = [
        "# OPT-RSVG Test Sample Filter",
        "",
        "- `test_samples_total = %d`" % len(test_recs),
        "- `test_samples_valid = %d`" % len(valid),
        "- `test_samples_invalid_bbox = %d`" % len(invalid),
        "- `test_images_total = %d`" % unique_test_images,
        "- `test_images_with_valid_samples = %d`" % unique_valid_images,
        "- `test_images_with_invalid_samples = %d`" % unique_invalid_images,
        "",
        "## Representative Invalid Samples",
        "",
    ]
    for r in invalid[:10]:
        lines.append(
            "- `sample_index = %d`, `image_file = %s`, `xml_file = %s`, `object_index_in_xml = %d`, `class = %s`, `gt_bbox_xyxy = %s`"
            % (
                int(r["sample_index"]),
                r["image_file"],
                r["xml_file"],
                int(r["object_index_in_xml"]),
                r["target_class"],
                r["gt_bbox_xyxy"],
            )
        )
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print("OPT-RSVG test samples filtered")
    print("=" * 72)
    print(f"test_samples_total={len(test_recs)}")
    print(f"test_samples_valid={len(valid)}")
    print(f"test_samples_invalid_bbox={len(invalid)}")
    print(f"test_images_total={unique_test_images}")
    print(f"test_images_with_valid_samples={unique_valid_images}")
    print(f"test_images_with_invalid_samples={unique_invalid_images}")
    print(f"out_valid_jsonl={out_valid}")
    print(f"out_invalid_jsonl={out_invalid}")
    print(f"out_summary_md={out_summary}")
    print("=" * 72)


if __name__ == "__main__":
    main()
