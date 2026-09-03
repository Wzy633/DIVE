#!/usr/bin/env python3
"""Filter OPT-RSVG samples by supported target classes.

This utility is intended for detector-aligned subset evaluation, e.g. keeping
only the OPT-RSVG classes covered by a fixed-class detector such as LAE-DINO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


DEFAULT_SUPPORTED_CLASSES = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter OPT-RSVG samples by target class.")
    p.add_argument("--samples-jsonl", type=str, required=True)
    p.add_argument("--out-jsonl", type=str, required=True)
    p.add_argument("--out-summary-json", type=str, required=True)
    p.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_SUPPORTED_CLASSES),
        help="Comma-separated canonical OPT-RSVG target classes to keep.",
    )
    return p.parse_args()


def norm_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def main() -> None:
    args = parse_args()
    src = Path(args.samples_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    keep_classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    keep_norm: Set[str] = {norm_name(c) for c in keep_classes}

    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept: List[Dict] = []
    dropped: List[Dict] = []
    kept_by_class: Dict[str, int] = {c: 0 for c in keep_classes}
    dropped_by_class: Dict[str, int] = {}

    for row in rows:
        cls = str(row.get("target_class", "") or "")
        cls_norm = norm_name(cls)
        if cls_norm in keep_norm:
            kept.append(row)
            for canonical in keep_classes:
                if norm_name(canonical) == cls_norm:
                    kept_by_class[canonical] += 1
                    break
        else:
            dropped.append(row)
            dropped_by_class[cls] = dropped_by_class.get(cls, 0) + 1

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": str(src),
        "out_jsonl": str(out_jsonl),
        "samples_total": len(rows),
        "samples_kept": len(kept),
        "samples_dropped": len(dropped),
        "keep_classes": keep_classes,
        "kept_by_class": kept_by_class,
        "dropped_by_class": dropped_by_class,
        "images_kept": len({str(r.get("image_file", "")) for r in kept}),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Filter OPT-RSVG samples by target classes")
    print("=" * 72)
    print(f"samples_total={len(rows)}")
    print(f"samples_kept={len(kept)}")
    print(f"samples_dropped={len(dropped)}")
    print(f"images_kept={summary['images_kept']}")
    print(f"out_jsonl={out_jsonl}")
    print(f"out_summary_json={out_summary}")
    print("=" * 72)


if __name__ == "__main__":
    main()
