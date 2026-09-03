#!/usr/bin/env python3
"""Create DIOR_RSVG XML subset directories from split txt lists.

DIOR_RSVG provides:
  - Annotations/<id>.xml
  - JPEGImages/<id>.jpg
  - train.txt / val.txt / test.txt where each line is an integer id (no extension)

This script copies the corresponding XML files into an output directory
so the teacher detector script can be run per split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-split XML dirs for DIOR_RSVG.")
    p.add_argument("--dior-rsvg-root", type=str, default="DIOR_RSVG", help="Root containing Annotations/ and split txts")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N ids")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dior_rsvg_root)
    ann_dir = root / "Annotations"
    split_txt = root / f"{args.split}.txt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_dir.is_dir():
        raise SystemExit(f"Missing Annotations dir: {ann_dir}")
    if not split_txt.is_file():
        raise SystemExit(f"Missing split file: {split_txt}")

    ids = []
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.append(int(s))
        except Exception:
            raise SystemExit(f"Bad id in {split_txt}: {s!r}")

    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    copied = 0
    missing = 0
    for i in ids:
        name = f"{i:05d}.xml"
        src = ann_dir / name
        dst = out_dir / name
        if not src.is_file():
            missing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"[DONE] split={args.split} copied_xml={copied} missing_xml={missing} out={out_dir}")


if __name__ == "__main__":
    main()
