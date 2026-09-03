#!/usr/bin/env python3
"""从 marker_meta_lae_stvg.json 导出「逐图片」marker JSON 文件。

为什么需要它：
  - `evaluate_rsvg_marker(1).py` 期望 marker-json-dir 下存在 `<image_stem>.json`，
    且结构形如：
      {"image_name":"00001","markers":[{"marker_id":1,"bbox":[x1,y1,x2,y2]}, ...]}
  - 你的 DIOR_marked 当前只有一个聚合文件 `marker_meta_lae_stvg.json`（按 file_name 存 instances）

本脚本把聚合 meta 拆成逐图 JSON，便于后续复用老师的评估脚本（或其它工具）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export per-image marker JSON from marker_meta_lae_stvg.json")
    p.add_argument(
        "--marker-meta",
        type=str,
        required=True,
        help="Path to marker_meta_lae_stvg.json",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for per-image JSON files",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, only export first N images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = Path(args.marker_meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.is_file():
        raise SystemExit(f"marker-meta not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("marker-meta must be a JSON list of image records")

    count = 0
    for rec in data:
        if args.limit and count >= args.limit:
            break
        if not isinstance(rec, dict):
            continue
        file_name = str(rec.get("file_name", "")).strip()
        if not file_name:
            continue
        stem = Path(file_name).stem
        instances = rec.get("instances", [])
        markers: List[Dict[str, Any]] = []
        if isinstance(instances, list):
            for ins in instances:
                if not isinstance(ins, dict):
                    continue
                mid = ins.get("marker_id", None)
                bbox = ins.get("bbox_xyxy", None)
                if mid is None or bbox is None:
                    continue
                markers.append(
                    {
                        "marker_id": int(mid),
                        "bbox": [float(x) for x in bbox],
                    }
                )

        out_path = out_dir / f"{stem}.json"
        with out_path.open("w", encoding="utf-8") as w:
            json.dump(
                {
                    "image_name": stem,
                    "file_name": file_name,
                    "markers": markers,
                },
                w,
                ensure_ascii=False,
                indent=2,
            )
        count += 1

    print(f"[DONE] exported={count} -> {out_dir}")


if __name__ == "__main__":
    main()
