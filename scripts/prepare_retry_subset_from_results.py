#!/usr/bin/env python3
"""Prepare a retry subset from a sequential result file and the full jsonl list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


SEP = "-" * 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-pairs-jsonl", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-summary-json", required=True)
    parser.add_argument("--warn-token", default="[WARN]")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_results(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = [x.strip() for x in text.split(SEP) if x.strip()]
    rows: List[Tuple[str, str]] = []
    for part in parts:
        lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
        if not lines:
            continue
        image_file = lines[0]
        body = "\n".join(lines[1:]).strip()
        rows.append((image_file, body))
    return rows


def main() -> None:
    args = parse_args()
    text_pairs = load_jsonl(Path(args.text_pairs_jsonl))
    results = parse_results(Path(args.results_file))

    mismatch_examples: List[Dict[str, Any]] = []
    for idx, ((img_res, _body), rec) in enumerate(zip(results, text_pairs), start=1):
        img_exp = str(rec.get("image_file", "")).strip()
        if img_res != img_exp:
            mismatch_examples.append(
                {
                    "original_index": idx,
                    "expected_image_file": img_exp,
                    "result_image_file": img_res,
                }
            )
            if len(mismatch_examples) >= 10:
                break

    if mismatch_examples:
        raise RuntimeError(
            "Results are not aligned with text_pairs. First mismatches:\n"
            + json.dumps(mismatch_examples, ensure_ascii=False, indent=2)
        )

    retry_records: List[Dict[str, Any]] = []
    retry_manifest: List[Dict[str, Any]] = []
    warn_count = 0
    missing_tail_count = 0

    for idx, rec in enumerate(text_pairs, start=1):
        if idx <= len(results):
            image_file, body = results[idx - 1]
            if args.warn_token in body:
                warn_count += 1
                retry_records.append(rec)
                retry_manifest.append(
                    {
                        "retry_index": len(retry_manifest) + 1,
                        "original_index": idx,
                        "image_file": image_file,
                        "reason": "warn",
                    }
                )
        else:
            missing_tail_count += 1
            retry_records.append(rec)
            retry_manifest.append(
                {
                    "retry_index": len(retry_manifest) + 1,
                    "original_index": idx,
                    "image_file": str(rec.get("image_file", "")).strip(),
                    "reason": "missing_tail",
                }
            )

    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    out_summary = Path(args.out_summary_json)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in retry_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out_manifest.open("w", encoding="utf-8") as f:
        for rec in retry_manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "text_pairs_total": len(text_pairs),
        "result_blocks_total": len(results),
        "warn_retry_count": warn_count,
        "missing_tail_count": missing_tail_count,
        "retry_total": len(retry_manifest),
        "out_jsonl": str(out_jsonl),
        "out_manifest": str(out_manifest),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
