#!/usr/bin/env python3
"""Merge retry results back into the original sequential result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


SEP = "-" * 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-results", required=True)
    parser.add_argument("--retry-results", required=True)
    parser.add_argument("--retry-manifest", required=True)
    parser.add_argument("--out-results", required=True)
    return parser.parse_args()


def parse_results(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [x.strip() for x in text.split(SEP) if x.strip()]


def parse_block_image(block: str) -> str:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def parse_block_body(block: str) -> str:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return "\n".join(lines[1:]).strip() if len(lines) > 1 else ""


def load_manifest(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    original_blocks = parse_results(Path(args.original_results))
    retry_blocks = parse_results(Path(args.retry_results))
    manifest = load_manifest(Path(args.retry_manifest))

    if len(retry_blocks) != len(manifest):
        raise RuntimeError(
            f"Retry results count ({len(retry_blocks)}) does not match manifest rows ({len(manifest)}). "
            "Do not merge a partially completed retry file."
        )

    merged = list(original_blocks)
    applied = 0
    image_mismatch_examples: List[Tuple[int, str, str]] = []
    warn_examples: List[Tuple[int, str]] = []

    for meta, retry_block in zip(manifest, retry_blocks):
        original_index = int(meta["original_index"])
        expected_image = str(meta["image_file"])
        retry_image = parse_block_image(retry_block)
        retry_body = parse_block_body(retry_block)
        if retry_image != expected_image and len(image_mismatch_examples) < 10:
            image_mismatch_examples.append((original_index, expected_image, retry_image))
        if "[WARN]" in retry_body and len(warn_examples) < 10:
            warn_examples.append((original_index, retry_image))
        while len(merged) < original_index:
            merged.append("")
        merged[original_index - 1] = retry_block
        applied += 1

    if image_mismatch_examples:
        raise RuntimeError(
            "Retry result image order does not match manifest. First mismatches:\n"
            + json.dumps(image_mismatch_examples, ensure_ascii=False, indent=2)
        )

    if warn_examples:
        raise RuntimeError(
            "Retry results still contain warning/error blocks. Clean or rerun retry before merge. "
            "First examples:\n"
            + json.dumps(warn_examples, ensure_ascii=False, indent=2)
        )

    if any(block == "" for block in merged):
        raise RuntimeError(
            "Merged results still contain empty placeholders. This indicates missing retry outputs "
            "or a broken manifest alignment."
        )

    out = Path(args.out_results)
    text = "\n".join([SEP + "\n" + block for block in merged]) + ("\n" if merged else "")
    out.write_text(text, encoding="utf-8")

    summary = {
        "original_blocks": len(original_blocks),
        "retry_blocks": len(retry_blocks),
        "manifest_rows": len(manifest),
        "applied_retry_blocks": applied,
        "merged_blocks": len(merged),
        "image_mismatch_examples": image_mismatch_examples,
        "out_results": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
