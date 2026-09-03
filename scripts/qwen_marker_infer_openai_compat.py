#!/usr/bin/env python3
"""用 OpenAI-兼容 API（如 oneapi）对 marker 图做推理，输出可评估的 results.txt。

适配你当前的 DIOR marker 数据（`LAE-DINO/data/DIOR_marked/...`）：
  - 输入图：已叠加红色数字 marker 的 jpg（例如 `.../images/00001.jpg`）
  - 输入问题：`text_pairs_stvg_style.jsonl`（每行一个样本，含 image_file、question、target_marker_id 等）
  - 输出：一个文本文件 results.txt，供 `evaluate_rsvg_marker(1).py` 读取并计算 IoU

输出格式（严格对齐 evaluate_rsvg_marker(1).py 的解析逻辑）：
  ----------------------------------------------------------------------
  00001.jpg
  <model response text, should contain <answer>123</answer>>
  ----------------------------------------------------------------------
  ...

注意：
  - 评估脚本默认只会从回答里提取数字 ID（优先 <answer>123</answer>），并用 marker_json_dir 映射到 bbox。
  - 因此你必须保证模型回答里出现 <answer>数字</answer>（或 answer: 数字）。
  - API 配置通过命令行参数或 OPENAI_* / DASHSCOPE_* 环境变量传入；
    可选 --extra-body 合并请求体；--usage-log 逐条记录 usage，结束时打印 token 合计。
  - --resume：断点续写已有 --out（截断不完整尾部后追加）；需与中断前相同的 --jsonl / --skip。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from text_pair_cn_ordinal import AREA_BBOX_RULE, RANK_VS_ID_RULE  # noqa: E402


SEP = "-" * 70
ANSWER_TAG_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)
ANSWER_ALT_RE = re.compile(r"answer[:\s]+(\d+)", re.IGNORECASE)
ID_HINT_RE = re.compile(r"\bID\b\s*[:：]?\s*\**\s*(\d+)\s*\**", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b(\d+)\b")


def truncate_results_to_complete_records(path: Path) -> int:
    """按 SEP 分块；丢弃末尾不完整块并重写文件；返回完整块数量。"""
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    raw = path.read_text(encoding="utf-8")
    if SEP not in raw:
        return 0
    raw = raw[raw.find(SEP) :]
    parts = raw.split(SEP)
    good: list[str] = []
    for k in range(1, len(parts)):
        block = parts[k].lstrip("\r\n")
        lines = block.split("\n")
        if len(lines) >= 2 and lines[0].strip() and lines[1].strip():
            good.append(parts[k])
        else:
            break
    n = len(good)
    if n == 0:
        return 0
    path.write_text("".join(SEP + p for p in good), encoding="utf-8")
    return n


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint configuration.
# Priority: command-line arguments > OPENAI_* / DASHSCOPE_* environment variables.
# Keep credentials outside source control.
# ---------------------------------------------------------------------------
_CONFIG_BASE_URL = ""
_CONFIG_API_KEY = ""
_CONFIG_MODEL = ""


def _default_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("DASHSCOPE_BASE_URL", "").strip()
        or _CONFIG_BASE_URL
    )


def _default_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or _CONFIG_API_KEY
    )


def _default_model() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip() or _CONFIG_MODEL


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON at {path}:{lineno}: {e}") from e


def b64_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(suffix, "image/jpeg")
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def default_system_prompt() -> str:
    return (
        "你是一个视觉定位助手。图片中每个检测框的四个角附近会出现数字标记(目标ID)，位置可能在任意角。\n"
        "同一颜色表示同一类别：检测框线条颜色与数字颜色一致，且同色框属于同一个 label_name 类别。\n"
        "若问题涉及「从左到右」「从上到下」等位置次序，一律按各目标检测框的几何质心(中心点)比较："
        "质心 x 小者在左、大者在右；质心 y 小者在上、大者在下。\n"
        f"{AREA_BBOX_RULE}\n"
        f"{RANK_VS_ID_RULE}\n"
        "题干中的英文类别名若带有中文括号(如 vehicle（车辆）)，按该类别理解目标。\n"
        "若问题中出现类别名，只在该类目标对应颜色的框上读取数字 ID；不要用其它类别目标的编号作答。\n"
        "请根据问题找到对应目标，并严格按如下格式输出：\n"
        "<answer>数字</answer>\n"
        "除非问题明确要求多个目标，否则只输出一个数字ID；不要输出坐标或其它文本。"
    )


def build_messages(question: str, image_data_url: str, system_prompt: str) -> list[dict]:
    # OpenAI-compatible multimodal: content is list of {type:text|image_url}.
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def _assistant_text_from_message(msg: Dict[str, Any]) -> str:
    """Prefer final content; fallback to reasoning_content (thinking models)."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    content = content.strip()
    if content:
        return content
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    return ""


def _extract_id_from_text(text: str) -> Optional[int]:
    """Extract a marker ID from free-form model text."""
    t = (text or "").strip()
    if not t:
        return None
    m = ANSWER_TAG_RE.search(t)
    if m:
        return int(m.group(1))
    m = ANSWER_ALT_RE.search(t)
    if m:
        return int(m.group(1))
    id_matches = ID_HINT_RE.findall(t)
    if id_matches:
        return int(id_matches[-1])
    nums = [int(x) for x in NUMBER_RE.findall(t)]
    uniq = sorted(set(nums))
    if len(uniq) == 1:
        return int(uniq[0])
    return None


def normalize_response_to_answer_tag(text: str, *, answer_only: bool = False) -> str:
    """Ensure output contains a parseable <answer>k</answer> when possible."""
    t = (text or "").strip()
    if not t:
        return "<answer>0</answer>" if answer_only else t
    k = _extract_id_from_text(t)
    if k is None:
        return "<answer>0</answer>" if answer_only else t
    if answer_only:
        return f"<answer>{k}</answer>"
    if ANSWER_TAG_RE.search(t) or ANSWER_ALT_RE.search(t):
        return t
    return t + f"\n<answer>{k}</answer>"


def post_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """返回 (response_text, error_message, usage_dict)。成功时 error_message=None。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # 即使仅含 enable_thinking=False 也需合并（dict 非空；若将来传 {} 则不调覆盖）
    normalized_base_url = (base_url or "").lower()
    supports_enable_thinking = ("dashscope" in normalized_base_url) or ("aliyuncs.com" in normalized_base_url)
    if extra_body is not None:
        for k, v in extra_body.items():
            if k == "enable_thinking" and not supports_enable_thinking:
                continue
            payload[k] = v

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                last_err = f"HTTP {resp.status_code}: {resp.text[:800]}"
            else:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    last_err = "empty choices in API response"
                    raise ValueError(last_err)
                msg = (choices[0].get("message") or {}) or {}
                content = _assistant_text_from_message(msg)
                usage = data.get("usage")
                if usage is not None and not isinstance(usage, dict):
                    usage = None
                return content, None, usage
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        # exponential-ish backoff with jitter
        sleep_s = min(3.0, 0.5 * (2 ** (attempt - 1)))
        sleep_s = sleep_s * (0.7 + 0.6 * random.random())
        time.sleep(sleep_s)

    return "", last_err, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marker-ID inference with an OpenAI-compatible multimodal API.")

    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="STVG-style question JSONL",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing marker-annotated images",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output results.txt path",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=_default_base_url(),
        help="e.g. https://dashscope.aliyuncs.com/compatible-mode/v1 (北京) or ...-intl... (新加坡)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=_default_api_key(),
    )
    p.add_argument("--model", type=str, default=_default_model())
    p.add_argument(
        "--extra-body",
        type=str,
        default="",
        help='Optional JSON object merged into POST body (DashScope extras), e.g. \'{"enable_thinking":false}\'',
    )
    p.add_argument(
        "--usage-log",
        type=str,
        default="",
        help="If set, append one JSON line per request with usage from API (for billing estimates).",
    )

    p.add_argument("--system-prompt", type=str, default=default_system_prompt())
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--answer-only-output",
        action="store_true",
        help="If ID is extractable, write output as only '<answer>ID</answer>' to keep responses short.",
    )
    p.add_argument("--timeout-s", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=6)

    p.add_argument("--limit", type=int, default=0, help="If >0, only run first N samples.")
    p.add_argument("--skip", type=int, default=0, help="Skip first N samples.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="断点续写：从已有 --out 中已完成的样本之后继续追加；不完整尾部会先截断。"
        " 需与中断前使用相同的 --jsonl / --skip（及顺序一致）。",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not call API; write placeholder answers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    images_dir = Path(args.images_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.is_file():
        raise SystemExit(f"jsonl not found: {jsonl_path}")
    if not images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {images_dir}")

    extra_body: Dict[str, Any]
    if args.extra_body and str(args.extra_body).strip():
        try:
            extra_body = json.loads(args.extra_body)
            if not isinstance(extra_body, dict):
                raise ValueError("extra-body must be a JSON object")
        except Exception as e:
            raise SystemExit(f"Invalid --extra-body JSON: {e}") from e
    else:
        extra_body = {}
    # 百炼 Qwen3 系列在兼容接口里可能开启「思考链」，会产生数千 reasoning token，极慢且易误判为卡死。
    # 默认关闭；若需思考链可传 --extra-body '{"enable_thinking":true}'。
    extra_body.setdefault("enable_thinking", False)
    if not args.dry_run:
        print(
            f"[config] enable_thinking={extra_body.get('enable_thinking')} "
            "(关闭可提速省 token；若要开启思考链请使用 --extra-body 传入 enable_thinking: true)",
            flush=True,
        )

    usage_log_path = Path(args.usage_log) if str(args.usage_log).strip() else None
    if usage_log_path:
        usage_log_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        if not args.base_url:
            raise SystemExit(
                "Missing base URL: pass --base-url or set OPENAI_BASE_URL / DASHSCOPE_BASE_URL."
            )
        if not args.api_key:
            raise SystemExit(
                "Missing API key: pass --api-key or set OPENAI_API_KEY / DASHSCOPE_API_KEY."
            )
        if not args.model:
            raise SystemExit("Missing model: pass --model or set OPENAI_MODEL.")

    total = 0
    written = 0
    sum_prompt = 0
    sum_completion = 0
    n_usage = 0

    resume_done = 0
    if args.resume:
        resume_done = truncate_results_to_complete_records(out_path)
        if resume_done == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            raise SystemExit(
                "[resume] 无法在 --out 中解析出完整样本块；请删除或修复该文件后再试，"
                "或去掉 --resume 从头写入。"
            )
        if resume_done > 0:
            print(f"[resume] 已有完整样本 {resume_done} 条，从不完整处截断后追加", flush=True)

    pending_skip = resume_done
    out_mode = "a" if (args.resume and resume_done > 0) else "w"
    # buffering=1：行缓冲，配合每样本后 flush，便于 nohup 时 results 文件立刻可见、非 0 字节。
    with out_path.open(out_mode, encoding="utf-8", buffering=1) as w:
        usage_fp = (
            usage_log_path.open("a", encoding="utf-8") if usage_log_path is not None else None
        )
        try:
            for sample in read_jsonl(jsonl_path):
                total += 1
                if args.skip and total <= args.skip:
                    continue
                if pending_skip > 0:
                    pending_skip -= 1
                    continue
                if args.limit and written >= args.limit:
                    break

                image_file = str(sample.get("image_file", "")).strip()
                question = str(sample.get("question", "")).strip()
                if not image_file or not question:
                    # 必须与 jsonl 行数一一对应，否则评估脚本按行对齐会错位
                    w.write(SEP + "\n")
                    w.write((image_file or "(missing_image_file)") + "\n")
                    w.write("<answer>0</answer>\n")
                    w.write("[WARN] empty image_file or question in jsonl\n")
                    written += 1
                    w.flush()
                    if not args.dry_run:
                        print(
                            f"[progress] {written} {image_file or '(empty)'} (skipped: bad row)",
                            flush=True,
                        )
                    continue

                img_path = images_dir / image_file
                if not img_path.is_file():
                    # 仍然写入一个失败样本，便于后续统计
                    w.write(SEP + "\n")
                    w.write(image_file + "\n")
                    w.write("<answer>0</answer>\n")
                    written += 1
                    w.flush()
                    continue

                if args.dry_run:
                    # placeholder: always output 0
                    resp_text = "<answer>0</answer>"
                    err = None
                    usage = None
                else:
                    data_url = b64_data_url(img_path)
                    messages = build_messages(question, data_url, args.system_prompt)
                    resp_text, err, usage = post_chat_completions(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        messages=messages,
                        temperature=float(args.temperature),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.timeout_s),
                        max_retries=int(args.max_retries),
                        extra_body=extra_body,
                    )
                    if err is not None:
                        # 保底：写一个可解析的 answer，避免整条链路断掉
                        resp_text = resp_text.strip() if resp_text else ""
                        resp_text = resp_text + ("\n" if resp_text else "") + "<answer>0</answer>"

                resp_text = normalize_response_to_answer_tag(
                    resp_text,
                    answer_only=bool(args.answer_only_output),
                )
                w.write(SEP + "\n")
                w.write(image_file + "\n")
                w.write(resp_text.strip() + "\n")
                if err:
                    w.write(f"[WARN] {err}\n")
                written += 1
                w.flush()
                if not args.dry_run:
                    print(f"[progress] {written} {image_file}", flush=True)

                if usage_fp is not None:
                    rec = {
                        "image_file": image_file,
                        "usage": usage,
                        "error": err,
                    }
                    usage_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    usage_fp.flush()
                if usage and isinstance(usage, dict):
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    if isinstance(pt, int):
                        sum_prompt += pt
                    if isinstance(ct, int):
                        sum_completion += ct
                    if isinstance(pt, int) or isinstance(ct, int):
                        n_usage += 1
        finally:
            if usage_fp is not None:
                usage_fp.close()

    print(f"[DONE] samples_written={written} out={out_path}")
    if n_usage > 0:
        print(
            f"[INFO] usage_sum prompt_tokens={sum_prompt} completion_tokens={sum_completion} "
            f"from {n_usage} responses (see --usage-log for per-request details)"
        )


if __name__ == "__main__":
    main()
