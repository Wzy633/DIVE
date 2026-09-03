#!/usr/bin/env bash
# DIOR-RSVG teacher marker：Qwen 推理 + 评测（XML description 对齐的 text pairs）
#
# 用法：
#   MODEL_NAME="qwen3.5-plus" bash LAE-DINO/scripts/run_rsvg_teacher_qwen_infer_and_eval.sh test
#   MODEL_NAME="qwen3-vl-plus" bash LAE-DINO/scripts/run_rsvg_teacher_qwen_infer_and_eval.sh test
#
# 可选环境变量（覆盖默认）：
#   PROJECT_ROOT   默认：本脚本所在仓库的 LAE-DINO 上级目录（含 DIOR_RSVG 与 LAE-DINO）
#   EPOCH_DIR      默认：teacher/epoch_26
#   MODEL_NAME     默认：qwen3.5-plus（输出 results_<slug>.txt / usage_<slug>.jsonl）
#   TEMPERATURE    默认：0
#   EXTRA_PY_ARGS  传给 qwen 的额外参数，例如：--limit 100 --dry-run
#   RESUME_INFER=1 断点续写（等价于追加 --resume，从中断处继续写 results）
#   SKIP_INFER=1   跳过推理，仅对已存在的结果文件做评测
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SPLIT="${1:-test}"

case "${SPLIT}" in
  train|val|test) ;;
  *)
    echo "用法: $0 {train|val|test}" >&2
    exit 1
    ;;
esac

EPOCH_DIR="${EPOCH_DIR:-teacher/epoch_26}"
BASE="${PROJECT_ROOT}/LAE-DINO/data/DIOR_RSVG_marked/${EPOCH_DIR}/${SPLIT}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-plus}"
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="model"
fi

JSONL="${BASE}/text_pairs_rsvg_teacher.fixed.jsonl"
IMAGES="${BASE}/vis_pred_marker"
RESULTS="${BASE}/results_${MODEL_SLUG}.txt"
USAGE="${BASE}/usage_${MODEL_SLUG}.jsonl"

LEGACY_RESULTS="${BASE}/results_qwen_marker.txt"
LEGACY_USAGE="${BASE}/qwen_usage.jsonl"
if [[ "${SKIP_INFER:-0}" == "1" && ! -f "${RESULTS}" && -f "${LEGACY_RESULTS}" ]]; then
  RESULTS="${LEGACY_RESULTS}"
fi
if [[ "${SKIP_INFER:-0}" == "1" && ! -f "${USAGE}" && -f "${LEGACY_USAGE}" ]]; then
  USAGE="${LEGACY_USAGE}"
fi

if [[ ! -f "${JSONL}" ]]; then
  echo "[ERROR] 未找到 jsonl: ${JSONL}" >&2
  echo "请先运行 build_text_pairs_from_teacher_rsvg_xml.py 生成 text pairs。" >&2
  exit 1
fi
if [[ ! -d "${IMAGES}" ]]; then
  echo "[ERROR] 未找到 marker 图目录: ${IMAGES}" >&2
  exit 1
fi

PY_EXTRA="${EXTRA_PY_ARGS:-}"
if [[ "${RESUME_INFER:-0}" == "1" ]]; then
  PY_EXTRA="${PY_EXTRA} --resume"
fi

# shellcheck disable=SC2086
if [[ "${SKIP_INFER:-0}" != "1" ]]; then
  echo "[1/2] Qwen 推理 split=${SPLIT} model=${MODEL_NAME}"
  python "${SCRIPT_DIR}/qwen_marker_infer_openai_compat.py" \
    --jsonl "${JSONL}" \
    --images-dir "${IMAGES}" \
    --out "${RESULTS}" \
    --usage-log "${USAGE}" \
    --model "${MODEL_NAME}" \
    --temperature "${TEMPERATURE:-0}" \
    --max-tokens "${MAX_TOKENS:-512}" \
    --answer-only-output \
    ${PY_EXTRA}
else
  echo "[1/2] 跳过推理 (SKIP_INFER=1, model=${MODEL_NAME})"
  if [[ "${RESULTS}" == "${LEGACY_RESULTS}" ]]; then
    echo "[INFO] 使用旧结果文件: ${RESULTS}"
  fi
fi

if [[ ! -f "${RESULTS}" ]]; then
  echo "[ERROR] 未找到推理结果: ${RESULTS}" >&2
  exit 1
fi

echo "[2/2] 评测 (RSVG teacher: pred bbox vs XML GT)"
python "${SCRIPT_DIR}/evaluate_rsvg_qwen_teacher_by_text_pairs.py" \
  --pred-file "${RESULTS}" \
  --text-pairs-jsonl "${JSONL}" \
  --marker-json-dir "${IMAGES}"

echo "[DONE] split=${SPLIT} model=${MODEL_NAME} 结果: ${RESULTS}"
