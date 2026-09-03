#!/usr/bin/env bash
# OPT-RSVG (LAE-DINO-covered 10 classes) teacher-style full pipeline:
#   1) filter valid OPT-RSVG samples to the 10 classes covered by LAE-DINO
#   2) run teacher-style marker generation with a fixed-class detector
#   3) build teacher text pairs
#   4) run Qwen marker inference
#   5) evaluate final grounding metrics
#
# Example:
#   CONDA_ENV=LAE CUDA_VISIBLE_DEVICES=7 \
#   OPENAI_API_KEY=xxx OPENAI_BASE_URL=xxx \
#   ./run_opt_laedino_teacher_10cls_qwen_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="${ROOT}/LAE-DINO"
MMDET="${PROJ}/mmdetection_lae"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"

OPT_ROOT="${OPT_ROOT:-${PROJ}/data/OPT-RSVG}"
SAMPLES_JSONL="${SAMPLES_JSONL:-${OPT_ROOT}/opt_rsvg_test_samples.valid.jsonl}"
SUBSET_JSONL="${SUBSET_JSONL:-${OPT_ROOT}/opt_rsvg_test_samples.valid.lae10.jsonl}"
SUBSET_SUMMARY_JSON="${SUBSET_SUMMARY_JSON:-${OPT_ROOT}/opt_rsvg_test_samples.valid.lae10.summary.json}"

CFG="${CFG:-${MMDET}/configs/lae_dino/lae_dino_swin-t_finetune_DIOR_infer.py}"
CKPT="${CKPT:-${PROJ}/work_dirs/lae_dino_swin-t_finetune_DIOR/epoch_26.pth}"

OUT_ROOT="${OUT_ROOT:-${PROJ}/data/OPT_RSVG_marked/laedino_teacher_10cls/test}"
TEXT_PAIRS_JSONL="${TEXT_PAIRS_JSONL:-${OUT_ROOT}/text_pairs_opt_rsvg_teacher.fixed.jsonl}"
RESULTS_TXT="${RESULTS_TXT:-${OUT_ROOT}/results_qwen3_5_plus.txt}"
USAGE_JSONL="${USAGE_JSONL:-${OUT_ROOT}/usage_qwen3_5_plus.jsonl}"
EVAL_TXT="${EVAL_TXT:-${OUT_ROOT}/eval_qwen3_5_plus.txt}"
KEPT_SAMPLES_JSONL="${KEPT_SAMPLES_JSONL:-${OUT_ROOT}/samples_kept_after_teacher_filter.jsonl}"

QWEN_PYTHON="${QWEN_PYTHON:-python}"
QWEN_MODEL="${QWEN_MODEL:-qwen3.5-plus}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-512}"

[[ -f "${CFG}" ]] || { echo "Missing config: ${CFG}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -f "${SAMPLES_JSONL}" ]] || { echo "Missing samples jsonl: ${SAMPLES_JSONL}" >&2; exit 1; }

if [[ -n "${CONDA_SH}" ]]; then
  [[ -f "${CONDA_SH}" ]] || { echo "Missing conda.sh: ${CONDA_SH}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  [[ -n "${CONDA_ENV}" ]] && conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${MMDET}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

echo "======================================================================"
echo "Step 1/5: Filter OPT-RSVG valid samples to LAE-DINO-covered 10 classes"
echo "======================================================================"
python "${PROJ}/scripts/filter_opt_rsvg_samples_by_target_classes.py" \
  --samples-jsonl "${SAMPLES_JSONL}" \
  --out-jsonl "${SUBSET_JSONL}" \
  --out-summary-json "${SUBSET_SUMMARY_JSON}"

echo "======================================================================"
echo "Step 2/5: Run LAE-DINO teacher-style marker generation"
echo "======================================================================"
cd "${MMDET}"
python "${PROJ}/scripts/opt_rsvg_marker_pipeline_teacher_fixed.py" \
  --opt-root "${OPT_ROOT}" \
  --samples-jsonl "${SUBSET_JSONL}" \
  --config "${CFG}" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_ROOT}" \
  --device "${DEVICE}" \
  --save-images \
  --resume

echo "======================================================================"
echo "Step 3/5: Build teacher text pairs"
echo "======================================================================"
python "${PROJ}/scripts/build_text_pairs_from_teacher_opt_rsvg_samples.py" \
  --samples-jsonl "${KEPT_SAMPLES_JSONL}" \
  --teacher-marker-dir "${OUT_ROOT}/vis_pred_marker" \
  --out-jsonl "${TEXT_PAIRS_JSONL}" \
  --min-match-iou 0.5

echo "======================================================================"
echo "Step 4/5: Run Qwen marker inference"
echo "======================================================================"
cd "${ROOT}"
"${QWEN_PYTHON}" "${PROJ}/scripts/qwen_marker_infer_openai_compat.py" \
  --jsonl "${TEXT_PAIRS_JSONL}" \
  --images-dir "${OUT_ROOT}/vis_pred_marker" \
  --out "${RESULTS_TXT}" \
  --usage-log "${USAGE_JSONL}" \
  --model "${QWEN_MODEL}" \
  --temperature "${QWEN_TEMPERATURE}" \
  --max-tokens "${QWEN_MAX_TOKENS}" \
  --answer-only-output

echo "======================================================================"
echo "Step 5/5: Evaluate final grounding metrics"
echo "======================================================================"
python "${PROJ}/scripts/evaluate_opt_qwen_teacher_by_text_pairs.py" \
  --pred-file "${RESULTS_TXT}" \
  --text-pairs-jsonl "${TEXT_PAIRS_JSONL}" \
  --marker-json-dir "${OUT_ROOT}/vis_pred_marker" | tee "${EVAL_TXT}"

echo "======================================================================"
echo "[DONE] OPT-RSVG LAE-DINO teacher 10cls pipeline"
echo "======================================================================"
echo "subset_jsonl=${SUBSET_JSONL}"
echo "kept_samples_jsonl=${KEPT_SAMPLES_JSONL}"
echo "marker_out=${OUT_ROOT}"
echo "text_pairs=${TEXT_PAIRS_JSONL}"
echo "results=${RESULTS_TXT}"
echo "eval=${EVAL_TXT}"
echo "======================================================================"
