#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LAE_DIR="${LAE_DIR:-${ROOT_DIR}/LAE-DINO}"
MTP_ROOT="${MTP_ROOT:-${ROOT_DIR}/third_party/MTP}"
OPT_ROOT="${OPT_ROOT:-${LAE_DIR}/data/OPT-RSVG}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_PYTHON="${QWEN_PYTHON:-${PYTHON_BIN}}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-${LAE_DIR}/data/OPT_RSVG_marked/mtp_teacher_10cls/test}"

FILTERED_JSONL="${FILTERED_JSONL:-${OPT_ROOT}/opt_rsvg_test_samples.valid.mtp10.jsonl}"
FILTERED_SUMMARY="${FILTERED_SUMMARY:-${OPT_ROOT}/opt_rsvg_test_samples.valid.mtp10.summary.json}"
MARKER_CONFIG="${MARKER_CONFIG:-${MTP_ROOT}/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py}"
MARKER_CKPT="${MARKER_CKPT:-${MTP_ROOT}/weights/dior-rvsa-l-mae-mtp-epoch_12.pth}"

KEPT_JSONL="${OUT_ROOT}/samples_kept_after_teacher_filter.jsonl"
TEXT_PAIRS_JSONL="${OUT_ROOT}/text_pairs_opt_rsvg_teacher.fixed.jsonl"
QWEN_OUT="${OUT_ROOT}/results_qwen3_5_plus.txt"
QWEN_USAGE="${OUT_ROOT}/usage_qwen3_5_plus.jsonl"
EVAL_TXT="${OUT_ROOT}/eval_qwen3_5_plus.txt"
QWEN_MODEL="${QWEN_MODEL:-qwen3.5-plus}"

mkdir -p "${OUT_ROOT}"

echo "======================================================================"
echo "Step 1/5: Filter OPT-RSVG valid samples to MTP-covered 10 classes"
echo "======================================================================"
"${PYTHON_BIN}" "${LAE_DIR}/scripts/filter_opt_rsvg_samples_by_target_classes.py" \
  --samples-jsonl "${OPT_ROOT}/opt_rsvg_test_samples.valid.jsonl" \
  --out-jsonl "${FILTERED_JSONL}" \
  --out-summary-json "${FILTERED_SUMMARY}"

echo "======================================================================"
echo "Step 2/5: Run MTP teacher-style marker generation"
echo "======================================================================"
"${PYTHON_BIN}" "${LAE_DIR}/scripts/opt_rsvg_marker_pipeline_teacher_mtp_fixed.py" \
  --opt-root "${OPT_ROOT}" \
  --samples-jsonl "${FILTERED_JSONL}" \
  --config "${MARKER_CONFIG}" \
  --checkpoint "${MARKER_CKPT}" \
  --out-dir "${OUT_ROOT}" \
  --device "${DEVICE}" \
  --save-images \
  --resume

echo "======================================================================"
echo "Step 3/5: Build teacher text pairs from kept OPT-RSVG samples"
echo "======================================================================"
"${PYTHON_BIN}" "${LAE_DIR}/scripts/build_text_pairs_from_teacher_opt_rsvg_samples.py" \
  --samples-jsonl "${KEPT_JSONL}" \
  --teacher-marker-dir "${OUT_ROOT}/vis_pred_marker" \
  --out-jsonl "${TEXT_PAIRS_JSONL}" \
  --min-match-iou 0.5

echo "======================================================================"
echo "Step 4/5: Run VLM marker inference (${QWEN_MODEL})"
echo "======================================================================"
"${QWEN_PYTHON}" "${LAE_DIR}/scripts/qwen_marker_infer_openai_compat.py" \
  --jsonl "${TEXT_PAIRS_JSONL}" \
  --images-dir "${OUT_ROOT}/vis_pred_marker" \
  --out "${QWEN_OUT}" \
  --usage-log "${QWEN_USAGE}" \
  --model "${QWEN_MODEL}" \
  --temperature 0 \
  --max-tokens 512 \
  --answer-only-output

echo "======================================================================"
echo "Step 5/5: Evaluate Qwen predictions on OPT-RSVG teacher markers"
echo "======================================================================"
"${PYTHON_BIN}" "${LAE_DIR}/scripts/evaluate_opt_qwen_teacher_by_text_pairs.py" \
  --pred-file "${QWEN_OUT}" \
  --text-pairs-jsonl "${TEXT_PAIRS_JSONL}" \
  --marker-json-dir "${OUT_ROOT}/vis_pred_marker" | tee "${EVAL_TXT}"

echo "======================================================================"
echo "DONE: OPT-RSVG MTP teacher 10-class pipeline"
echo "Outputs: ${OUT_ROOT}"
echo "======================================================================"
