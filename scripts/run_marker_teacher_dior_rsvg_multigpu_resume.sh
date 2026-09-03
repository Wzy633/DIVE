#!/usr/bin/env bash
# Teacher-style DIOR_RSVG marker with multi-GPU parallelism + resume.
#
# Features:
# - Multi-GPU parallel detection for pending samples only
# - Resume-safe: skip images already finished in vis_pred_marker/*.json
# - Keep teacher pipeline logic:
#     image_demo_dior_multiple.py -> outputs/preds
#     visualize_marker.py         -> vis_pred_marker/{jpg,json}
#
# Default GPU: 0. Set GPU_LIST=0,1,... for parallel inference.
#
# Usage:
#   bash LAE-DINO/scripts/run_marker_teacher_dior_rsvg_multigpu_resume.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="${ROOT}/LAE-DINO"
MMDET="${PROJ}/mmdetection_lae"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
GPU_LIST="${GPU_LIST:-0}"   # physical GPU ids
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

CKPT="${CKPT:-${PROJ}/work_dirs/lae_dino_swin-t_finetune_DIOR/epoch_26.pth}"
CFG="${CFG:-${MMDET}/configs/lae_dino/lae_dino_swin-t_finetune_DIOR_infer.py}"

RSVG_ROOT="${RSVG_ROOT:-${ROOT}/DIOR_RSVG}"
OUT_ROOT="${OUT_ROOT:-${PROJ}/data/DIOR_RSVG_marked/teacher/epoch_26}"

SPLITS="${SPLITS:-test,val,train}"
IFS=',' read -r -a SPLIT_ARR <<< "${SPLITS}"

if [[ -n "${CONDA_SH}" ]]; then
  [[ -f "${CONDA_SH}" ]] || { echo "Missing conda.sh: ${CONDA_SH}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  [[ -n "${CONDA_ENV}" ]] && conda activate "${CONDA_ENV}"
fi

echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] GPU_LIST=${GPU_LIST}"

python - <<'PY'
import torch
print("[INFO] torch", torch.__version__)
print("[INFO] cuda_available", torch.cuda.is_available())
print("[INFO] cuda_device_count", torch.cuda.device_count())
PY

build_xml_split_if_needed () {
  local split="$1"
  local xml_dir="${RSVG_ROOT}/_xml_${split}"
  mkdir -p "${xml_dir}"
  shopt -s nullglob
  local arr=("${xml_dir}"/*.xml)
  shopt -u nullglob
  if [[ ${#arr[@]} -eq 0 ]]; then
    python "${PROJ}/scripts/rsvg_make_split_xml_dir.py" \
      --dior-rsvg-root "${RSVG_ROOT}" --split "${split}" --out-dir "${xml_dir}"
  fi
}

run_split () {
  local split="$1"
  local xml_dir="${RSVG_ROOT}/_xml_${split}"
  local out_dir="${OUT_ROOT}/${split}"
  local main_preds="${out_dir}/outputs/preds"
  local main_vis="${out_dir}/outputs/vis"
  local main_marker="${out_dir}/vis_pred_marker"
  local work_dir="${out_dir}/_resume_work"

  mkdir -p "${main_preds}" "${main_vis}" "${main_marker}" "${work_dir}"
  build_xml_split_if_needed "${split}"

  # 1) Compute pending stems (xml stems not yet finished in vis_pred_marker json)
  local pending_txt="${work_dir}/pending_stems.txt"
  python - <<PY
from pathlib import Path
xml_dir = Path(r"${xml_dir}")
marker_dir = Path(r"${main_marker}")
pending = []
for x in sorted(xml_dir.glob("*.xml")):
    stem = x.stem
    done = (marker_dir / f"{stem}.json").exists()
    if not done:
        pending.append(stem)
Path(r"${pending_txt}").write_text("\\n".join(pending), encoding="utf-8")
print(f"[INFO] split=${split} total_xml={len(list(xml_dir.glob('*.xml')))} pending={len(pending)}")
PY

  local pending_n
  pending_n=$(wc -l < "${pending_txt}" || echo 0)
  if [[ "${pending_n}" -eq 0 ]]; then
    echo "[INFO] split=${split} nothing pending, skip."
    return 0
  fi

  # 2) Determine stems that still need detection (no preds json yet)
  local detect_txt="${work_dir}/need_detect_stems.txt"
  python - <<PY
from pathlib import Path
pending = [s.strip() for s in Path(r"${pending_txt}").read_text(encoding="utf-8").splitlines() if s.strip()]
preds = Path(r"${main_preds}")
need = [s for s in pending if not (preds / f"{s}.json").exists()]
Path(r"${detect_txt}").write_text("\\n".join(need), encoding="utf-8")
print(f"[INFO] split=${split} need_detect={len(need)}")
PY

  local need_detect_n
  need_detect_n=$(wc -l < "${detect_txt}" || echo 0)

  # 3) Multi-GPU detect on only missing preds
  if [[ "${need_detect_n}" -gt 0 ]]; then
    echo "[INFO] split=${split} launching parallel detect on ${#GPUS[@]} GPUs ..."
    local i=0
    local pids=()
    for gpu in "${GPUS[@]}"; do
      local shard_stems="${work_dir}/need_detect_gpu${gpu}.txt"
      local shard_root="${work_dir}/shard_gpu${gpu}"
      local shard_xml_dir="${shard_root}/Annotations"
      local shard_jpeg_link="${shard_root}/JPEGImages"
      local shard_out="${work_dir}/out_gpu${gpu}"
      mkdir -p "${shard_xml_dir}" "${shard_out}" "${shard_root}"
      # keep teacher script path inference happy: parent(Annotations)/JPEGImages
      if [[ ! -e "${shard_jpeg_link}" ]]; then
        ln -s "${RSVG_ROOT}/JPEGImages" "${shard_jpeg_link}"
      fi
      # round-robin split by line number
      awk -v idx="${i}" -v n="${#GPUS[@]}" 'NF{ if((NR-1)%n==idx) print $0 }' "${detect_txt}" > "${shard_stems}"
      local shard_n
      shard_n=$(wc -l < "${shard_stems}" || echo 0)
      if [[ "${shard_n}" -eq 0 ]]; then
        i=$((i+1))
        continue
      fi

      # materialize shard xml directory
      python - <<PY
from pathlib import Path
xml_dir = Path(r"${xml_dir}")
dst = Path(r"${shard_xml_dir}")
dst.mkdir(parents=True, exist_ok=True)
for s in Path(r"${shard_stems}").read_text(encoding="utf-8").splitlines():
    s=s.strip()
    if not s: continue
    src = xml_dir / f"{s}.xml"
    if src.exists():
        (dst / src.name).write_bytes(src.read_bytes())
PY

      (
        set -euo pipefail
        cd "${MMDET}"
        CUDA_VISIBLE_DEVICES="${gpu}" python "${PROJ}/scripts/image_demo_dior_multiple.py" \
          --inputs "${shard_xml_dir}" \
          --model "${CFG}" \
          --weights "${CKPT}" \
          --device "cuda:0" \
          --out-dir "${shard_out}" \
          --pred-score-thr 0.4 --batch-size 2 --palette random
      ) &
      pids+=("$!")
      i=$((i+1))
    done

    # wait parallel jobs
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    echo "[INFO] split=${split} parallel detect finished."

    # merge shard outputs back to main outputs
    for gpu in "${GPUS[@]}"; do
      local shard_out="${work_dir}/out_gpu${gpu}"
      local sp="${shard_out}/preds"
      local sv="${shard_out}/vis"
      if [[ -d "${sp}" ]]; then
        cp -n "${sp}"/*.json "${main_preds}/" 2>/dev/null || true
      fi
      if [[ -d "${sv}" ]]; then
        cp -n "${sv}"/*.jpg "${main_vis}/" 2>/dev/null || true
      fi
    done
  fi

  # 4) Visualize marker for pending stems only (uses main preds)
  local pending_xml_dir="${work_dir}/pending_xml"
  mkdir -p "${pending_xml_dir}"
  python - <<PY
from pathlib import Path
xml_dir = Path(r"${xml_dir}")
pending = [s.strip() for s in Path(r"${pending_txt}").read_text(encoding="utf-8").splitlines() if s.strip()]
dst = Path(r"${pending_xml_dir}")
dst.mkdir(parents=True, exist_ok=True)
for s in pending:
    src = xml_dir / f"{s}.xml"
    if src.exists():
        (dst / src.name).write_bytes(src.read_bytes())
print(f"[INFO] split=${split} pending_xml_prepared={len(list(dst.glob('*.xml')))}")
PY

  (
    set -euo pipefail
    cd "${MMDET}"
    python "${PROJ}/scripts/visualize_marker.py" \
      --preds-dir "${main_preds}" \
      --annotations-dir "${pending_xml_dir}" \
      --images-dir "${RSVG_ROOT}/JPEGImages" \
      --output-dir "${main_marker}" \
      --score-threshold 0.25
  )

  # 5) Final progress report for split
  python - <<PY
from pathlib import Path
xml_n = len(list(Path(r"${xml_dir}").glob("*.xml")))
pred_n = len(list(Path(r"${main_preds}").glob("*.json")))
mk_n = len(list(Path(r"${main_marker}").glob("*.json")))
print(f"[DONE] split=${split} xml={xml_n} preds={pred_n} marker_json={mk_n}")
PY
}

for s in "${SPLIT_ARR[@]}"; do
  run_split "${s}"
done

echo "[DONE] teacher DIOR_RSVG multi-gpu resume -> ${OUT_ROOT}"
