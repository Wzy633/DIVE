# DIVE: Discrete Instance Visual Grounding for Remote Sensing

Official evaluation code for **DIVE**, a detector-marker-VLM framework that
reformulates remote sensing visual grounding as discrete instance selection.

DIVE first generates object candidates with a remote sensing detector, filters
and assigns a visible identifier to each retained candidate, asks a multimodal
large language model to select an identifier from the marked image, and finally
retrieves the corresponding detector box. The VLM therefore solves a compact
instance-disambiguation problem instead of directly regressing coordinates.

> This repository is the evaluation and inference release. It contains the
> candidate construction, marker rendering, text-pair construction, VLM
> inference, retry-safe result merging, and metric computation code used in the
> paper. Dataset files, detector checkpoints, API credentials, and generated
> results are intentionally not committed.

## Method at a Glance

Given an image `I` and a referring expression `q`, DIVE performs

```text
(I, q) -> detector candidates -> filtering -> visible marker IDs
       -> VLM identifier prediction -> deterministic ID-to-box retrieval
```

The corresponding formulation is

```text
D(I) = {(b_i, c_i, s_i)}                      raw detections
C(I) = Phi(D(I))                              retained candidates
M(I) = {(k, b^(k), c^(k), s^(k))}             indexed hypotheses
I_tilde = Gamma(I, M(I))                      marker image
k_hat = f_id(I_tilde, q)                      predicted identifier
b_hat = R(k_hat; M(I)) = b^(k_hat)            retrieved box
```

The detector handles geometry-sensitive proposal construction, while the VLM
handles language-guided instance comparison and relational reasoning.

## Repository Layout

```text
DIVE-GitHub/
|-- README.md
|-- requirements.txt
|-- LAE-DINO/
|   |-- mmdetection_lae/        # MMDetection 3.3.0-based LAE-DINO code/configs
|   `-- scripts/
|       |-- image_demo_dior_multiple.py
|       |-- visualize_marker.py
|       |-- rsvg_marker_pipeline_ours.py
|       |-- opt_rsvg_marker_pipeline_teacher_fixed.py
|       |-- opt_rsvg_marker_pipeline_teacher_mtp_fixed.py
|       |-- build_text_pairs_from_teacher_*.py
|       |-- qwen_marker_infer_openai_compat.py
|       |-- evaluate_*_qwen_*_by_text_pairs.py
|       |-- prepare_retry_subset_from_results.py
|       |-- merge_retry_results_by_manifest.py
|       `-- ours/               # LAE-DINO/MTP marker adapters
`-- third_party/
    `-- README.md               # optional MTP setup
```

The historical filename `qwen_marker_infer_openai_compat.py` is retained for
compatibility with existing experiments. The implementation is model-agnostic
and supports OpenAI-compatible multimodal endpoints, including Qwen, GPT, and
Gemini relays that implement `/v1/chat/completions` with image input.

## Installation

### 1. Create an environment

The detector code is based on MMDetection 3.3.0 and requires
`mmcv>=2.0.0rc4,<2.2.0` and `mmengine>=0.7.1,<1.0.0`.

```bash
conda create -n dive python=3.9 -y
conda activate dive

# Install a PyTorch build compatible with your CUDA runtime first.
pip install torch torchvision

pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
pip install -r requirements.txt
pip install -v -e ./LAE-DINO/mmdetection_lae
```

Verify the environment:

```bash
python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

### 2. Configure the optional MTP detector

LAE-DINO is included in this release. MTP is an optional external dependency:

```bash
git clone https://github.com/ViTAE-Transformer/MTP.git third_party/MTP
```

See [`third_party/README.md`](third_party/README.md) for the expected MTP
checkpoint and directory layout.

## Data Preparation

### DIOR-RSVG

Arrange the official dataset as follows:

```text
DIOR_RSVG/
|-- Annotations/
|-- JPEGImages/
|-- train.txt
|-- val.txt
`-- test.txt
```

The split files identify the images used for each split. The release scripts
create `_xml_train`, `_xml_val`, and `_xml_test` without modifying the original
annotations.

### OPT-RSVG

Arrange OPT-RSVG under `LAE-DINO/data/OPT-RSVG`:

```text
LAE-DINO/data/OPT-RSVG/
|-- Annotations/
|-- Image/
|-- train.txt
|-- val.txt
`-- test.txt
```

OPT-RSVG split files are sample-level rather than image-level. Build and
validate the manifest before running a detector:

```bash
python LAE-DINO/scripts/build_opt_rsvg_sample_index.py \
  --opt-root LAE-DINO/data/OPT-RSVG \
  --out-jsonl LAE-DINO/data/OPT-RSVG/opt_rsvg_samples.jsonl

python LAE-DINO/scripts/filter_opt_rsvg_test_samples.py \
  --samples-jsonl LAE-DINO/data/OPT-RSVG/opt_rsvg_samples.jsonl \
  --out-valid-jsonl LAE-DINO/data/OPT-RSVG/opt_rsvg_test_samples.valid.jsonl \
  --out-invalid-jsonl LAE-DINO/data/OPT-RSVG/opt_rsvg_test_samples.invalid.jsonl \
  --out-summary-md LAE-DINO/data/OPT-RSVG/opt_rsvg_test_samples.summary.md
```

The paper reports OPT-RSVG results on the category-compatible ten-class subset
covered by the DIOR-trained detector. The released run scripts create this
subset deterministically from the validated manifest.

## Detector Checkpoints

Checkpoints are not stored in Git because of their size. Use the exact file
names below or override every path with the corresponding environment variable.

| Detector | Default checkpoint path | Configuration |
|---|---|---|
| LAE-DINO | `LAE-DINO/work_dirs/lae_dino_swin-t_finetune_DIOR/epoch_26.pth` | `LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_finetune_DIOR_infer.py` |
| MTP | `third_party/MTP/weights/dior-rvsa-l-mae-mtp-epoch_12.pth` | MTP DIOR Faster R-CNN/RVSA-L config |

## Reproduce DIOR-RSVG

The released teacher pipeline follows the paper's detector-assisted,
category-constrained candidate protocol. Official referring expressions are
preserved; detector candidates are matched to the target in identifier space.

### 1. Generate LAE-DINO candidates and marker images

```bash
GPU_LIST=0 \
SPLITS=test \
RSVG_ROOT="$PWD/DIOR_RSVG" \
CKPT="$PWD/LAE-DINO/work_dirs/lae_dino_swin-t_finetune_DIOR/epoch_26.pth" \
bash LAE-DINO/scripts/run_marker_teacher_dior_rsvg_multigpu_resume.sh
```

Set `GPU_LIST=0,1,2,3` for multi-GPU inference. Existing marker JSON files are
detected and skipped, so the script is resume-safe.

### 2. Construct identifier-space text pairs

```bash
python LAE-DINO/scripts/build_text_pairs_from_teacher_rsvg_xml.py \
  --xml-dir DIOR_RSVG/_xml_test \
  --teacher-marker-dir LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/vis_pred_marker \
  --out-jsonl LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/text_pairs_rsvg_teacher.fixed.jsonl \
  --min-match-iou 0.5
```

### 3. Run a multimodal VLM

Do not put credentials in source files. Set them only in the current shell or
pass `--api-key`, `--base-url`, and `--model` explicitly.

Linux/macOS:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://YOUR_PROVIDER/v1"
export OPENAI_MODEL="qwen3.6-plus"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:OPENAI_BASE_URL = "https://YOUR_PROVIDER/v1"
$env:OPENAI_MODEL = "qwen3.6-plus"
```

Run inference:

```bash
python LAE-DINO/scripts/qwen_marker_infer_openai_compat.py \
  --jsonl LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/text_pairs_rsvg_teacher.fixed.jsonl \
  --images-dir LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/vis_pred_marker \
  --out LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/results_qwen3_6_plus.txt \
  --usage-log LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/usage_qwen3_6_plus.jsonl \
  --model qwen3.6-plus \
  --temperature 0 \
  --max-tokens 128 \
  --answer-only-output \
  --resume
```

The expected answer format is `<answer>ID</answer>`. The parser also accepts a
small set of legacy answer forms, then normalizes successful predictions to the
same identifier representation.

### 4. Evaluate

```bash
python LAE-DINO/scripts/evaluate_rsvg_qwen_teacher_by_text_pairs.py \
  --pred-file LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/results_qwen3_6_plus.txt \
  --text-pairs-jsonl LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/text_pairs_rsvg_teacher.fixed.jsonl \
  --marker-json-dir LAE-DINO/data/DIOR_RSVG_marked/teacher/epoch_26/test/vis_pred_marker
```

The evaluator reports ID accuracy, meanIoU, cumIoU, and
`Pr@{0.5,0.6,0.7,0.8,0.9}`. For compatibility with the experiment logs, the
same threshold score is also printed using the `PR` and `AP` aliases.

## Reproduce OPT-RSVG

### LAE-DINO candidates

After building `opt_rsvg_test_samples.valid.jsonl`, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OPT_ROOT="$PWD/LAE-DINO/data/OPT-RSVG" \
CKPT="$PWD/LAE-DINO/work_dirs/lae_dino_swin-t_finetune_DIOR/epoch_26.pth" \
QWEN_MODEL="qwen3.6-plus" \
bash LAE-DINO/scripts/run_opt_laedino_teacher_10cls_qwen_eval.sh
```

### MTP candidates

```bash
CUDA_VISIBLE_DEVICES=0 \
MTP_ROOT="$PWD/third_party/MTP" \
MARKER_CKPT="$PWD/third_party/MTP/weights/dior-rvsa-l-mae-mtp-epoch_12.pth" \
QWEN_MODEL="qwen3.6-plus" \
bash LAE-DINO/scripts/run_opt_mtp_teacher_10cls_qwen_eval.sh
```

Both scripts execute candidate filtering, marker generation, text-pair
construction, VLM inference, and evaluation. Set environment variables such as
`OUT_ROOT`, `DEVICE`, `QWEN_MODEL`, and `QWEN_MAX_TOKENS` to override defaults.

## Retry Failed API Samples Safely

API failures and unparsable responses are written as warning-bearing records.
The retry utilities use an explicit manifest, so repeated image names or
non-contiguous failures cannot shift predictions to another sample.

```bash
python LAE-DINO/scripts/prepare_retry_subset_from_results.py \
  --text-pairs-jsonl FULL_TEXT_PAIRS.jsonl \
  --results-file RESULTS.txt \
  --out-jsonl RETRY_TEXT_PAIRS.jsonl \
  --out-manifest RETRY_MANIFEST.jsonl \
  --out-summary-json RETRY_SUMMARY.json

python LAE-DINO/scripts/qwen_marker_infer_openai_compat.py \
  --jsonl RETRY_TEXT_PAIRS.jsonl \
  --images-dir MARKER_IMAGE_DIR \
  --out RETRY_RESULTS.txt \
  --model YOUR_MODEL \
  --temperature 0 \
  --answer-only-output

python LAE-DINO/scripts/merge_retry_results_by_manifest.py \
  --original-results RESULTS.txt \
  --retry-results RETRY_RESULTS.txt \
  --retry-manifest RETRY_MANIFEST.jsonl \
  --out-results RESULTS_MERGED.txt
```

Always evaluate `RESULTS_MERGED.txt`, not the retry-only file.

## Evaluation Protocol

- `meanIoU` is the sample-wise mean of predicted-box IoU.
- `cumIoU` is the sum of intersections divided by the sum of unions.
- `Pr@t` is the fraction of evaluated samples with `IoU >= t`.
- Invalid or missing candidate lookups receive IoU zero.
- The released evaluators default to `--exclude-pred-ids 0`, matching the
  experiment scripts' placeholder-ID protocol. Pass an empty value to disable
  this exclusion when auditing a different protocol.
- DIOR-RSVG uses the official expressions and detector-derived candidate boxes.
- OPT-RSVG uses the validated, unambiguous, detector-compatible ten-class
  subset; excluded samples are recorded by the preprocessing scripts.

## Main Results

Results below are reported as fractions and were reproduced with the released
evaluation scripts.

### DIOR-RSVG test set

| Detector | VLM | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAE-DINO | Qwen3.6-Plus | 0.8680 | 0.8520 | 0.8189 | 0.7317 | 0.5392 | 0.7816 | 0.8498 |
| MTP | Qwen3.6-Plus | 0.8690 | 0.8582 | 0.8322 | 0.7630 | 0.5470 | 0.7836 | 0.8446 |

### OPT-RSVG compatible test subset

| Detector | VLM | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAE-DINO | Qwen3.6-Plus | 0.8520 | 0.8099 | 0.7479 | 0.6052 | 0.2769 | 0.7184 | 0.7424 |

## Reproducibility Notes

- Use deterministic decoding (`temperature=0`) for all reported VLM results.
- Preserve JSONL order. Do not concatenate retry results directly; use the
  manifest-aware merge utility.
- Record the exact detector checkpoint, VLM model ID, endpoint provider, and
  preprocessing summary with every run.
- Marker IDs are image-local. An identifier must never be reused as a global
  sample index.
- Generated images, marker JSON, result text files, usage logs, and checkpoints
  are ignored by Git by default.

## Acknowledgements

This code builds on MMDetection and supports the optional
[MTP](https://github.com/ViTAE-Transformer/MTP) detector. Please follow the
licenses and citation requirements of the original datasets and detector
implementations.

## Citation

The BibTeX entry will be added after publication. If this repository helps your
research, please cite the DIVE paper and the detector/dataset papers used in your
experiments.
