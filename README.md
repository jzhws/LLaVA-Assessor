# LLaVA-Assessor

This repository contains the training and evaluation code used for the LLaVA-Assessor video/image quality assessment workflows. It is organized as three self-contained code snapshots:

- `training/foundation`: foundation-model fine-tuning code copied from `train_foundation`.
- `evaluation/scoring`: quality scoring inference/evaluation code.
- `evaluation/interpreting`: quality interpreting and benchmark VQA evaluation code.

Large files are intentionally not included. Model checkpoints, optimizer states, generated outputs, local datasets, result JSON files, media examples, and wheel files were excluded from the repository.

## Repository Layout

```text
training/foundation/
  finetune_onevision.sh          # main full-parameter fine-tuning entry
  scripts/finetune_onevision.sh  # portable example fine-tuning script
  scripts/zero*.json             # DeepSpeed configs
  llava/                         # LLaVA-Assessor training/model code
  trl/                           # bundled TRL training utilities

evaluation/scoring/
  predict.py                     # prediction helper
  llava/eval/model_score_UGC_video.py
  llava/eval/model_score_image.py
  llava/                         # scoring model/eval code
  trl/                           # bundled TRL utilities

evaluation/interpreting/
  app.py                         # interactive interpreting demo
  predict.py                     # prediction helper
  llava-openvision-vllm.py       # vLLM/OpenVision helper
  llava/eval/model_vqa_q_bench_video.py
  llava/eval/model_vqa_image.py
  llava/eval/model_vqa_q_bench_FG.py
  llava/                         # interpreting model/eval code
  trl/                           # bundled TRL utilities
```

## Environment

Use Python 3.10 and install each subproject in editable mode from the directory you want to run. The code follows the original LLaVA-OneVision/LLaVA-NeXT dependency stack and expects CUDA GPUs for training and inference.

```bash
cd training/foundation
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

Repeat the same pattern inside `evaluation/scoring` or `evaluation/interpreting` when running those workflows independently.

Notes:

- The original training setup used `transformers==4.44.0`.
- Full fine-tuning used 8 A100/A800-class GPUs with DeepSpeed ZeRO-3, `per_device_train_batch_size=1`, and gradient accumulation.
- Install `flash-attn` from your own environment-compatible wheel or package index. The source wheel was not committed.
- Checkpoint directories must be provided locally or downloaded separately; no model weights are stored in this repository.

## Training

The main training entry is:

```bash
cd training/foundation
bash finetune_onevision.sh
```

Before running, edit the following values in `training/foundation/finetune_onevision.sh` or create your own copy of the script:

- `LLM_VERSION`: base checkpoint or previous-stage checkpoint.
- `--data_path`: JSON training annotation file.
- `--video_folder`: root directory for training videos.
- `--image_folder`: root directory for training images.
- `--output_dir`: destination for checkpoints and logs.
- `deepspeed --include`: GPU IDs available on the machine.
- `--save_steps`, `--save_total_limit`, `--num_train_epochs`, and batch settings as needed.

The portable example at `training/foundation/scripts/finetune_onevision.sh` uses relative paths such as `./example.json` and `./model_output`; replace those placeholders with your actual local dataset and output paths.

Expected training JSON format follows the LLaVA conversation style, with each sample pointing to local image/video assets and containing `conversations` entries for user prompts and assistant responses. Dataset files are excluded, so prepare them outside the repository.

## Quality Scoring Evaluation

Use the scoring subproject for scalar quality prediction and correlation evaluation.

Video scoring:

```bash
cd evaluation/scoring
python llava/eval/model_score_UGC_video.py \
  --model-path /path/to/llava-assessor-checkpoint \
  --model-base None
```

Image scoring:

```bash
cd evaluation/scoring
python llava/eval/model_score_image.py \
  --model-path /path/to/llava-assessor-checkpoint \
  --model-base None
```

Before running, edit the dataset paths inside each script:

- `image_paths`: image or video root directories.
- `json_prefix` and `jsons`: benchmark annotation JSON files.
- output paths under `results/` or any hard-coded local result directory.

The scoring scripts compute logits for ordinal quality words such as `high`, `good`, `fair`, `poor`, and `low`, convert them to a weighted average score, and report Spearman/Pearson correlations against MOS-style ground truth fields.

## Quality Interpreting Evaluation

Use the interpreting subproject for open-ended and multiple-choice VQA-style quality assessment.

Video benchmark evaluation:

```bash
cd evaluation/interpreting
python llava/eval/model_vqa_q_bench_video.py \
  --model-path /path/to/llava-assessor-checkpoint \
  --model-base None \
  --temperature 0
```

Image benchmark evaluation:

```bash
cd evaluation/interpreting
python llava/eval/model_vqa_image.py \
  --model-path /path/to/llava-assessor-checkpoint \
  --model-base None \
  --temperature 0
```

Additional entry points include:

- `llava/eval/model_vqa_q_bench_FG.py` for fine-grained Q-Bench style evaluation.
- `llava/eval/model_vqa_llvisionqa_pair.py` for paired visual-quality QA.
- `app.py` for an interactive local demo.

As with scoring, the benchmark JSON files and media roots are not committed. Update the hard-coded local paths in the scripts before running them in a new environment.

## Files Not Included

The following classes of files were deliberately excluded:

- Model checkpoints and generated model directories: `model_output*`, `checkpoint-*`, `*.safetensors`, `*.pth`, `*.pt`, `*.bin`.
- Dataset and benchmark annotation dumps from the source environment.
- Evaluation result files under `results/`.
- Local media examples such as `*.mp4`, `*.jpg`, `*.png`.
- Environment-specific binary wheels such as `flash_attn*.whl`.

Keep these files in external storage and reference them with local paths when launching training or evaluation.
