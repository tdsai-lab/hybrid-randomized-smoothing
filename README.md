# Hybrid Randomized Smoothing for Multimodal Certified Robustness

Hybrid randomized-smoothing certification over text + image inputs against a
LLaVA-Guard-based multimodal safety detector.

This repository contains code for reproducing the main certification routines
from:

**Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized
Smoothing**
Blaise Delattre, Hengyu Wu, Paul Caillon, Wei Yang Bryan Lim, Yang Cao.
ICML 2026. arXiv: <https://arxiv.org/abs/2605.12876>

The code supports:

- hybrid text + image randomized-smoothing certificates;
- text-only certificates under the same discrete threat model;
- multimodal `L0` text + image certification baselines;
- adaptive text + image attack / prediction routines;
- a tabular SVM experiment on the Adult dataset.

## Layout

    src/hybrid_rs/
      certify/   # joint text + image certification entry points
      attack/    # prediction / attack entry points
      svm_exp/   # tabular SVM baseline (Adult dataset)
      data/      # dataset loaders
      models/    # LLaVAGuardSafetyDetector + policy prompts
      utils_diffusion.py
      utils_hybrid.py

## Install

Create an environment with PyTorch, then install the package in editable mode:

    pip install -e .

Scripts are run as package modules, for example:

    python -m hybrid_rs.certify.certify_multimodal_text_append_suffix_image ...

## Setup

### Diffusion denoiser

Image-side certification can optionally use denoised randomized smoothing,
following Carlini et al., **Certified!! Adversarial Robustness for Free!**

Clone the original denoised randomized-smoothing repository at the root of this
repository:

    git clone https://github.com/ethz-spylab/diffusion_denoised_smoothing

Expected layout:

    hybrid-rs/
      diffusion_denoised_smoothing/
      src/
      scripts/
      ...

To enable image denoising, provide the path to the guided-diffusion checkpoint
via one of:

- `export DIFFUSION_CKPT=/path/to/256x256_diffusion_uncond.pt`
- `export HYBRID_RS_DATA_ROOT=/path/to/data_root` and place the checkpoint at
  `$HYBRID_RS_DATA_ROOT/256x256_diffusion_uncond.pt`
- `--diffusion_ckpt /path/to/256x256_diffusion_uncond.pt` on the CLI

The checkpoint (~2.1 GB, unconditional 256x256 ImageNet) is released by OpenAI
as part of [guided-diffusion](https://github.com/openai/guided-diffusion) and
can be downloaded directly:

    wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

`scripts/download_data.sh` fetches it automatically.

## Datasets

Path resolution: dataset paths are read from environment variables.

The simplest setup is to define a common data root:

    export HYBRID_RS_DATA_ROOT=/path/to/data_root

Individual locations can also be overridden via:

- `UNSAFEBENCH_DATASET_DIR`
- `MM_SAFETYBENCH_ROOT`
- `FLICKR30K_CACHE_DIR`
- `HF_HOME`

Recognized dataset names passed through `--dataset`:

- `unsafebench_interaction_unsafe_mm`

### Where to get the source data

| Dataset | Source | Format expected by the loader |
| --- | --- | --- |
| MM-SafetyBench | `git clone https://github.com/isXinLiu/MM-SafetyBench` and follow the upstream README to download images | Raw layout with `<root>/data/processed_questions/*.json` and `<root>/data/imgs/<scenario>/<SD|SD_TYPO|TYPO>/*.jpg` |
| MM-SafetyBench HF | `huggingface-cli download liuxin2002/MM-SafetyBench --repo-type dataset` | Same logical layout; point `MM_SAFETYBENCH_ROOT` to the extracted root |
| UnsafeBench raw | `huggingface-cli download yiting/UnsafeBench --repo-type dataset` | HuggingFace dataset |
| Flickr30k | Auto-pulled by `datasets` from `nlphuji/flickr30k` on first use | Cached under `$HF_HOME` or `$FLICKR30K_CACHE_DIR` |
| Diffusion checkpoint | Guided-diffusion checkpoint used by denoised randomized smoothing | Place at `$DIFFUSION_CKPT` or `$HYBRID_RS_DATA_ROOT/256x256_diffusion_uncond.pt` |

### Quick start with raw MM-SafetyBench

Clone MM-SafetyBench:

    git clone https://github.com/isXinLiu/MM-SafetyBench /path/to/MM-SafetyBench

Follow the upstream MM-SafetyBench README to download the images into:

    /path/to/MM-SafetyBench/data/imgs/

Then set:

    export MM_SAFETYBENCH_ROOT=/path/to/MM-SafetyBench

The loader can then be used with:

    --dataset unsafebench_interaction_unsafe_mm
    --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT

Expected raw layout:

    MM-SafetyBench/
      data/
        processed_questions/
          *.json
        imgs/
          <scenario>/
            SD/
            SD_TYPO/
            TYPO/

### Download helper

The helper script:

    scripts/download_data.sh /path/to/data_root

handles public auxiliary files used by the codebase, such as the diffusion
checkpoint and MM-SafetyBench question files.

MM-SafetyBench images must still be downloaded from the upstream source
according to the upstream instructions.

## 1. Hybrid certification: append-only text + image noise

Threat model:

- Text threat: append `m` tokens, then corrupt the first `d` appended
  positions.
- Image threat: Gaussian noise with standard deviation `sigma`.
- The script loops over `d = 1, ..., m` and estimates `p_A(d)` with `n`
  Monte Carlo samples.

Installation smoke test (not a valid paper reproduction — runs in seconds, only checks that the pipeline executes end-to-end):

    python -m hybrid_rs.certify.certify_multimodal_text_append_suffix_image \
      --dataset unsafebench_interaction_unsafe_mm \
      --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT \
      --text_threat append_only \
      --kernel uniform \
      --beta 0.25 \
      --m 2 \
      --n 10 \
      --batch_size 4 \
      --sigma 0.5 \
      --tau 4.6e-5 \
      --max_examples 1 \
      --no-use_diffusion \
      --guard_decision_rule logit_mass

Main run:

    python -m hybrid_rs.certify.certify_multimodal_text_append_suffix_image \
      --dataset unsafebench_interaction_unsafe_mm \
      --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT \
      --text_threat append_only \
      --kernel uniform \
      --beta 0.25 \
      --m 8 \
      --n 10000 \
      --batch_size 32 \
      --sigma 0.5 \
      --tau 4.6e-5 \
      --use_diffusion \
      --guard_decision_rule logit_mass \
      --flash_attn

## 2. Text-only certification

Pure-text certificate under the same text threat model. This is useful for
comparing hybrid certification against text-only certification.

    python -m hybrid_rs.certify.certify_text_L0 \
      --dataset unsafebench_interaction_unsafe_mm \
      --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT \
      --kernel uniform \
      --beta 0.25 \
      --n 10000 \
      --batch_size 32 \
      --tau 4.6e-5 \
      --guard_decision_rule logit_mass \
      --flash_attn

## 3. Multimodal `L0` text + image certification baseline

This script evaluates an `L0` text threat combined with image perturbations.
The uniform text kernel is used by default unless another kernel is specified.

    python -m hybrid_rs.certify.certify_multimodal_text_L0_image \
      --dataset unsafebench_interaction_unsafe_mm \
      --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT \
      --kernel uniform \
      --beta 0.25 \
      --n 10000 \
      --batch_size 32 \
      --sigma 0.5 \
      --tau 4.6e-5 \
      --guard_decision_rule logit_mass \
      --flash_attn

## 4. Attack / prediction

Prediction under joint text + image adversarial attack, with optional
randomized-smoothing evaluation at the attacked point.

    python -m hybrid_rs.attack.predict_adv_text_image_multimodal \
      --dataset unsafebench_interaction_unsafe_mm \
      --unsafebench_dataset_dir $MM_SAFETYBENCH_ROOT \
      --batch_size 8 \
      --max_examples 100 \
      --guard_decision_rule logit_mass \
      --flash_attn

## 5. SVM baseline

Tabular SVM experiment on the Adult dataset, independent of the multimodal
pipeline.

    python -m hybrid_rs.svm_exp.SVM_exp

## Citation

If you use this code, please cite:

    @inproceedings{delattre2026hybridrs,
      title         = {Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing},
      author        = {Delattre, Blaise and Wu, Hengyu and Caillon, Paul and Lim, Wei Yang Bryan and Cao, Yang},
      booktitle     = {Proceedings of the International Conference on Machine Learning},
      year          = {2026},
    }

## License

[MIT](LICENSE) © 2026 TDSAI Lab @ Science Tokyo.
