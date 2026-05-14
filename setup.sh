#!/usr/bin/env bash
# One-shot setup for the hybrid-rs codebase.
#
#   - Installs the package in editable mode.
#   - Clones the upstream diffusion_denoised_smoothing repo at the project root
#     so that hybrid_rs.utils_diffusion can import its guided_diffusion module.
#   - Verifies that every entry-point module imports cleanly.
#   - Reports whether the diffusion checkpoint is locatable.
#
# Usage:
#   bash setup.sh
#
# After this script, you still need to obtain the diffusion checkpoint
# (256x256_diffusion_uncond.pt) and point to it with one of:
#   - DIFFUSION_CKPT=/path/to/256x256_diffusion_uncond.pt
#   - HYBRID_RS_DATA_ROOT=/path/to/data_root  (checkpoint at $HYBRID_RS_DATA_ROOT/256x256_diffusion_uncond.pt)
#   - --diffusion_ckpt /path/... on the CLI

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFF_DIR="$ROOT/diffusion_denoised_smoothing"
DIFF_REPO="https://github.com/ethz-spylab/diffusion_denoised_smoothing"

echo "==> Project root: $ROOT"

if [ -f "$ROOT/env.local" ]; then
    echo "==> Sourcing $ROOT/env.local"
    # shellcheck disable=SC1090,SC1091
    source "$ROOT/env.local"
else
    echo "==> No env.local found. Copy env.example to env.local and edit if you"
    echo "    need to point at machine-specific dataset / checkpoint paths."
fi

echo "==> Installing hybrid-rs in editable mode"
pip install -e "$ROOT"

if [ -d "$DIFF_DIR/.git" ]; then
    echo "==> diffusion_denoised_smoothing already cloned at $DIFF_DIR"
else
    echo "==> Cloning $DIFF_REPO into $DIFF_DIR"
    git clone --depth 1 "$DIFF_REPO" "$DIFF_DIR"
fi

# Sanity check: the directory hybrid_rs.utils_diffusion expects must exist.
EXPECTED="$DIFF_DIR/imagenet/guided_diffusion/script_util.py"
if [ ! -f "$EXPECTED" ]; then
    echo "ERROR: expected file not found after clone: $EXPECTED" >&2
    echo "       Upstream layout may have changed." >&2
    exit 1
fi

echo "==> Adding $ROOT to PYTHONPATH for the import checks"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Verifying every entry-point module imports cleanly"
python - <<'PY'
import importlib, sys
mods = [
    "hybrid_rs",
    "hybrid_rs.utils_hybrid",
    "hybrid_rs.utils_diffusion",
    "hybrid_rs.data",
    "hybrid_rs.models",
    "hybrid_rs.svm_exp.SVM_exp",
    "hybrid_rs.certify.certify_text_L0",
    "hybrid_rs.certify.certify_multimodal_text_L0_image",
    "hybrid_rs.certify.certify_multimodal_text_append_suffix_image",
    "hybrid_rs.certify.certify_shared_D",
    "hybrid_rs.attack.predict_adv_text_image",
    "hybrid_rs.attack.predict_adv_text_image_multimodal",
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
        failed.append(m)
sys.exit(1 if failed else 0)
PY

echo "==> Resolving runtime paths from environment"
python - <<'PY'
import os
from pathlib import Path

def report(label, candidates, suffix=None):
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if suffix is not None:
            p = p / suffix
        if p.exists():
            print(f"  OK     {label}: {p}")
            return
    print(f"  MISSING {label} — checked: {[c for c in candidates if c] or '(none set)'}")

report("Diffusion checkpoint",
       [os.environ.get("DIFFUSION_CKPT"),
        (os.environ["HYBRID_RS_DATA_ROOT"] + "/256x256_diffusion_uncond.pt")
            if "HYBRID_RS_DATA_ROOT" in os.environ else None,
        "imagenet/256x256_diffusion_uncond.pt"])
report("UnsafeBench dataset",
       [os.environ.get("UNSAFEBENCH_DATASET_DIR"),
        (os.environ["HYBRID_RS_DATA_ROOT"] + "/unsafebench_interaction_unsafe_hf")
            if "HYBRID_RS_DATA_ROOT" in os.environ else None])
report("MM-SafetyBench root",
       [os.environ.get("MM_SAFETYBENCH_ROOT"),
        (os.environ["HYBRID_RS_DATA_ROOT"] + "/MM-SafetyBench")
            if "HYBRID_RS_DATA_ROOT" in os.environ else None])
report("Hateful memes dataset",
       [os.environ.get("HATEFUL_MEMES_DATASET_DIR"),
        (os.environ["HYBRID_RS_DATA_ROOT"] + "/HFdataset")
            if "HYBRID_RS_DATA_ROOT" in os.environ else None])
report("HF cache (Flickr30k)",
       [os.environ.get("FLICKR30K_CACHE_DIR"),
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE")])
PY

cat <<EOF

==> Setup complete.

To make the diffusion repo importable in new shells, add this to your shell rc
(or prepend it to commands):

    export PYTHONPATH="$ROOT:\$PYTHONPATH"

EOF
