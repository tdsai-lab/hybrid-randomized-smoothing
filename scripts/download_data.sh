#!/usr/bin/env bash
# Download the publicly available pieces of the data pipeline.
#
# Covers:
#   - the OpenAI guided-diffusion 256x256 unconditional checkpoint
#     (used by --use_diffusion / hybrid_rs.utils_diffusion)
#   - the raw MM-SafetyBench question files (images must be downloaded
#     separately following the upstream repo's README — they're on Drive)
#
# Does NOT cover:
#   - UnsafeBench / Hateful Memes (registration or custom preprocessing
#     required; see README.md "Building the variants")
#   - Flickr30k (auto-pulled by `datasets` on first use)
#
# Usage:
#   bash scripts/download_data.sh /path/to/data_root
#
# After this script:
#   export DIFFUSION_CKPT=/path/to/data_root/256x256_diffusion_uncond.pt
#   export MM_SAFETYBENCH_ROOT=/path/to/data_root/MM-SafetyBench

set -euo pipefail

DATA_ROOT="${1:-${HYBRID_RS_DATA_ROOT:-./data}}"
mkdir -p "$DATA_ROOT"
echo "==> Data root: $DATA_ROOT"

CKPT="$DATA_ROOT/256x256_diffusion_uncond.pt"
CKPT_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
if [ -f "$CKPT" ]; then
    echo "==> Diffusion checkpoint already present: $CKPT"
else
    echo "==> Downloading diffusion checkpoint (~2.1 GB) to $CKPT"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$CKPT" "$CKPT_URL"
    else
        curl -L -o "$CKPT" "$CKPT_URL"
    fi
fi

MM_DIR="$DATA_ROOT/MM-SafetyBench"
if [ -d "$MM_DIR/.git" ]; then
    echo "==> MM-SafetyBench repo already cloned: $MM_DIR"
else
    echo "==> Cloning MM-SafetyBench question metadata into $MM_DIR"
    git clone --depth 1 https://github.com/isXinLiu/MM-SafetyBench "$MM_DIR"
fi

if [ ! -d "$MM_DIR/data/imgs" ]; then
    cat <<EOF

==> NOTE: MM-SafetyBench images are hosted on Google Drive, not in the git
    repo. Follow the upstream README to download them into:

        $MM_DIR/data/imgs/

    Once the images are in place, the loader will pick them up automatically.

EOF
fi

cat <<EOF

==> Public-data download complete.

Suggested env vars to add to env.local:

    export DIFFUSION_CKPT=$CKPT
    export MM_SAFETYBENCH_ROOT=$MM_DIR

EOF
