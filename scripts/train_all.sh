#!/usr/bin/env bash
# Full training sweep on A40 — saturated config.
#
# Defaults: bf16 autocast, batch=64, crop=256, n=192, m=320 (≈17 M-param codec).
# Use after scripts/setup_remote.sh and after data is in $DATA.
#
# Wallclock estimate (A40, 15 sequences, ~3-5k pairs after preprocessing):
#   - I-frame run ~ 30-45 min for 60 epochs at this scale
#   - P-frame run ~ 45-60 min (slightly heavier — extra warp module)
#   - Three lambdas × (1 I-frame + 2 P-frame variants) ≈ 9 runs ≈ 5-7 hours
#
# Override LAMBDAS / EPOCHS / BATCH if needed.

set -euo pipefail

DATA="${DATA:-data/aria_proc}"
OUT="${OUT:-runs}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-128}"
CROP="${CROP:-256}"
N="${N:-320}"
M="${M:-448}"
WORKERS="${WORKERS:-8}"
LENGTH_PER_SEQ="${LENGTH_PER_SEQ:-2000}"
LAMBDAS=(${LAMBDAS:-0.0018 0.0067 0.0250})

mkdir -p "$OUT"

COMMON=(--data "$DATA" --out "$OUT" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --crop "$CROP" --n "$N" --m "$M" --workers "$WORKERS" \
        --length-per-seq "$LENGTH_PER_SEQ")

# 1) I-frame codec, one run per lambda.
for L in "${LAMBDAS[@]}"; do
  echo "=== I-frame lambda=$L ==="
  python -m ego_codec.train iframe "${COMMON[@]}" --lmbda "$L"
done

# 2) P-frame variants, initialized from matching I-frame ckpt.
for L in "${LAMBDAS[@]}"; do
  IFRAME_CKPT="$OUT/iframe-l${L}/best.pt"
  echo "=== P-frame IMU lambda=$L (init from $IFRAME_CKPT) ==="
  python -m ego_codec.train pframe "${COMMON[@]}" --iframe-ckpt "$IFRAME_CKPT" --lmbda "$L"

  echo "=== P-frame NO-IMU ablation lambda=$L ==="
  python -m ego_codec.train pframe-noimu "${COMMON[@]}" --iframe-ckpt "$IFRAME_CKPT" --lmbda "$L"
done

# 3) x265 industry baseline (requires ffmpeg with libx265).
if command -v ffmpeg >/dev/null 2>&1; then
  echo "=== x265 baseline ==="
  python scripts/eval_x265.py --data "$DATA" --out figures/x265_baseline.json --max-pairs 200 || true
  X265_FLAG=(--x265-json figures/x265_baseline.json)
else
  echo "[skip] ffmpeg not found — skipping x265 baseline."
  X265_FLAG=()
fi

# 4) Build the RD curve plot.
CONDS=()
for L in "${LAMBDAS[@]}"; do
  CONDS+=(--condition iframe-aria "$OUT/iframe-l${L}/best.pt")
  CONDS+=(--condition pframe-imu "$OUT/pframe-imu-l${L}/best.pt")
  CONDS+=(--condition pframe-noimu "$OUT/pframe-noimu-l${L}/best.pt")
done
python -m ego_codec.eval --data "$DATA" --out figures "${X265_FLAG[@]}" "${CONDS[@]}"
echo "Done. See figures/rd_curve.png"
