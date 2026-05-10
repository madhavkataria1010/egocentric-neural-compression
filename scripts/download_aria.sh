#!/usr/bin/env bash
# Bootstrap a small Aria Pilot Dataset / Aria Everyday Activities subset.
#
# We don't pin a specific dataset URL because Project Aria gates downloads behind a
# per-user CDN URL list (you sign the license at https://www.projectaria.com/datasets/).
# This script:
#   1. Reads ARIA_URLS_FILE (one signed URL per line) from your env
#   2. Downloads the first N (default 5) sequences into ./data/aria_raw
#   3. Runs scripts/preprocess_aria.py to extract frames + IMU
#
# Usage:
#   export ARIA_URLS_FILE=~/Downloads/aria_aea_urls.txt
#   ./scripts/download_aria.sh           # 5 sequences, 10 fps
#   N=10 FPS=15 ./scripts/download_aria.sh

set -euo pipefail

N="${N:-5}"
FPS="${FPS:-10}"
RAW_DIR="${RAW_DIR:-./data/aria_raw}"
OUT_DIR="${OUT_DIR:-./data/aria_proc}"

if [[ -z "${ARIA_URLS_FILE:-}" ]]; then
  echo "ERROR: set ARIA_URLS_FILE to the path of your signed URL list (one URL per line)." >&2
  echo "       Get it from https://www.projectaria.com/datasets/aea/ after accepting the license." >&2
  exit 1
fi

mkdir -p "$RAW_DIR" "$OUT_DIR"

i=0
while IFS= read -r url && [[ $i -lt $N ]]; do
  [[ -z "$url" || "${url:0:1}" == "#" ]] && continue
  i=$((i + 1))
  fname="$(basename "$(echo "$url" | cut -d'?' -f1)")"
  dest="$RAW_DIR/$fname"
  if [[ -f "$dest" ]]; then
    echo "[skip] $fname (already present)"
  else
    echo "[get ] $fname"
    curl -L --fail --output "$dest" "$url"
  fi
done < "$ARIA_URLS_FILE"

echo "Preprocessing $i sequences..."
python scripts/preprocess_aria.py --vrs-dir "$RAW_DIR" --out "$OUT_DIR" --fps "$FPS"
echo "Done. Processed sequences in $OUT_DIR"
