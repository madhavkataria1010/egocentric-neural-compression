"""Download main_vrs files for the N smallest Aria sequences from the official JSON URL list.

Aria distributes per-user signed CDN URLs in a JSON like:
    AriaEverydayActivities_download_urls.json -> {"sequences": {<seq>: {<asset>: {download_url, ...}}}}

We pick the smallest sequences first to keep the download budget bounded.

Usage:
    python scripts/download_aria_json.py \\
        --json AriaEverydayActivities_download_urls.json \\
        --out data/aria_raw \\
        --num 15 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_download(url: str, dest: Path, expected_bytes: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length", expected_bytes or 0))
        downloaded = 0
        last_print = t0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print > 5 and total:
                pct = downloaded * 100.0 / total
                rate = downloaded / (now - t0) / 1e6
                print(f"  [{dest.name}] {pct:5.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s",
                      flush=True)
                last_print = now
    tmp.rename(dest)
    print(f"  [{dest.name}] done in {time.time()-t0:.0f}s  {dest.stat().st_size/1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Aria URL JSON")
    ap.add_argument("--out", required=True, help="Output dir for VRS files")
    ap.add_argument("--num", type=int, default=15, help="Number of sequences to download (smallest first)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    ap.add_argument("--asset", default="main_vrs", help="Asset key to fetch per sequence")
    ap.add_argument("--max-gb", type=float, default=20.0, help="Hard cap on total download budget")
    ap.add_argument("--verify-sha", action="store_true", help="Verify SHA1 after download (slow)")
    args = ap.parse_args()

    with open(args.json) as f:
        meta = json.load(f)
    seqs = meta["sequences"]

    # Gather candidates with the requested asset.
    candidates = []
    for seq_name, assets in seqs.items():
        if args.asset not in assets:
            continue
        a = assets[args.asset]
        candidates.append((seq_name, a["download_url"], a["filename"], a["file_size_bytes"], a.get("sha1sum")))
    candidates.sort(key=lambda c: c[3])  # smallest first

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    selected = []
    total_bytes = 0
    for c in candidates:
        if len(selected) >= args.num:
            break
        if (total_bytes + c[3]) / 1e9 > args.max_gb:
            continue
        selected.append(c)
        total_bytes += c[3]

    print(f"Will download {len(selected)} sequences, {total_bytes/1e9:.1f} GB total -> {out}")
    for c in selected:
        print(f"  {c[0]:<40} {c[3]/1e6:>7.0f} MB  {c[2]}")

    def fetch(c):
        seq_name, url, filename, size, sha = c
        dest = out / filename
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [skip] {filename} (already present)", flush=True)
            return
        try:
            stream_download(url, dest, expected_bytes=size)
            if args.verify_sha and sha:
                got = sha1_of(dest)
                if got != sha:
                    print(f"  [WARN] sha mismatch on {filename}: got {got[:10]}, want {sha[:10]}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {filename}: {e!r}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(fetch, selected))

    print(f"Done. Files in {out}")


if __name__ == "__main__":
    main()
