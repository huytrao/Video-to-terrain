"""Put Qwen3-VL-8B onto a Kaggle dataset, so the RUN kernel can attach it.

Runs on the BUILD kernel — the one with internet and working Kaggle credentials.

Why a dataset and not a relay. The relay path pushes chunks through the
receiving notebook's own working directory, and RUN has 9.2 GB free against a
17.55 GB model: the transfer would fill the disk somewhere in the middle and
fail with an error about space rather than about the model. A dataset sidesteps
the working directory entirely — Kaggle mounts it read-only at
/kaggle/input/datasets, which costs RUN nothing.

The trade is that attaching a dataset is a UI action and it restarts the
session, so this should finish *before* anyone attaches, and attaching should
wait until the stages that do not need a VLM have finished.

The model is the Holi-Spatial fine-tune, not stock Qwen. That matters: the
vocabulary the paper reports comes from this checkpoint's behaviour on the
paper's prompt, and a stock checkpoint answers differently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.environ.get("V3D_QWEN_REPO", "Holi-Spatial/HoliSpatial-2M-QA-Qwen3-VL-8B")
DEST = os.environ.get("V3D_QWEN_DIR", "/kaggle/working/qwen3vl8b")
SLUG = os.environ.get("V3D_QWEN_SLUG", "v3d-qwen3vl8b-holispatial")


def sh(cmd: list[str], what: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(f"  [{time.time() - t:6.1f}s] {what} rc={r.returncode}", flush=True)
    if r.returncode != 0:
        print((r.stderr or r.stdout).strip()[-800:], flush=True)
    return r


def main() -> int:
    os.makedirs(DEST, exist_ok=True)

    print(f"downloading {REPO} -> {DEST}", flush=True)
    r = sh([sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download", REPO, "--local-dir", DEST], "hf download")
    if r.returncode != 0:
        # The CLI moves around between versions; the python API does not.
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=REPO, local_dir=DEST,
                          token=os.environ.get("HF_TOKEN") or None)
        print("  fell back to snapshot_download", flush=True)

    files = [f for f in os.listdir(DEST) if not f.startswith(".")]
    total = sum(os.path.getsize(os.path.join(DEST, f))
                for f in files if os.path.isfile(os.path.join(DEST, f)))
    print(f"{len(files)} entries, {total / 1e9:.2f} GB", flush=True)
    if total < 5e9:
        raise SystemExit(f"only {total / 1e9:.2f} GB downloaded; refusing to publish a partial model")

    user = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"] \
        if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) else os.environ["KAGGLE_USERNAME"]
    meta = {"title": "v3d Qwen3-VL-8B (Holi-Spatial fine-tune)",
            "id": f"{user}/{SLUG}",
            "licenses": [{"name": "other"}]}
    with open(os.path.join(DEST, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    r = sh(["kaggle", "datasets", "create", "-p", DEST, "-r", "zip", "--dir-mode", "zip"],
           "kaggle datasets create")
    if r.returncode != 0:
        sh(["kaggle", "datasets", "version", "-p", DEST, "-m", "refresh", "--dir-mode", "zip"],
           "kaggle datasets version")

    print(f"ATTACH THIS ON RUN: {meta['id']}", flush=True)
    print("QWEN_DATASET_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
