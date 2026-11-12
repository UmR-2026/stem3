#!/usr/bin/env python3
"""Download the two UVR ONNX checkpoints used by stem3 and verify SHA-256.

The weights are NOT stored in this repository. They originate from the
Ultimate Vocal Remover project (MIT + attribution) and are re-hosted as ONNX
exports with metadata in the k2-fsa/sherpa-onnx release archive.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models"

# (exact byte size, sha256) of the artifacts published in the release above
MODELS = {
    "UVR_MDXNET_Main.onnx": (
        66759516,
        "900fb63c1a2405fe860316ca03c831ce4c7a4129ff1fbc55f99aee4422b98592",
    ),
    "UVR_MDXNET_KARA_2.onnx": (
        52787030,
        "c05b3b7a3ab73362c85511dd0386ad173f28570e23ce1d8d3f87391396a85c08",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in MODELS.items():
        target = DEST / name
        if target.is_file():
            if target.stat().st_size == size and sha256(target) == expected:
                print(f"[ok]      {name}")
                continue
            print(f"[replace] {name} (size or checksum mismatch)")
            target.unlink()

        url = f"{BASE}/{name}"
        print(f"[get]     {url}", flush=True)
        part = target.with_suffix(".onnx.part")
        try:
            urllib.request.urlretrieve(url, part)
            if part.stat().st_size != size:
                raise RuntimeError(f"size mismatch for {name}")
            if sha256(part) != expected:
                raise RuntimeError(f"SHA-256 mismatch for {name}")
            part.replace(target)
            print(f"[done]    {name} ({size / 1024 / 1024:.1f} MiB)")
        finally:
            part.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
