"""Day 1 proof: generate, stamp, verify, tamper, catch.

Runs the smallest real pipeline that exercises the whole provenance loop
without touching storage, so the risky part (embedding and byte-level
verification) is proven before any of the app is built on top of it.

    python scripts/smoke_provenance.py

Costs one cheap image generation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genblaze_core import Manifest, Modality, Pipeline  # noqa: E402
from genblaze_core.media import get_handler  # noqa: E402
from genblaze_gmicloud import GMICloudImageProvider  # noqa: E402

from hallmark.integrity import verify_file  # noqa: E402

MODEL = "Z-Image-Turbo"
PROMPT = (
    "A ceramic coffee cup on a sunlit windowsill, steam rising, "
    "shallow depth of field, warm morning light"
)
OUT_DIR = ROOT / "out"


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("GMI_API_KEY"):
        print("GMI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    print(f"[1/6] Generating with {MODEL}")
    result = (
        Pipeline("hallmark-smoke")
        .step(
            GMICloudImageProvider(),
            model=MODEL,
            prompt=PROMPT,
            modality=Modality.IMAGE,
        )
        .run(timeout=300)
    )

    step = result.run.steps[0]
    print(f"      status   {step.status}")
    print(f"      cost     {step.cost_usd}")
    if not step.assets:
        print(f"      ERROR    no assets returned. error={step.error!r}")
        return 1

    asset = step.assets[0]
    print(f"      media    {asset.media_type}")
    print(f"      sha256   {asset.sha256}")
    print(f"      url      {str(asset.url)[:90]}")

    manifest: Manifest = result.manifest
    print(f"[2/6] Manifest hash {manifest.canonical_hash[:16]}...")
    print(f"      self-verify   {manifest.verify()}")

    suffix = ".png" if "png" in (asset.media_type or "") else ".jpg"
    raw_path = OUT_DIR / f"smoke_raw{suffix}"
    print(f"[3/6] Downloading to {raw_path.name}")
    _download(str(asset.url), raw_path)
    print(f"      bytes on disk {raw_path.stat().st_size}")

    stamped_path = OUT_DIR / f"smoke_stamped{suffix}"
    handler = get_handler(asset.media_type)
    if handler is None:
        print(f"      ERROR    no genblaze handler for {asset.media_type}")
        return 1
    print(f"[4/6] Stamping manifest into {stamped_path.name}")
    handler.embed(raw_path, manifest, stamped_path)
    print(f"      bytes after stamp {stamped_path.stat().st_size}")

    print("[5/6] Verifying the stamped file offline")
    good = verify_file(stamped_path)
    print(f"      manifest_ok {good.manifest_ok}")
    print(f"      bytes_ok    {good.bytes_ok}")
    print(f"      computed    {good.computed_sha256}")
    print(f"      declared    {good.declared_sha256}")
    print(f"      reason      {good.reason}")

    tampered_path = OUT_DIR / f"smoke_tampered{suffix}"
    data = bytearray(stamped_path.read_bytes())
    flip_at = int(len(data) * 0.75)
    data[flip_at] ^= 0xFF
    tampered_path.write_bytes(bytes(data))

    print("[6/6] Verifying a tampered copy (this must fail)")
    bad = verify_file(tampered_path)
    print(f"      manifest_ok {bad.manifest_ok}")
    print(f"      bytes_ok    {bad.bytes_ok}")
    print(f"      reason      {bad.reason}")

    if good.ok and not bad.ok:
        print("\nPASS  clean file verifies, tampered file is caught.")
        return 0

    print("\nFAIL  the provenance loop did not behave as required.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
