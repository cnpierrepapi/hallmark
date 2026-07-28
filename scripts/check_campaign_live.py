"""Verify a generated campaign's assets against the deployed service.

Every stamped asset carries a pointer, so this exercises the whole path the
public page uses: read the pointer, fetch the manifest from B2, bind it to the
hash in the file, and withhold the private prompt.

    python scripts/check_campaign_live.py [base_url]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "out" / "campaign"

MEDIA_TYPES = {".png": "image/png", ".mp4": "video/mp4", ".mp3": "audio/mpeg"}


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "https://hallmark-rust.vercel.app").rstrip("/")

    health = httpx.get(f"{base}/health", timeout=60.0).json()
    print(f"health: {health}")
    if health.get("storage") != "configured":
        print("Storage is not configured on the deployment; pointer records cannot resolve.")
        return 1

    stamped = sorted(CAMPAIGN.glob("*_stamped.*"))
    if not stamped:
        print("No stamped assets found. Run scripts/run_campaign.py first.")
        return 1

    failures = 0
    for path in stamped:
        media_type = MEDIA_TYPES.get(path.suffix, "application/octet-stream")
        started = time.monotonic()
        response = httpx.post(
            f"{base}/api/verify",
            files={"file": (path.name, path.read_bytes(), media_type)},
            timeout=180.0,
        )
        elapsed = time.monotonic() - started

        if response.status_code != 200:
            print(f"FAIL  {path.name}: HTTP {response.status_code} {response.text[:200]}")
            failures += 1
            continue

        data = response.json()
        ok = data["verdict"] == "verified"
        withheld = all(s["prompt_withheld"] for s in data.get("steps", []))
        if not ok or not withheld:
            failures += 1

        print(f"{'ok  ' if ok else 'FAIL'}  {path.name:26} {data['verdict']:9} {elapsed:5.1f}s")
        print(f"      record   {data['manifest_source'][:70]}")
        print(f"      prompts withheld: {withheld}")
        if data.get("reason"):
            print(f"      reason   {data['reason']}")

    print("\nPASS" if failures == 0 else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
