"""Verify every published gallery tile against the deployed checker.

This is the exact path a visitor takes when they click a tile: fetch the
stamped bytes from the site, post them back, and read the verdict.

    python scripts/check_gallery_live.py [base_url]
"""

from __future__ import annotations

import sys
import time

import httpx

DEFAULT = "https://hallmark-rust.vercel.app"


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")

    show = httpx.get(f"{base}/api/showcase", timeout=90.0).json()
    tiles = show.get("gallery", [])
    if not tiles:
        print("No gallery tiles published yet.")
        return 1

    print(f"{len(tiles)} tiles published\n")
    failures = 0

    for tile in tiles:
        slug = tile["slug"]
        started = time.monotonic()

        media = httpx.get(f"{base}/api/gallery/{slug}", timeout=180.0)
        if media.status_code != 200:
            print(f"FAIL  {slug:20} media HTTP {media.status_code}")
            failures += 1
            continue

        res = httpx.post(
            f"{base}/api/verify",
            files={"file": (f"{slug}.png", media.content, "image/png")},
            timeout=180.0,
        )
        data = res.json()
        elapsed = time.monotonic() - started

        ok = data.get("verdict") == "verified"
        withheld = all(s["prompt_withheld"] for s in data.get("steps", []))
        if not ok or not withheld:
            failures += 1

        print(
            f"{'ok  ' if ok else 'FAIL'}  {slug:20} {data.get('verdict'):9} "
            f"{len(media.content) / 1024:7.0f}KB {elapsed:5.1f}s  withheld={withheld}"
        )
        if data.get("reason"):
            print(f"      {data['reason']}")

    print("\nPASS" if failures == 0 else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
