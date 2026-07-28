"""Verify every published showcase tile against the deployed checker.

This is the exact path a visitor takes when they click a tile: fetch the
stamped bytes from the site, post them back, and read the verdict. Stills and
clips both, because the wall carries both.

It also checks the two display paths, which is where a page like this usually
starts lying. The poster and the preview clip are re-encoded copies, small
enough to put on a wall; the asset is the full stamped file. So the display
copy of a clip is posted too, and it MUST come back unsigned. If a resized
copy ever verified, the check would be measuring the wrong bytes.

    python scripts/check_gallery_live.py [base_url]
"""

from __future__ import annotations

import sys
import time

import httpx

DEFAULT = "https://hallmark-rust.vercel.app"


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    client = httpx.Client(timeout=240.0)

    show = client.get(f"{base}/api/showcase").json()
    tiles = show.get("gallery", [])
    if not tiles:
        print("No showcase tiles published yet.")
        return 1

    stills = sum(1 for t in tiles if t.get("kind", "image") == "image")
    clips = sum(1 for t in tiles if t.get("kind") == "video")
    hero = next((t for t in tiles if t.get("hero")), tiles[0])
    print(f"{len(tiles)} tiles: {stills} stills, {clips} clips. Hero is {hero['slug']}.\n")

    failures = 0
    first_paint = 0

    for tile in tiles:
        slug = tile["slug"]
        kind = tile.get("kind", "image")
        suffix = ".mp4" if kind == "video" else ".png"
        mime = tile.get("media_type", "image/png")
        started = time.monotonic()

        thumb = client.get(f"{base}/api/thumb/{slug}")
        if thumb.status_code != 200:
            print(f"FAIL  {slug:24} poster HTTP {thumb.status_code}")
            failures += 1
            continue
        first_paint += len(thumb.content)

        media = client.get(f"{base}/api/gallery/{slug}")
        if media.status_code != 200:
            print(f"FAIL  {slug:24} media HTTP {media.status_code}")
            failures += 1
            continue

        data = client.post(
            f"{base}/api/verify", files={"file": (f"{slug}{suffix}", media.content, mime)}
        ).json()
        elapsed = time.monotonic() - started

        ok = data.get("verdict") == "verified"
        withheld = all(s["prompt_withheld"] for s in data.get("steps", []))
        signed = (data.get("approval") or {}).get("approver")
        if not ok or not withheld or not signed:
            failures += 1

        display = ""
        if kind == "video":
            clip = client.get(f"{base}/api/clip/{slug}")
            if clip.status_code != 200:
                print(f"FAIL  {slug:24} display clip HTTP {clip.status_code}")
                failures += 1
                continue
            display = f" display {len(clip.content) / 1024:5.0f}KB"

        print(
            f"{'ok  ' if ok else 'FAIL'}  {slug:24} {data.get('verdict'):9} "
            f"{len(media.content) / 1024 / 1024:5.1f}MB {elapsed:5.1f}s  "
            f"withheld={withheld} signed by {signed}{display}"
        )
        if data.get("reason"):
            print(f"      {data['reason']}")

    # A display copy is not the asset, and must never pass as one.
    for tile in [t for t in tiles if t.get("kind") == "video"][:1]:
        slug = tile["slug"]
        copy = client.get(f"{base}/api/clip/{slug}").content
        data = client.post(
            f"{base}/api/verify", files={"file": (f"{slug}.mp4", copy, "video/mp4")}
        ).json()
        passed = data.get("verdict") != "verified"
        if not passed:
            failures += 1
        print(
            f"\n{'ok  ' if passed else 'FAIL'}  the display copy of {slug} reports "
            f"{data.get('verdict')}, as it should: it is not the asset"
        )

    print(f"\nfirst paint, every poster: {first_paint / 1024:.0f}KB")
    print("\nPASS" if failures == 0 else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
