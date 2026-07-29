"""Publish the public showcase payload for the homepage.

Runs locally, not in the web app. It reads the campaign record and the Parquet
ledger, strips anything private, and writes one small JSON object to B2. The
deployed service then only has to read that file, so pyarrow never needs to
ship into a serverless function.

The redaction here is not cosmetic. The campaign record holds the prompts, and
the product's claim is that prompts stay private, so the public payload is
built by naming the fields that may appear rather than by deleting the ones
that may not.

    python scripts/publish_showcase.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark import ledger, storage  # noqa: E402

SHOWCASE_KEY = "showcase/current.json"
CAMPAIGN_JSON = ROOT / "out" / "campaign" / "campaign.json"

SPECIMEN_FILES = {
    "image": ("image_stamped.png", "image/png"),
    "video": ("video_stamped.mp4", "video/mp4"),
    "audio": ("audio_stamped.mp3", "audio/mpeg"),
}


THUMB_PX = 620


def _poster_frame(source: Path) -> bytes | None:
    """Pull the first frame out of a clip, as PNG bytes.

    A wall of clips that all start loading on first paint would be several
    megabytes before a visitor has done anything, so the tiles show a poster
    and only fetch the clip when it is actually played. Extraction runs here,
    locally, with an ffmpeg binary that never goes anywhere near the deployed
    function.
    """
    import subprocess

    try:
        import imageio_ffmpeg
    except ImportError:
        print("    imageio-ffmpeg is not installed, so clips get no poster frame")
        return None

    done = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v", "error",
            "-i", str(source),
            "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
    )
    if done.returncode != 0 or not done.stdout:
        print(f"    could not read a frame from {source.name}: {done.stderr[:160]!r}")
        return None
    return done.stdout


DISPLAY_HEIGHT = 540
DISPLAY_CRF = "31"


def _publish_display_clip(slug: str) -> str | None:
    """Publish a web-weight copy of a clip, for playing only.

    The generated clips come back 1080p30 at 11 to 29 Mbps, which is 7 to 18MB
    for five seconds. A wall of those is not a web page. This is the same
    bargain the thumbnails already make: the tile plays a display copy, and
    clicking it still fetches the full stamped file, because a re-encoded copy
    is not the asset and would rightly fail its own check.

    Audio is dropped. These are silent product clips and the track is a
    128kbps stereo AAC of nothing.
    """
    import subprocess
    import tempfile

    try:
        import imageio_ffmpeg
    except ImportError:
        print("    imageio-ffmpeg is not installed, so clips get no display copy")
        return None

    source = ROOT / "out" / "gallery" / f"{slug}.mp4"
    if not source.exists():
        return None

    with tempfile.TemporaryDirectory(prefix="hallmark-display-") as tmp:
        out = Path(tmp) / f"{slug}.mp4"
        done = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v", "error", "-y",
                "-i", str(source),
                "-vf", f"scale=-2:{DISPLAY_HEIGHT}",
                "-c:v", "libx264", "-crf", DISPLAY_CRF, "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-an",
                "-movflags", "+faststart",
                str(out),
            ],
            capture_output=True,
        )
        if done.returncode != 0 or not out.exists():
            print(f"    could not transcode {slug}: {done.stderr[:200]!r}")
            return None

        body = out.read_bytes()
        shrink = source.stat().st_size / len(body)
        print(f"    display   {slug:24} {len(body) / 1024:7.0f}KB  {shrink:4.1f}x smaller")

    key = f"showcase/display/{slug}.mp4"
    storage.client().put_object(
        Bucket=storage.bucket(), Key=key, Body=body, ContentType="video/mp4"
    )
    return key


def _publish_thumb(slug: str, kind: str = "image") -> str | None:
    """Publish a small WebP for display only.

    The tiles are 1024px assets of over a megabyte each, which is far too much
    to load a wall of on first paint. The thumbnail is for looking at; clicking
    a tile still fetches the full stamped file, so what gets verified is the
    real asset and not a resized copy that would fail its own check.

    For a clip this is the poster frame, and it does the same job twice: it is
    what the tile shows before playback, and what it falls back to if the
    visitor never hovers.
    """
    from io import BytesIO

    from PIL import Image

    folder = ROOT / "out" / "gallery"
    if kind == "video":
        source = folder / f"{slug}.mp4"
        if not source.exists():
            return None
        frame = _poster_frame(source)
        if frame is None:
            return None
        handle = BytesIO(frame)
    else:
        # Stills are delivered as JPEG so the operating system will show the
        # signature. Older sets on disk are PNG, so both are accepted here.
        source = next(
            (folder / f"{slug}{ext}" for ext in (".jpg", ".png")
             if (folder / f"{slug}{ext}").exists()),
            None,
        )
        if source is None:
            return None
        handle = source

    with Image.open(handle) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, "WEBP", quality=82, method=5)

    key = f"showcase/thumbs/{slug}.webp"
    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=key,
        Body=buffer.getvalue(),
        ContentType="image/webp",
    )
    return key


def main() -> int:
    load_dotenv(ROOT / ".env")

    if not CAMPAIGN_JSON.exists():
        print(f"No campaign record at {CAMPAIGN_JSON}. Run scripts/run_campaign.py first.")
        return 1

    campaign = json.loads(CAMPAIGN_JSON.read_text(encoding="utf-8"))
    run_id = campaign["run_id"]

    if campaign.get("status") != "approved":
        print(f"Campaign {run_id} is {campaign.get('status')}, not approved. Nothing to publish.")
        return 1

    # Upload the stamped specimens under a stable showcase prefix so the page
    # keeps working when a later campaign is generated.
    specimens = []
    for modality, (filename, media_type) in SPECIMEN_FILES.items():
        local = ROOT / "out" / "campaign" / filename
        if not local.exists():
            print(f"  skip {modality}: {filename} not found")
            continue

        key = f"showcase/{modality}{local.suffix}"
        storage.upload(local, key, media_type)

        asset = next((a for a in campaign["assets"] if a["modality"] == modality), None)
        specimens.append(
            {
                "modality": modality,
                "key": key,
                "media_type": media_type,
                "size_bytes": local.stat().st_size,
                "model": asset["model"] if asset else None,
                "sha256": asset["sha256"] if asset else None,
                "score": asset["score"] if asset else None,
            }
        )
        print(f"  published {modality:6} {key}")

    # Attempts, with the reject reasons kept because they are the interesting
    # part, and nothing that reveals a prompt.
    attempts = [
        {
            "modality": c["modality"],
            "model": c["model"],
            "accepted": c["accepted"],
            "score": c["score"],
            "passed": c["passed"],
            "reject_reason": c["reject_reason"],
            "latency_seconds": c["latency_seconds"],
            "size_bytes": c["size_bytes"],
            "sha256": c["sha256"],
            "checks": [
                {"name": k["name"], "passed": k["passed"], "detail": k["detail"]}
                for k in c["checks"]
            ],
        }
        for c in campaign["candidates"]
    ]

    # Gallery tiles are real stamped assets too, so a visitor can drop any of
    # them into the checker. Published if a gallery run exists.
    gallery = []
    gallery_json = ROOT / "out" / "gallery" / "gallery.json"
    if gallery_json.exists():
        record = json.loads(gallery_json.read_text(encoding="utf-8"))
        for t in record.get("tiles", []):
            kind = t.get("kind", "image")
            thumb_key = _publish_thumb(t["slug"], kind)
            display_key = _publish_display_clip(t["slug"]) if kind == "video" else None
            gallery.append(
                {
                    "slug": t["slug"],
                    "title": t["title"],
                    "key": t["key"],
                    "kind": kind,
                    "media_type": t.get("media_type", "image/png"),
                    "hero": bool(t.get("hero")),
                    "thumb_key": thumb_key,
                    "display_key": display_key,
                    "model": t["model"],
                    "sha256": t["sha256"],
                    "size_bytes": t["size_bytes"],
                    "latency_seconds": t["latency_seconds"],
                }
            )
        stills = sum(1 for t in gallery if t["kind"] == "image")
        clips = sum(1 for t in gallery if t["kind"] == "video")
        print(f"  gallery   {len(gallery)} tiles: {stills} stills, {clips} clips")

    # Attempts made on the live page arrive as JSON because the function has no
    # Parquet writer. Fold them in first, or the summary below reports only the
    # runs started from this machine.
    drained = ledger.drain_pending()
    if drained:
        print(f"  ledger    folded in {drained} attempts from the live demo")

    payload = {
        "run_id": run_id,
        "gallery": gallery,
        "product": campaign["brief"]["product"],
        "audience": campaign["brief"]["audience"],
        "canonical_hash": campaign["canonical_hash"],
        "manifest_uri": campaign["manifest_uri"],
        "approval": campaign["approval"],
        "specimens": specimens,
        "attempts": attempts,
        "ledger": ledger.summary(),
    }

    leaked = [p for p in (campaign["plan"] or {}).values() if p and p in json.dumps(payload)]
    if leaked:
        print("REFUSING to publish: a prompt appears in the public payload.")
        return 1

    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=SHOWCASE_KEY,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"\nShowcase written to {SHOWCASE_KEY}")
    print(f"  run       {run_id}")
    print(f"  approver  {(payload['approval'] or {}).get('approver')}")
    print(f"  specimens {len(specimens)}")
    print(f"  attempts  {len(attempts)}")
    print(f"  models    {len(payload['ledger'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
