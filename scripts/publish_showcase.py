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
        gallery = [
            {
                "slug": t["slug"],
                "title": t["title"],
                "key": t["key"],
                "model": t["model"],
                "sha256": t["sha256"],
                "size_bytes": t["size_bytes"],
                "latency_seconds": t["latency_seconds"],
            }
            for t in record.get("tiles", [])
        ]
        print(f"  gallery   {len(gallery)} tiles")

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
