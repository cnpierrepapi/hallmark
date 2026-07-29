"""Generate the showcase: the assets a campaign would actually ship.

Every tile on the homepage is a real generated asset, stamped and stored like
any campaign asset, so a visitor can drop any of them into the checker and get
a genuine answer. The visuals are the proof, not decoration.

Stills and clips both, because an ad set is both. The hero is a clip: a page
about moving pictures that only shows stills is arguing against itself.

    python scripts/generate_gallery.py            # everything
    python scripts/generate_gallery.py image      # only the stills
    python scripts/generate_gallery.py video      # only the clips
    python scripts/generate_gallery.py redeliver  # re-sign what is on disk

Costs one generation per tile. Clips are the expensive half, so they are a
separate pipeline run: a failure in one modality does not throw away the other.
The redeliver mode calls no model at all: it re-signs the renders already
approved, which is what a change to the delivery format should do.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genblaze_core import Modality, Pipeline  # noqa: E402
from genblaze_core.models.asset import Asset  # noqa: E402
from genblaze_core.models.enums import PromptVisibility, StepType  # noqa: E402
from genblaze_core.models.manifest import Manifest  # noqa: E402
from genblaze_core.models.policy import EmbedPolicy  # noqa: E402
from genblaze_core.models.run import Run  # noqa: E402
from genblaze_core.models.step import Step  # noqa: E402

from hallmark import approval, ledger, metadata, storage  # noqa: E402
from hallmark.approval import Approval  # noqa: E402
from hallmark.evaluate import evaluate  # noqa: E402
from hallmark.ledger import Attempt  # noqa: E402
from hallmark.providers import (  # noqa: E402
    IMAGE_MODEL,
    VIDEO_MODEL,
    image_provider,
    video_provider,
)
from hallmark.stamp import stamp  # noqa: E402

OUT = ROOT / "out" / "gallery"

APPROVER = "Studio Lead"
APPROVAL_NOTE = "Campaign set signed off for the public page"

# The wall is what a visitor downloads, so it has to honour the claim the page
# makes two sections further down: right click, properties, there it is. That
# only works in a container the operating system will read, which is JPEG and
# is not PNG. See hallmark/metadata.py for the measurement.
DELIVERY = {"image": (".jpg", "image/jpeg")}

VERIFY_BASE = os.environ.get("HALLMARK_VERIFY_BASE", "https://hallmark-rust.vercel.app")

# The set is deliberately unbranded. These are the shots a product campaign is
# built from, described by what is in frame rather than by whose product it is.
STILLS = [
    (
        "still-coldbrew-can",
        "Cold brew, wet slate",
        "Matte black cold brew coffee can standing on wet dark slate, heavy condensation "
        "beading down the aluminium, single hard key light from the left, deep shadow "
        "falloff, shallow depth of field, commercial product photography, 50mm, crisp "
        "specular highlights on the rim",
    ),
    (
        "still-runner-salt",
        "Runner, salt flat",
        "A performance running shoe on a cracked salt flat at low sun, fine dust suspended "
        "in the warm backlight, long shadow across the cracks, hero product angle three "
        "quarters, commercial sports photography, sharp detail in the knit upper",
    ),
    (
        "still-serum-drop",
        "Serum, droplets",
        "Frosted glass skincare serum bottle with a brushed gold dropper cap, water "
        "droplets across the glass, soft gradient backdrop from pale sand to deep ochre, "
        "clean beauty product photography, soft box light, subtle reflection on the base",
    ),
    (
        "still-headphones",
        "Headphones, gels",
        "Over ear headphones floating against a dark seamless background, lit with magenta "
        "and cyan colour gels, soft matte plastic and brushed aluminium finish, premium "
        "electronics product photography, controlled reflections, sharp edge definition",
    ),
    (
        "still-watch-macro",
        "Watch, macro",
        "Extreme macro of a steel dive watch bezel and crown, brushed and polished metal "
        "meeting, deep blue sunburst dial just in frame, single raking light, luxury watch "
        "photography, razor sharp focus stack, black background",
    ),
    (
        "still-perfume-caustics",
        "Perfume, caustics",
        "A faceted glass perfume bottle on a pale stone plinth, hard sunlight throwing "
        "caustic light patterns across the surface behind it, amber liquid catching the "
        "light, luxury fragrance advertising still, warm palette, clean negative space",
    ),
]

CLIPS = [
    (
        "clip-pour",
        "Pour, slow motion",
        "Slow motion cold brew coffee pouring over clear ice cubes into a heavy glass, "
        "dark studio background, single hard key light catching the falling liquid, "
        "condensation on the glass, commercial beverage advertising shot, locked off "
        "camera, shallow depth of field",
        True,  # the hero
    ),
    (
        "clip-can-turn",
        "Can, turntable",
        "A matte black drinks can rotating slowly on a turntable, heavy condensation "
        "beading on the aluminium, dark reflective surface below, controlled studio "
        "lighting sweeping across the label area, commercial product turntable shot",
        False,
    ),
    (
        "clip-fabric",
        "Fabric, motion",
        "Slow motion close up of premium athletic fabric rippling in a controlled breeze, "
        "warm sand and deep charcoal tones, raking light picking out the weave, luxury "
        "apparel advertising texture shot, macro, shallow depth of field",
        False,
    ),
]


def _latency(step) -> float:
    if step.started_at and step.completed_at:
        return (step.completed_at - step.started_at).total_seconds()
    return 0.0


def _run(kind: str, entries: list[tuple], modality: Modality, provider, model: str,
         suffix: str, mime: str) -> tuple[list[dict], list[Attempt]]:
    """Generate one modality, gate it, stamp it, and upload what passes."""
    print(f"\n{'=' * 60}\n{kind}: {len(entries)} assets with {model}\n{'=' * 60}")

    pipeline = Pipeline(f"hallmark-showcase-{kind}")
    for entry in entries:
        pipeline = pipeline.step(
            provider,
            model=model,
            prompt=entry[2],
            modality=modality,
            prompt_visibility=PromptVisibility.PRIVATE,
        )

    started = time.monotonic()
    result = pipeline.run(sink=storage.sink(), timeout=2400, raise_on_failure=False)
    print(f"Pipeline finished in {time.monotonic() - started:.0f}s")

    out_suffix, out_mime = DELIVERY.get(kind, (suffix, mime))
    signed = out_mime != mime

    attempts: list[Attempt] = []
    passed: list[dict] = []

    for entry, step in zip(entries, result.run.steps):
        slug, title = entry[0], entry[1]
        hero = entry[3] if len(entry) > 3 else False

        if not step.assets or step.status.value != "succeeded":
            print(f"  FAILED {slug}: {step.error}")
            continue

        asset = step.assets[0]
        raw = OUT / f"{slug}_raw{suffix}"
        storage.download(str(asset.url), raw)

        check = evaluate(raw, kind)
        latency = _latency(step)

        attempts.append(
            Attempt(
                run_id=result.run.run_id,
                campaign="showcase",
                modality=kind,
                model=step.model,
                provider="gmicloud",
                accepted=check.passed,
                score=check.score,
                latency_seconds=latency,
                reject_reason=check.reason,
                cost_usd=step.cost_usd,
                sha256=asset.sha256,
                size_bytes=raw.stat().st_size,
                media_type=asset.media_type,
                checks=json.dumps(check.to_dict()["checks"]),
            )
        )

        if not check.passed:
            print(f"  reject {slug}: {check.reason}")
            continue

        passed.append(
            _deliver(slug, title, hero, raw, step.model, entry[2], latency,
                     out_suffix, out_mime, signed)
        )

    if not passed:
        return [], attempts

    return _publish(kind, modality, passed, out_suffix, out_mime), attempts


def _deliver(slug: str, title: str, hero: bool, raw: Path, model: str, prompt: str,
             latency: float, out_suffix: str, out_mime: str, signed: bool) -> dict:
    """Produce the file a visitor downloads, and hash that.

    For stills the visible credit is written first, so editing the credit back
    out breaks verification exactly like editing the picture does. It also means
    the pipeline's own manifest cannot describe this file: the sink hashed the
    PNG that came back from the model, not the JPEG we ship.
    """
    delivered = OUT / f"{slug}_delivery{out_suffix}"
    if signed:
        metadata.to_jpeg(
            raw,
            delivered,
            metadata.Signature(
                approver=APPROVER,
                model=model,
                note=APPROVAL_NOTE,
                brief=title,
                verify_url=VERIFY_BASE,
            ),
        )
    else:
        delivered.write_bytes(raw.read_bytes())

    body = delivered.read_bytes()
    return {
        "slug": slug,
        "title": title,
        "hero": hero,
        "model": model,
        "prompt": prompt,
        "delivered": delivered,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
        "latency": latency,
    }


def _publish(kind: str, modality: Modality, passed: list[dict],
             out_suffix: str, out_mime: str) -> list[dict]:
    """Sign off the delivered set, stamp each file, and upload it."""
    # One record over the whole set, built so every hash in it belongs to a file
    # a visitor can actually download. Prompts stay private throughout.
    manifest = Manifest.from_run(
        Run(
            name=f"hallmark-showcase-{kind}",
            steps=[
                Step(
                    step_type=StepType.GENERATE,
                    modality=modality,
                    provider="gmicloud",
                    model=p["model"],
                    prompt=p["prompt"],
                    prompt_visibility=PromptVisibility.PRIVATE,
                    assets=[
                        Asset(
                            url=f"s3://{storage.bucket()}/showcase/gallery/"
                                f"{p['slug']}{out_suffix}",
                            media_type=out_mime,
                            sha256=p["sha256"],
                            size_bytes=p["size_bytes"],
                        )
                    ],
                    metadata={"slug": p["slug"], "title": p["title"]},
                )
                for p in passed
            ],
        )
    )

    # Same gate as a campaign: sign off, publish the record, then stamp.
    approval.apply(manifest, Approval(approver=APPROVER, decision="approved",
                                      note=APPROVAL_NOTE))
    manifest_uri = approval.publish_manifest(manifest, manifest.run.run_id)
    print(f"Record published to {manifest_uri}\n")

    tiles: list[dict] = []
    for p in passed:
        stamped = OUT / f"{p['slug']}{out_suffix}"
        stamp(
            p["delivered"],
            manifest,
            stamped,
            policy=EmbedPolicy(embed_mode="pointer", prompt_visibility=PromptVisibility.PRIVATE),
            mime_type=out_mime,
        )

        key = f"showcase/gallery/{p['slug']}{out_suffix}"
        storage.upload(stamped, key, out_mime)

        tiles.append(
            {
                "slug": p["slug"],
                "title": p["title"],
                "key": key,
                "kind": kind,
                "media_type": out_mime,
                "hero": p["hero"],
                "model": p["model"],
                "manifest_uri": manifest_uri,
                "sha256": p["sha256"],
                "size_bytes": stamped.stat().st_size,
                "latency_seconds": round(p["latency"], 1),
            }
        )
        print(f"  ok     {p['slug']:24} {p['latency']:6.1f}s  "
              f"{stamped.stat().st_size / 1024:7.0f}KB  {out_mime}")

    return tiles


def _redeliver(kind: str, entries: list[tuple], modality: Modality,
               suffix: str, mime: str, previous: list[dict]) -> list[dict]:
    """Re-deliver an approved set from the renders already on disk.

    Changing the delivery format should not mean six new pictures nobody chose.
    The point of the gate is that a person approved these particular frames, so
    a format change re-delivers those frames rather than rolling the dice again.
    Costs nothing and cannot come back worse.
    """
    out_suffix, out_mime = DELIVERY.get(kind, (suffix, mime))
    signed = out_mime != mime
    was = {t["slug"]: t for t in previous}

    passed: list[dict] = []
    for entry in entries:
        slug, title = entry[0], entry[1]
        hero = entry[3] if len(entry) > 3 else False
        raw = OUT / f"{slug}_raw{suffix}"
        if not raw.exists():
            print(f"  skip   {slug}: no render at {raw.name}")
            continue

        before = was.get(slug, {})
        passed.append(
            _deliver(slug, title, hero, raw, before.get("model", "unknown"), entry[2],
                     float(before.get("latency_seconds") or 0.0),
                     out_suffix, out_mime, signed)
        )

    if not passed:
        return []

    print(f"\n{'=' * 60}\n{kind}: re-delivering {len(passed)} approved assets"
          f"\n{'=' * 60}")
    return _publish(kind, modality, passed, out_suffix, out_mime)


def main() -> int:
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)

    wanted = sys.argv[1] if len(sys.argv) > 1 else "all"
    if wanted not in ("all", "image", "video", "redeliver"):
        print("Usage: generate_gallery.py [all|image|video|redeliver]")
        return 2

    # Keep whatever the other modality published last time, so a rerun of one
    # half does not wipe the other off the page.
    record = {"tiles": []}
    existing = OUT / "gallery.json"
    if existing.exists():
        record = json.loads(existing.read_text(encoding="utf-8"))

    previous = record.get("tiles", [])
    kept = [t for t in previous if wanted not in ("all", "redeliver") and t.get("kind") != wanted]

    tiles: list[dict] = []
    attempts: list[Attempt] = []

    if wanted == "redeliver":
        # No model is called and no attempt is made, so nothing goes on the
        # ledger: these are the same assets, delivered in a different container.
        tiles += _redeliver("image", STILLS, Modality.IMAGE, ".png", "image/png", previous)
        tiles += _redeliver("video", CLIPS, Modality.VIDEO, ".mp4", "video/mp4", previous)
        if not tiles:
            print("Nothing to re-deliver: no renders found in out/gallery.")
            return 1

    if wanted in ("all", "image"):
        got, tried = _run("image", STILLS, Modality.IMAGE, image_provider(), IMAGE_MODEL,
                          ".png", "image/png")
        tiles += got
        attempts += tried

    if wanted in ("all", "video"):
        got, tried = _run("video", CLIPS, Modality.VIDEO, video_provider(), VIDEO_MODEL,
                          ".mp4", "video/mp4")
        tiles += got
        attempts += tried

    ledger_key = ledger.write(attempts) if attempts else None

    # Clips first: the wall opens on movement.
    merged = kept + tiles
    merged.sort(key=lambda t: (not t.get("hero"), t.get("kind") != "video", t["slug"]))

    (OUT / "gallery.json").write_text(
        json.dumps({"tiles": merged}, indent=2), encoding="utf-8"
    )

    stills = sum(1 for t in merged if t.get("kind") == "image")
    clips = sum(1 for t in merged if t.get("kind") == "video")
    print(f"\n{len(merged)} tiles on the wall: {stills} stills, {clips} clips")
    print(f"Ledger {ledger_key}")
    print(f"Record written to {OUT / 'gallery.json'}")
    print("\nNext: python scripts/publish_showcase.py")
    return 0 if tiles else 1


if __name__ == "__main__":
    raise SystemExit(main())
