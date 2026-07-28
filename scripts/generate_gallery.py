"""Generate the homepage gallery.

Every tile on the page is a real generated asset, stamped and stored like any
campaign asset, so a visitor can drop any of them into the checker and get a
genuine answer. The visuals are the proof, not decoration.

    python scripts/generate_gallery.py

Costs one image generation per tile.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genblaze_core import Modality, Pipeline  # noqa: E402
from genblaze_core.models.enums import PromptVisibility  # noqa: E402
from genblaze_core.models.policy import EmbedPolicy  # noqa: E402

from hallmark import approval, ledger, storage  # noqa: E402
from hallmark.approval import Approval  # noqa: E402
from hallmark.evaluate import evaluate  # noqa: E402
from hallmark.ledger import Attempt  # noqa: E402
from hallmark.providers import IMAGE_MODEL, image_provider  # noqa: E402
from hallmark.stamp import stamp  # noqa: E402

OUT = ROOT / "out" / "gallery"

TILES = [
    (
        "slime-lime",
        "Slime, lime",
        "A glossy translucent slime creature with huge glossy eyes and a wobbling body, "
        "vivid acid lime green, thick subsurface scattering, wet specular highlights, "
        "3D character render, soft studio rim light, deep charcoal background",
    ),
    (
        "slime-magenta",
        "Slime, magenta",
        "A cute gooey slime blob character in translucent magenta and hot pink, "
        "dripping viscous strands, iridescent sheen, big cartoon eyes, "
        "3D render, octane style lighting, dark background",
    ),
    (
        "horse-iridescent",
        "Horse, iridescent",
        "A stylised horse sculpture with an iridescent oil slick coat shifting between "
        "teal, violet and gold, glossy ceramic finish, mid gallop, "
        "3D render, dramatic studio lighting, dark seamless backdrop",
    ),
    (
        "horse-candy",
        "Horse, candy",
        "A toy horse figurine in bright candy colours, glossy plastic finish, "
        "coral pink body with electric blue mane, rainbow tail, "
        "3D product render, clean studio light, dark background",
    ),
    (
        "creature-tentacle",
        "Creature, tentacle",
        "A friendly rubbery creature with soft tentacle limbs and translucent skin, "
        "warm amber and deep teal, glistening wet surface, big curious eyes, "
        "3D character render, cinematic rim lighting, dark background",
    ),
    (
        "horse-chrome",
        "Horse, chrome",
        "A liquid chrome horse head sculpture, mirror polished metal reflecting "
        "coloured studio gels in magenta and cyan, sharp reflections, "
        "3D render, high contrast, black background",
    ),
    (
        "slime-cluster",
        "Slime, cluster",
        "A cluster of small translucent jelly characters in assorted neon colours, "
        "squishing together, glossy wet surfaces, playful expressions, "
        "3D render, soft top light, dark background",
    ),
    (
        "horse-holo",
        "Horse, holo",
        "A galloping horse rendered in holographic foil, prismatic rainbow refraction, "
        "soft motion in the mane, glossy translucent material, "
        "3D render, studio lighting, deep dark background",
    ),
]


def main() -> int:
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(TILES)} tiles with {IMAGE_MODEL}\n")

    pipeline = Pipeline("hallmark-gallery")
    for _, _, prompt in TILES:
        pipeline = pipeline.step(
            image_provider(),
            model=IMAGE_MODEL,
            prompt=prompt,
            modality=Modality.IMAGE,
            prompt_visibility=PromptVisibility.PRIVATE,
        )

    started = time.monotonic()
    result = pipeline.run(sink=storage.sink(), timeout=1800, raise_on_failure=False)
    print(f"Pipeline finished in {time.monotonic() - started:.0f}s\n")

    manifest = result.manifest

    # Same gate as a campaign: sign off, republish the record, then stamp.
    approval.apply(manifest, Approval(approver="Studio Lead", decision="approved",
                                      note="Gallery specimens for the public page"))
    manifest_uri = approval.publish_manifest(manifest, result.run.run_id)
    print(f"Record published to {manifest_uri}\n")

    tiles = []
    attempts = []

    for (slug, title, _), step in zip(TILES, result.run.steps):
        if not step.assets or step.status.value != "succeeded":
            print(f"  FAILED {slug}: {step.error}")
            continue

        asset = step.assets[0]
        raw = OUT / f"{slug}_raw.png"
        storage.download(str(asset.url), raw)

        check = evaluate(raw, "image")
        latency = (
            (step.completed_at - step.started_at).total_seconds()
            if step.started_at and step.completed_at
            else 0.0
        )

        attempts.append(
            Attempt(
                run_id=result.run.run_id,
                campaign="gallery",
                modality="image",
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

        stamped = OUT / f"{slug}.png"
        stamp(
            raw,
            manifest,
            stamped,
            policy=EmbedPolicy(embed_mode="pointer", prompt_visibility=PromptVisibility.PRIVATE),
            mime_type="image/png",
        )

        key = f"showcase/gallery/{slug}.png"
        storage.upload(stamped, key, "image/png")

        tiles.append(
            {
                "slug": slug,
                "title": title,
                "key": key,
                "model": step.model,
                "sha256": asset.sha256,
                "size_bytes": stamped.stat().st_size,
                "latency_seconds": round(latency, 1),
            }
        )
        print(f"  ok     {slug:20} {latency:6.1f}s  {stamped.stat().st_size / 1024:7.0f}KB")

    ledger_key = ledger.write(attempts)

    (OUT / "gallery.json").write_text(
        json.dumps({"run_id": result.run.run_id, "manifest_uri": manifest_uri, "tiles": tiles},
                   indent=2),
        encoding="utf-8",
    )

    print(f"\n{len(tiles)} tiles published, ledger {ledger_key}")
    print(f"Record written to {OUT / 'gallery.json'}")
    return 0 if tiles else 1


if __name__ == "__main__":
    raise SystemExit(main())
