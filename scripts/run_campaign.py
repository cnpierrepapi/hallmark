"""Generate a full stamped campaign from a brief.

    python scripts/run_campaign.py "single origin coffee subscription"

Costs one image, one short video and one voiceover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark.campaign import Brief, run_campaign  # noqa: E402
from hallmark.verify import verify  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")

    product = sys.argv[1] if len(sys.argv) > 1 else "single origin coffee subscription"
    brief = Brief(product=product, audience="office workers who care about coffee")

    outdir = ROOT / "out" / "campaign"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Brief: {product}\n")
    result = run_campaign(brief, workdir=outdir)

    print("Plan")
    print(f"  image     {result.plan.image_prompt[:100]}")
    print(f"  video     {result.plan.video_prompt[:100]}")
    print(f"  voiceover {result.plan.voiceover}\n")

    print(f"Run       {result.run_id}")
    print(f"Manifest  {result.manifest_uri}")
    print(f"Hash      {result.canonical_hash[:32]}...\n")

    for asset in result.assets:
        print(f"  {asset.modality:6} {asset.model:28} {asset.embed_mode:8} {asset.media_type}")
        print(f"         stored  {asset.stored_url[-70:]}")
        print(f"         stamped {asset.stamped_url[-70:]}")

    for failure in result.failures:
        print(f"  FAILED {failure['modality']:6} {failure['model']}: {failure['error'][:140]}")

    print("\nVerifying every stamped file offline")
    bad = 0
    for asset in result.assets:
        report = verify(asset.local_path)
        mark = "ok  " if report.verdict == "verified" else "FAIL"
        if report.verdict != "verified":
            bad += 1
        print(f"  {mark} {asset.modality:6} {report.verdict:9} source={report.manifest_source[:48]}")
        if report.reason:
            print(f"       {report.reason}")
        if report.steps:
            withheld = all(s.prompt_withheld for s in report.steps)
            print(f"       prompts withheld: {withheld}")

    (outdir / "campaign.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nRecord written to {outdir / 'campaign.json'}")

    if result.failures or bad:
        print("\nINCOMPLETE")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
