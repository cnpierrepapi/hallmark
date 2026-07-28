"""Generate, score, approve and stamp a campaign.

    python scripts/run_campaign.py "single origin coffee subscription"
    python scripts/run_campaign.py "..." --no-approve      # stop at the gate
    python scripts/run_campaign.py "..." --approver "Ada"

Every attempt is written to the ledger, including the ones that lose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark import ledger  # noqa: E402
from hallmark.campaign import Brief, approve, generate  # noqa: E402
from hallmark.verify import verify  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")

    args = sys.argv[1:]
    approver = "Studio Lead"
    do_approve = True

    if "--no-approve" in args:
        do_approve = False
        args.remove("--no-approve")
    if "--approver" in args:
        i = args.index("--approver")
        approver = args[i + 1]
        del args[i : i + 2]

    product = args[0] if args else "single origin coffee subscription"
    brief = Brief(product=product, audience="office workers who care about coffee")

    outdir = ROOT / "out" / "campaign"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Brief: {product}\n")
    campaign = generate(brief, workdir=outdir)

    print("Plan")
    print(f"  image     {campaign.plan.image_prompt[:96]}")
    print(f"  video     {campaign.plan.video_prompt[:96]}")
    print(f"  voiceover {campaign.plan.voiceover}\n")

    print(f"Run {campaign.run_id}")
    print(f"Ledger {campaign.ledger_key}\n")

    print("Attempts")
    for c in campaign.candidates:
        mark = "KEEP  " if c.accepted else "reject"
        print(
            f"  {mark} {c.modality:6} score={c.evaluation.score:<5} "
            f"{c.latency_seconds:6.1f}s {c.size_bytes / 1024:8.0f}KB {c.model}"
        )
        for check in c.evaluation.checks:
            flag = "ok " if check.passed else "NO "
            print(f"           {flag}{check.name:16} {check.detail}")
        if c.reject_reason:
            print(f"           rejected: {c.reject_reason}")

    for failure in campaign.failures:
        print(f"  FAILED {failure['modality']:6} {failure['model']}: {failure['error'][:120]}")

    if not do_approve:
        print(f"\nStatus: {campaign.status}. Nothing stamped, nothing published.")
        return 0

    print(f"\nApproving as {approver!r}")
    campaign = approve(campaign, approver=approver, note="Cleared for paid social")

    print(f"Status   {campaign.status}")
    print(f"Approval {campaign.approval}")
    print(f"Manifest {campaign.manifest_uri}")
    print(f"Hash     {campaign.canonical_hash[:32]}...\n")

    for asset in campaign.assets:
        print(f"  {asset.modality:6} score={asset.score:<5} {asset.embed_mode:8} {asset.model}")
        print(f"         {asset.stamped_url[-72:]}")

    print("\nVerifying every published file offline")
    bad = 0
    for asset in campaign.assets:
        report = verify(asset.local_path)
        ok = report.verdict == "verified"
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {asset.modality:6} {report.verdict}")
        if report.reason:
            print(f"       {report.reason}")

    (outdir / "campaign.json").write_text(
        json.dumps(campaign.to_dict(), indent=2), encoding="utf-8"
    )

    print("\nLedger summary across all runs")
    for row in ledger.summary():
        print(
            f"  {row['modality']:6} {row['model']:30} "
            f"attempts={row['attempts']:<3} accepted={row['accepted']:<3} "
            f"rate={row['acceptance_rate']:<6} avg={row['avg_latency_seconds']}s"
        )

    if campaign.failures or bad or campaign.status != "approved":
        print("\nINCOMPLETE")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
