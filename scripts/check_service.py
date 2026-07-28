"""Exercise the running verify service against real generated assets.

Uses the files produced by scripts/smoke_provenance.py, so this checks the
service against genuine GMI output stored in B2 rather than synthetic
fixtures.

    python scripts/check_service.py [base_url]
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

CASES = [
    ("smoke_stamped.png", "verified"),
    ("smoke_tampered.png", "altered"),
    ("smoke_raw.png", "unsigned"),
]


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099").rstrip("/")

    health = httpx.get(f"{base}/health", timeout=30.0)
    print(f"health {health.status_code} {health.json()}")

    failures = 0
    for filename, expected in CASES:
        path = OUT / filename
        if not path.exists():
            print(f"SKIP  {filename} not found, run scripts/smoke_provenance.py first")
            continue

        response = httpx.post(
            f"{base}/api/verify",
            files={"file": (filename, path.read_bytes(), "image/png")},
            timeout=120.0,
        )
        if response.status_code != 200:
            print(f"FAIL  {filename}: HTTP {response.status_code} {response.text[:200]}")
            failures += 1
            continue

        data = response.json()
        got = data["verdict"]
        mark = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1

        print(f"{mark}  {filename:24} verdict={got:10} expected={expected}")
        print(f"      summary  {data['summary']}")
        if data.get("reason"):
            print(f"      reason   {data['reason']}")
        if data.get("steps"):
            step = data["steps"][0]
            print(f"      chain    {step['provider']} / {step['model']} / {step['modality']}")
        print(f"      now      {data['computed_sha256'] or '-'}")
        print(f"      recorded {data['declared_sha256'] or '-'}")
        print()

    print("PASS" if failures == 0 else f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
