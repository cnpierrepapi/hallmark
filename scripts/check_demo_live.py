"""Walk the whole demo against the deployed site, as a visitor would.

Generate, poll, pick, prove, edit, disprove, then read the inventory back.
Costs one generation of seven images.

    python scripts/check_demo_live.py [base_url]
"""

from __future__ import annotations

import sys
import time

import httpx

DEFAULT = "https://hallmark-rust.vercel.app"
BRIEF = "a slimy creature made of lime jelly"


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    client = httpx.Client(timeout=180.0)

    print(f"quota: {client.get(f'{base}/api/demo/quota').json()}")

    print("\n[1] generate")
    res = client.post(f"{base}/api/demo/generate", json={"brief": BRIEF})
    if res.status_code == 429:
        print(f"    capped: {res.json().get('detail')}")
        return 0
    if res.status_code != 200:
        print(f"    FAIL {res.status_code}: {res.text[:400]}")
        return 1

    session = res.json()
    sid = session["session_id"]
    print(f"    session {sid}")
    print(f"    {len(session['candidates'])} jobs submitted")

    print("\n[2] poll")
    started = time.monotonic()
    while time.monotonic() - started < 600:
        time.sleep(8)
        session = client.get(f"{base}/api/demo/session/{sid}").json()
        ready = sum(1 for c in session["candidates"] if c["status"] == "ready")
        failed = sum(1 for c in session["candidates"] if c["status"] == "failed")
        print(f"    {time.monotonic() - started:5.0f}s  ready={ready} failed={failed} status={session['status']}")
        if session["status"] != "generating":
            break

    ready = [c for c in session["candidates"] if c["status"] == "ready"]
    if not ready:
        print("    FAIL no candidates came back")
        return 1

    picked = ready[0]["index"]
    print(f"\n[3] pick candidate {picked}")
    res = client.post(
        f"{base}/api/demo/select",
        json={"session_id": sid, "picked": picked, "reason": "the eyes read better at small sizes"},
    )
    if res.status_code != 200:
        print(f"    FAIL {res.status_code}: {res.text[:400]}")
        return 1
    session = res.json()

    chosen = next(c for c in session["candidates"] if c["accepted"])
    rejects = [c for c in session["candidates"] if c.get("asset") and not c["accepted"]]
    print(f"    stored {len(rejects) + 1} candidates, run {session['run_id']}")
    for r in rejects[:3]:
        print(f"      reject {r['index']}: {r['reason']}")

    print("\n[4] prove")
    media = client.get(f"{base}{chosen['asset']}").content
    good = client.post(
        f"{base}/api/verify", files={"file": ("chosen.png", media, "image/png")}
    ).json()
    print(f"    verdict={good['verdict']}  withheld={all(s['prompt_withheld'] for s in good['steps'])}")

    print("\n[5] edit and disprove")
    edited = bytearray(media)
    at = int(len(edited) * 0.8)
    edited[at] ^= 0xFF
    bad = client.post(
        f"{base}/api/verify", files={"file": ("edited.png", bytes(edited), "image/png")}
    ).json()
    print(f"    verdict={bad['verdict']}  reason={bad.get('reason')}")

    print("\n[6] inventory")
    again = client.get(f"{base}/api/demo/session/{sid}").json()
    kept = sum(1 for c in again["candidates"] if c["accepted"])
    stored = sum(1 for c in again["candidates"] if c.get("asset"))
    print(f"    {stored} stored, {kept} kept, {stored - kept} rejects retained")

    ok = (
        good["verdict"] == "verified"
        and bad["verdict"] in ("altered", "unsigned", "broken")
        and stored > 1
    )
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
