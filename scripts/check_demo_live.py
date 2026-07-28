"""Walk the whole demo against the deployed site, as a visitor would.

Generate, poll, pick, check, edit, check again, then read the inventory back.
Costs one generation of three images.

The signature check is the part worth watching. It asserts the delivered file
carries the approval in metadata a file browser will show, and that the same
file still hashes to what its record declares. Those two have to hold together:
metadata written after hashing would be a caption anyone could rewrite.

    python scripts/check_demo_live.py [base_url]
"""

from __future__ import annotations

import sys
import time

import httpx

DEFAULT = "https://hallmark-rust.vercel.app"
BRIEF = "a bottle of cold brew on wet stone"
STYLE = "product"
SIGNER = "Live check"
REASON = "it has a good stance with very vivid features"


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    client = httpx.Client(timeout=180.0, follow_redirects=True)

    print("[0] pages")
    for path in ("/", "/demo", "/hallmark.css"):
        res = client.get(f"{base}{path}")
        print(f"    {path:14} {res.status_code} {len(res.content):>7} bytes")
        if res.status_code != 200:
            print(f"    FAIL {path} did not serve")
            return 1

    styles = client.get(f"{base}/api/demo/styles").json()
    print(f"    styles         {[s['slug'] for s in styles['styles']]}")
    print(f"    quota          {client.get(f'{base}/api/demo/quota').json()}")

    print("\n[1] generate")
    res = client.post(f"{base}/api/demo/generate", json={"brief": BRIEF, "style": STYLE})
    if res.status_code == 429:
        print(f"    capped: {res.json().get('detail')}")
        return 0
    if res.status_code != 200:
        print(f"    FAIL {res.status_code}: {res.text[:400]}")
        return 1

    session = res.json()
    sid = session["session_id"]
    print(f"    session {sid}")
    print(f"    {len(session['candidates'])} jobs submitted, style {session.get('style_label')}")

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
    print(f"\n[3] pick candidate {picked} and sign it off")
    res = client.post(
        f"{base}/api/demo/select",
        json={"session_id": sid, "picked": picked, "reason": REASON, "signer": SIGNER},
    )
    if res.status_code != 200:
        print(f"    FAIL {res.status_code}: {res.text[:400]}")
        return 1
    session = res.json()

    chosen = next(c for c in session["candidates"] if c["accepted"])
    rejects = [c for c in session["candidates"] if c.get("asset") and not c["accepted"]]
    print(f"    stored {len(rejects) + 1} candidates, run {session['run_id']}")
    print(f"    delivered as {chosen['media_type']}")
    for r in rejects:
        print(f"      reject {r['index']}: {r['reason']}")

    # The notes must answer the reason the reviewer gave, not quote metrics at
    # them. Checked loosely, because the wording is the model's.
    words = {w for w in REASON.lower().split() if len(w) > 4}
    echoed = sum(
        1 for r in rejects if any(w in (r["reason"] or "").lower() for w in words)
    )
    print(f"    {echoed}/{len(rejects)} rejection notes echo the reviewer's own words")

    print("\n[4] check the signed file")
    media_type = chosen["media_type"]
    name = "chosen.jpg" if media_type == "image/jpeg" else "chosen.png"
    media = client.get(f"{base}{chosen['asset']}").content
    good = client.post(f"{base}/api/verify", files={"file": (name, media, media_type)}).json()
    visible = good.get("visible") or {}
    print(f"    verdict={good['verdict']}  withheld={all(s['prompt_withheld'] for s in good['steps'])}")
    print(f"    approval={good.get('approval')}")
    print("    file properties a browser would show:")
    for key, value in visible.items():
        print(f"      {key:12} {value[:90]}")

    signed_in_file = SIGNER in " ".join(visible.values())
    print(f"    signer visible in the file itself: {signed_in_file}")

    print("\n[5] edit and check again")
    edited = bytearray(media)
    at = int(len(edited) * 0.8)
    edited[at] ^= 0xFF
    bad = client.post(
        f"{base}/api/verify", files={"file": (f"edited{name[-4:]}", bytes(edited), media_type)}
    ).json()
    print(f"    verdict={bad['verdict']}  reason={bad.get('reason')}")

    print("\n[6] inventory")
    again = client.get(f"{base}/api/demo/session/{sid}").json()
    kept = sum(1 for c in again["candidates"] if c["accepted"])
    stored = sum(1 for c in again["candidates"] if c.get("asset"))
    print(f"    {stored} stored, {kept} kept, {stored - kept} rejects retained")

    checks = {
        "signed file verifies": good["verdict"] == "verified",
        "prompt withheld": all(s["prompt_withheld"] for s in good["steps"]),
        "approver recorded": (good.get("approval") or {}).get("approver") == SIGNER,
        "signature visible in the file": signed_in_file,
        "edited file refused": bad["verdict"] in ("altered", "unsigned", "broken"),
        "rejects retained": stored > kept,
        "notes echo the reviewer": echoed > 0,
    }
    print()
    for label, passed in checks.items():
        print(f"    {'ok  ' if passed else 'FAIL'} {label}")

    ok = all(checks.values())
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
