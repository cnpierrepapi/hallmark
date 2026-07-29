"""Walk the whole demo against a running site, as a visitor would.

Generate, poll, pick, check, edit, check again, read the inventory back, then
open the marking record.

    python scripts/check_demo_live.py [base_url] [image|video] [session_id]

Costs one generation: three images, or two clips. Video is much the more
expensive half, which is why it is never the default.

Pass a session id to resume one that has already rendered instead of starting
another. A slow clip and a bug in the steps after it should not cost a second
render, and paying twice to test the same code is how a small budget goes.

The two kinds do not take the same route through steps 4 and 5, and that is the
point of running both. A still is small enough to post back for checking, so the
visitor's own bytes are what gets verified. A clip is 7 to 18MB and the platform
refuses any request body over 4.5MB, so it is checked where it lies and the
caller hashes its own copy to confirm the checker read the same file.

The signature check is the part worth watching. It asserts the delivered file
carries the approval in metadata a file browser will show, and that the same
file still hashes to what its record declares. Those two have to hold together:
metadata written after hashing would be a caption anyone could rewrite.
"""

from __future__ import annotations

import hashlib
import sys
import time

import httpx

DEFAULT = "https://hallmark-rust.vercel.app"
SIGNER = "Live check"
REASON = "it has a good stance with very vivid features"

BRIEFS = {
    "image": "a bottle of cold brew on wet stone",
    "video": "a bottle of cold brew turning slowly on wet stone",
}

# How long a render is allowed to take before the walk gives up. A clip off
# wan2.7 runs about two and a half minutes and both are submitted together.
PATIENCE = {"image": 600, "video": 900}


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    kind = (sys.argv[2] if len(sys.argv) > 2 else "image").lower()
    resume = sys.argv[3] if len(sys.argv) > 3 else None
    if kind not in BRIEFS:
        print("Usage: check_demo_live.py [base_url] [image|video] [session_id]")
        return 2

    brief = BRIEFS[kind]
    client = httpx.Client(timeout=300.0, follow_redirects=True)

    print(f"[0] pages   ({kind})")
    for path in ("/", "/demo", "/hallmark.css"):
        res = client.get(f"{base}{path}")
        print(f"    {path:14} {res.status_code} {len(res.content):>7} bytes")
        if res.status_code != 200:
            print(f"    FAIL {path} did not serve")
            return 1

    options = client.get(f"{base}/api/demo/styles").json()
    kinds = {k["slug"]: k for k in options.get("kinds", [])}
    print(f"    styles         {[s['slug'] for s in options['styles']]}")
    print(f"    kinds          {list(kinds)}")
    if kind not in kinds:
        print(f"    FAIL the service does not offer {kind}")
        return 1
    quota = client.get(f"{base}/api/demo/quota").json()
    print(f"    quota          today {quota['used_today']}/{quota['daily_cap']}, "
          f"video {quota.get('videos_today')}/{quota.get('video_daily_cap')}")

    if resume:
        print(f"\n[1] resume {resume}, nothing new is generated")
        res = client.get(f"{base}/api/demo/session/{resume}")
        if res.status_code != 200:
            print(f"    FAIL {res.status_code}: {res.text[:400]}")
            return 1
        session = res.json()
    else:
        print(f"\n[1] generate {kinds[kind]['candidates']} {kind}")
        res = client.post(
            f"{base}/api/demo/generate", json={"brief": brief, "style": "product", "kind": kind}
        )
        if res.status_code == 429:
            print(f"    capped: {res.json().get('detail')}")
            return 0
        if res.status_code != 200:
            print(f"    FAIL {res.status_code}: {res.text[:400]}")
            return 1
        session = res.json()

    sid = session["session_id"]
    print(f"    session {sid}, run {session.get('run_seq')}")
    print(f"    {len(session['candidates'])} jobs submitted, "
          f"{session.get('kind_label')} / {session.get('style_label')}")

    print("\n[2] poll")
    started = time.monotonic()
    while time.monotonic() - started < PATIENCE[kind]:
        time.sleep(8)
        session = client.get(f"{base}/api/demo/session/{sid}").json()
        ready = sum(1 for c in session["candidates"] if c["status"] == "ready")
        failed = sum(1 for c in session["candidates"] if c["status"] == "failed")
        print(f"    {time.monotonic() - started:5.0f}s  ready={ready} failed={failed} "
              f"status={session['status']}")
        if session["status"] != "generating":
            break

    ready = [c for c in session["candidates"] if c["status"] == "ready"]
    if not ready:
        print("    FAIL no candidates came back")
        return 1

    picked = ready[0]["index"]
    print(f"\n[3] pick candidate {picked} and sign it off")
    # Timed, because for video this one request downloads every clip, rewrites
    # the winner's property table, stamps it, signs a credential over it and
    # uploads the lot. The platform kills a function at 60 seconds.
    began = time.monotonic()
    res = client.post(
        f"{base}/api/demo/select",
        json={"session_id": sid, "picked": picked, "reason": REASON, "signer": SIGNER},
    )
    took = time.monotonic() - began
    # Only meaningful against the deployment. Run locally, almost all of this
    # is the operator's own upstream link pushing the clip to B2: measured at a
    # flat 55 KB/s here regardless of file size, which turns 7.6MB into well
    # over two minutes on its own. A function in a datacentre is not on that
    # link, so failing a local run on this number would be measuring the wrong
    # machine.
    remote = not any(host in base for host in ("127.0.0.1", "localhost", "0.0.0.0"))
    print(f"    selection took {took:.1f}s"
          + ("" if remote else "  (local run: mostly this machine's upload speed)"))
    if res.status_code != 200:
        print(f"    FAIL {res.status_code}: {res.text[:400]}")
        return 1
    session = res.json()

    chosen = next(c for c in session["candidates"] if c["accepted"])
    rejects = [c for c in session["candidates"] if c.get("asset") and not c["accepted"]]
    print(f"    stored {len(rejects) + 1} candidates, run {session['run_id']}")
    print(f"    delivered as {chosen['media_type']}, {chosen['size_bytes'] / 1024:,.0f}KB, "
          f"uploadable={chosen['uploadable']}")
    for r in rejects:
        print(f"      reject {r['index']}: {r['reason']}")

    # Storage keys have to be unique per run, or a later run overwrites this
    # one's files while the page goes on serving them from cache.
    names = [c["name"] for c in session["candidates"] if c.get("name")]
    unique_names = len(names) == len(set(names)) and all(
        n.startswith(f"r{session['run_seq']}_") for n in names
    )
    print(f"    stored under {names}")

    # The notes must answer the reason the reviewer gave, not quote metrics at
    # them. Checked loosely, because the wording is the model's.
    words = {w for w in REASON.lower().split() if len(w) > 4}
    echoed = sum(1 for r in rejects if any(w in (r["reason"] or "").lower() for w in words))
    print(f"    {echoed}/{len(rejects)} rejection notes echo the reviewer's own words")

    print("\n[4] check the signed file")
    media_type = chosen["media_type"]
    media = client.get(f"{base}{chosen['asset']}").content
    same_bytes = None

    if chosen["uploadable"]:
        suffix = "jpg" if media_type == "image/jpeg" else "png"
        good = client.post(
            f"{base}/api/verify", files={"file": (f"chosen.{suffix}", media, media_type)}
        ).json()
        route = "uploaded and re-hashed"
    else:
        good = client.get(f"{base}/api/demo/verify/{sid}/{chosen['name']}").json()
        mine = hashlib.sha256(media).hexdigest()
        same_bytes = mine == good.get("raw_sha256")
        route = "checked in storage, hash compared against my own copy"
        print(f"    my copy   {mine[:32]}")
        print(f"    checker   {str(good.get('raw_sha256'))[:32]}")

    visible = good.get("visible") or {}
    print(f"    route     {route}")
    print(f"    verdict={good['verdict']}  "
          f"withheld={all(s['prompt_withheld'] for s in good['steps'])}")
    print(f"    approval={good.get('approval')}")
    print("    file properties a browser would show:")
    for key, value in visible.items():
        print(f"      {key:12} {str(value)[:90]}")

    signed_in_file = SIGNER in " ".join(str(v) for v in visible.values())
    print(f"    signer visible in the file itself: {signed_in_file}")

    print("\n[5] edit and check again")
    if chosen["uploadable"]:
        edited = bytearray(media)
        edited[int(len(edited) * 0.8)] ^= 0xFF
        suffix = "jpg" if media_type == "image/jpeg" else "png"
        bad = client.post(
            f"{base}/api/verify",
            files={"file": (f"edited.{suffix}", bytes(edited), media_type)},
        ).json()
        print("    edit made in this script, then posted back")
    else:
        res = client.post(
            f"{base}/api/demo/tamper/{sid}/{chosen['name']}", json={"mode": "byte"}
        )
        bad = res.json()
        print(f"    edit made server side on a throwaway copy: {bad.get('edit')}")
    print(f"    verdict={bad['verdict']}  reason={bad.get('reason')}")

    # Removing only the visible credit has to fail too. Not one frame or pixel
    # changes, and it still refuses, because the credit was inside what was
    # hashed at delivery. That is the whole argument for writing it first.
    stripped = client.post(
        f"{base}/api/demo/tamper/{sid}/{chosen['name']}", json={"mode": "credit"}
    )
    credit_gone = stripped.json() if stripped.status_code == 200 else {}
    if credit_gone:
        print(f"    credit removed: verdict={credit_gone['verdict']} "
              f"({credit_gone.get('size_before')} bytes before)")
    else:
        print(f"    credit removal not available: {stripped.text[:160]}")

    # Nothing that edit touched may have reached storage.
    after = client.get(f"{base}{chosen['asset']}").content
    untouched = hashlib.sha256(after).hexdigest() == hashlib.sha256(media).hexdigest()
    print(f"    the stored file is unchanged by either edit: {untouched}")

    print("\n[6] inventory")
    again = client.get(f"{base}/api/demo/session/{sid}").json()
    kept = sum(1 for c in again["candidates"] if c["accepted"])
    stored = sum(1 for c in again["candidates"] if c.get("asset"))
    totals = again.get("totals") or {}
    print(f"    this run: {stored} stored, {kept} kept, {stored - kept} rejects retained")
    print(f"    this browser: {totals}")

    print("\n[7] marking record")
    sheet = client.get(f"{base}/compliance/session/{sid}")
    record = client.get(f"{base}/compliance/session/{sid}/download")
    covers_all = str(totals.get("assets", 0)) in sheet.text and sheet.status_code == 200
    print(f"    page     {sheet.status_code} {len(sheet.content):>7} bytes")
    print(f"    download {record.status_code} "
          f"{record.headers.get('content-disposition', '')}")

    checks = {
        "signed file verifies": good["verdict"] == "verified",
        "prompt withheld": all(s["prompt_withheld"] for s in good["steps"]),
        "approver recorded": (good.get("approval") or {}).get("approver") == SIGNER,
        "signature visible in the file": signed_in_file,
        "edited file refused": bad["verdict"] in ("altered", "unsigned", "broken"),
        "removing the credit refused": credit_gone.get("verdict")
        in ("altered", "unsigned", "broken"),
        "storage untouched by the edits": untouched,
        "rejects retained": stored > kept,
        "notes echo the reviewer": echoed > 0,
        "each run has its own object names": unique_names,
        "marking record covers the browser": covers_all,
    }
    if remote:
        # The platform kills a function at 60 seconds, and selection is the one
        # request that moves whole files around. This is the check that decides
        # whether video can be offered to the public at all.
        checks["selection fits the 60s function"] = took < 60
    if same_bytes is not None:
        checks["checker read the same bytes I hold"] = same_bytes

    print()
    for label, passed in checks.items():
        print(f"    {'ok  ' if passed else 'FAIL'} {label}")

    ok = all(checks.values())
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
