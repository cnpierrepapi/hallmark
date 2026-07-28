"""Inspect the GMI Cloud request queue directly.

The Genblaze adapter submits to POST /requests and polls GET /requests/{id}.
This talks to the same endpoints without the pipeline in the way, so a stuck
job can be diagnosed and model latency can be measured for real.

    python scripts/gmi_probe.py list
    python scripts/gmi_probe.py time <model-id> [<model-id> ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey"
PROMPT = "A ceramic coffee cup on a sunlit windowsill, warm morning light"
POLL_INTERVAL = 3.0
MAX_WAIT = 300.0


def _client() -> httpx.Client:
    key = os.environ.get("GMI_API_KEY")
    if not key:
        raise SystemExit("GMI_API_KEY is not set")
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=60.0,
    )


def cmd_list() -> int:
    with _client() as client:
        resp = client.get("/requests")
        print(f"GET /requests -> {resp.status_code}")
        if resp.status_code >= 400:
            print(resp.text[:2000])
            return 1

        data = resp.json()
        rows = data if isinstance(data, list) else data.get("requests") or data.get("data") or []
        if not rows:
            print("No jobs returned. Raw response:")
            print(json.dumps(data, indent=2)[:2000])
            return 0

        print(f"{len(rows)} job(s), newest first:\n")
        for row in rows[:15]:
            print(
                f"  {str(row.get('request_id') or row.get('id'))[:28]:30}"
                f"{str(row.get('model')):34}"
                f"{str(row.get('status'))}"
            )
            for field in ("error", "error_message", "message", "failure_reason"):
                if row.get(field):
                    print(f"      {field}: {str(row[field])[:200]}")
        print("\nFull first record:")
        print(json.dumps(rows[0], indent=2)[:2500])
    return 0


TERMINAL_OK = ("succeeded", "success", "completed")
TERMINAL_BAD = ("failed", "error", "cancelled")


def cmd_time(models: list[str]) -> int:
    """Submit every model at once, then poll them together.

    Submitting serially would multiply each model's queue wait by the number
    of models under test. Fanning out first means the whole comparison costs
    roughly as long as the slowest single model.
    """
    inflight: dict[str, str] = {}
    results: list[tuple[str, float | None, str]] = []

    with _client() as client:
        for model in models:
            try:
                resp = client.post(
                    "/requests",
                    json={"model": model, "payload": {"prompt": PROMPT}},
                    timeout=90.0,
                )
            except httpx.HTTPError as exc:
                # A submit that blocks this long is a synchronous model: the
                # POST holds the connection while it renders instead of
                # returning a queued job id. The async adapter cannot use it.
                print(f"[submit] {model:34} HUNG ({type(exc).__name__}) - likely synchronous")
                results.append((model, None, "sync-only (submit blocks)"))
                continue
            if resp.status_code >= 400:
                print(f"[submit] {model:34} FAILED {resp.status_code}: {resp.text[:200]}")
                results.append((model, None, f"submit {resp.status_code}"))
                continue
            body = resp.json()
            request_id = body.get("request_id") or body.get("id")
            inflight[model] = request_id
            print(f"[submit] {model:34} {request_id}")

        started = time.monotonic()
        print(f"\nPolling {len(inflight)} job(s) every {POLL_INTERVAL:.0f}s\n")

        while inflight and time.monotonic() - started < MAX_WAIT:
            time.sleep(POLL_INTERVAL)
            elapsed = time.monotonic() - started

            for model, request_id in list(inflight.items()):
                detail = client.get(f"/requests/{request_id}")
                if detail.status_code >= 400:
                    print(f"  {elapsed:6.0f}s {model:34} poll {detail.status_code}")
                    results.append((model, None, f"poll {detail.status_code}"))
                    del inflight[model]
                    continue

                payload = detail.json()
                status = payload.get("status", "unknown")

                if status in TERMINAL_OK:
                    print(f"  {elapsed:6.0f}s {model:34} {status.upper()}")
                    results.append((model, elapsed, status))
                    del inflight[model]
                elif status in TERMINAL_BAD:
                    print(f"  {elapsed:6.0f}s {model:34} {status.upper()}")
                    print(f"          {json.dumps(payload.get('outcome') or payload)[:300]}")
                    results.append((model, elapsed, status))
                    del inflight[model]

            if inflight:
                waiting = ", ".join(sorted(inflight))
                print(f"  {elapsed:6.0f}s waiting: {waiting}")

        for model in inflight:
            results.append((model, None, "never left queue"))

    print("\n\nSummary")
    print(f"{'model':36}{'seconds':>10}  status")
    for model, elapsed, status in sorted(results, key=lambda r: (r[1] is None, r[1] or 0)):
        shown = f"{elapsed:.1f}" if elapsed is not None else "-"
        print(f"{model:36}{shown:>10}  {status}")
    return 0


def main() -> int:
    load_dotenv(ROOT / ".env")
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "list":
        return cmd_list()
    if sys.argv[1] == "time":
        models = sys.argv[2:]
        if not models:
            print("Give at least one model id")
            return 1
        return cmd_time(models)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
