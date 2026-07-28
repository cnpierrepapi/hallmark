"""HTTP service for verifying stamped media.

Deliberately public and unauthenticated. A provenance check that requires an
account is not a provenance check: the people who most need to test a file are
the ones with no relationship to whoever made it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from hallmark.verify import verify

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Video is the large case. Beyond this a verifier should be checking the
# stored asset rather than uploading, and an unbounded limit is a free
# denial-of-service on a public endpoint.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="HALLMARK",
    description="Verify how a piece of media was generated.",
    version="0.1.0",
)


SHOWCASE_KEY = "showcase/current.json"

# Cached with a short life rather than for the process lifetime. A warm
# instance would otherwise serve the same payload forever, so republishing
# content would need a code deploy to show up.
SHOWCASE_TTL_SECONDS = 60
_showcase_cache: dict | None = None
_showcase_fetched_at: float = 0.0


def _showcase() -> dict:
    """Read the published showcase payload.

    Built offline by scripts/publish_showcase.py, so this never touches the
    ledger or pyarrow and the deployed function stays small.
    """
    global _showcase_cache, _showcase_fetched_at

    fresh = time.monotonic() - _showcase_fetched_at < SHOWCASE_TTL_SECONDS
    if _showcase_cache is not None and fresh:
        return _showcase_cache

    from hallmark import storage

    body = storage.client().get_object(Bucket=storage.bucket(), Key=SHOWCASE_KEY)["Body"].read()
    _showcase_cache = json.loads(body)
    _showcase_fetched_at = time.monotonic()
    return _showcase_cache


@app.get("/api/showcase")
def showcase() -> JSONResponse:
    """Public data for the homepage. Contains no prompts by construction."""
    try:
        return JSONResponse(_showcase())
    except Exception as exc:  # noqa: BLE001 - the page degrades rather than breaks
        return JSONResponse({"error": str(exc), "specimens": [], "attempts": []}, status_code=503)


def _stream(key: str, media_type: str) -> StreamingResponse:
    """Proxy an object out of the private bucket.

    Proxied rather than linked, because the bucket stays private and the point
    of the page is that a visitor can download the real stamped file and check
    it themselves.
    """
    from hallmark import storage

    obj = storage.client().get_object(Bucket=storage.bucket(), Key=key)
    return StreamingResponse(
        obj["Body"].iter_chunks(CHUNK_BYTES),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{Path(key).name}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/specimen/{modality}")
def specimen(modality: str) -> StreamingResponse:
    """Stream a stamped campaign specimen."""
    entry = next((s for s in _showcase().get("specimens", []) if s["modality"] == modality), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No specimen for {modality}")
    return _stream(entry["key"], entry["media_type"])


@app.get("/api/gallery/{slug}")
def gallery(slug: str) -> StreamingResponse:
    """Stream the full stamped gallery tile. This is what gets verified."""
    entry = next((t for t in _showcase().get("gallery", []) if t["slug"] == slug), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No gallery tile {slug}")
    return _stream(entry["key"], "image/png")


@app.get("/api/thumb/{slug}")
def thumb(slug: str) -> StreamingResponse:
    """Stream a display-sized thumbnail.

    Display only. A resized copy is not the stamped asset and would not
    verify, which is why the tile click fetches the full PNG instead.
    """
    entry = next((t for t in _showcase().get("gallery", []) if t["slug"] == slug), None)
    if entry is None or not entry.get("thumb_key"):
        raise HTTPException(status_code=404, detail=f"No thumbnail for {slug}")
    return _stream(entry["thumb_key"], "image/webp")


@app.post("/api/demo/generate")
def demo_generate(payload: dict) -> JSONResponse:
    """Submit the candidates and return immediately with their job ids."""
    from hallmark import demo

    brief = (payload.get("brief") or "").strip()
    if not brief:
        raise HTTPException(status_code=400, detail="Describe what to generate")
    if len(brief) > 300:
        raise HTTPException(status_code=400, detail="Keep the brief under 300 characters")

    try:
        session = demo.start_generation(brief, payload.get("session_id"))
    except demo.QuotaExceeded as exc:
        # Not an error state for the page: it falls back to replaying a
        # recorded run so the demo still works once the budget is spent.
        return JSONResponse(
            {"quota_exceeded": True, "detail": str(exc), "quota": demo.quota_status()},
            status_code=429,
        )
    return JSONResponse(_session_public(session))


@app.get("/api/demo/session/{session_id}")
def demo_session(session_id: str) -> JSONResponse:
    """Poll the run. Safe to call repeatedly."""
    from hallmark import demo

    try:
        session = demo.poll_generation(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such session") from None
    return JSONResponse(_session_public(session))


@app.post("/api/demo/select")
def demo_select(payload: dict) -> JSONResponse:
    """Record the pick, store every candidate, and stamp the chosen one."""
    from hallmark import demo

    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        picked = int(payload.get("picked"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="picked must be a number") from None

    with tempfile.TemporaryDirectory(prefix="hallmark-demo-") as tmp:
        try:
            session = demo.select_candidate(
                session_id, picked, (payload.get("reason") or "")[:400], Path(tmp)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="No such session") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    return JSONResponse(_session_public(session))


@app.get("/api/demo/asset/{session_id}/{name}")
def demo_asset(session_id: str, name: str) -> StreamingResponse:
    """Stream a stored candidate out of the private bucket."""
    from hallmark import demo

    session = demo.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such session")

    entry = next(
        (c for c in session["candidates"] if c.get("stored_key", "").endswith(f"/{name}")),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="No such asset")
    return _stream(entry["stored_key"], "image/png")


@app.get("/api/demo/quota")
def demo_quota(session_id: str | None = None) -> dict:
    from hallmark import demo

    return demo.quota_status(session_id)


def _session_public(session: dict) -> dict:
    """Strip the prompt before a session goes over the wire.

    The brief is the visitor's own words so it stays, but the expanded prompt
    sent to the model is exactly what this product promises to withhold.
    """
    return {
        "session_id": session["session_id"],
        "brief": session["brief"],
        "model": session["model"],
        "status": session["status"],
        "created_at": session["created_at"],
        "run_id": session.get("run_id"),
        "manifest_uri": session.get("manifest_uri"),
        "canonical_hash": session.get("canonical_hash"),
        "selection": session.get("selection"),
        "candidates": [
            {
                "index": c["index"],
                "status": c["status"],
                "url": c.get("url"),
                "sha256": c.get("sha256"),
                "size_bytes": c.get("size_bytes"),
                "score": c.get("score"),
                "passed": c.get("passed"),
                "checks": c.get("checks") or [],
                "accepted": c.get("accepted", False),
                "reason": c.get("reason"),
                "asset": (
                    f"/api/demo/asset/{session['session_id']}/{Path(c['stored_key']).name}"
                    if c.get("stored_key")
                    else None
                ),
            }
            for c in session["candidates"]
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    configured = all(
        os.environ.get(key) for key in ("B2_ENDPOINT", "B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET")
    )
    return {"status": "ok", "storage": "configured" if configured else "unconfigured"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/verify")
async def api_verify(file: UploadFile = File(...)) -> JSONResponse:
    """Verify an uploaded file and return its provenance record."""
    suffix = Path(file.filename or "upload").suffix or ".bin"
    written = 0

    # Spooling to disk rather than reading into memory keeps a large upload
    # from being held in RAM, and the container parsers work on paths anyway.
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(handle.name)
    try:
        with handle:
            while chunk := await file.read(CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
                    )
                handle.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        result = verify(tmp_path)
        payload = result.to_dict()
        payload["filename"] = file.filename
        payload["size_bytes"] = written
        return JSONResponse(payload)
    finally:
        tmp_path.unlink(missing_ok=True)
