"""HTTP service for verifying stamped media.

Deliberately public and unauthenticated. A provenance check that requires an
account is not a provenance check: the people who most need to test a file are
the ones with no relationship to whoever made it.
"""

from __future__ import annotations

import json
import os
import tempfile
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
_showcase_cache: dict[str, object] | None = None


def _showcase() -> dict:
    """Read the published showcase payload, cached for the process lifetime.

    Built offline by scripts/publish_showcase.py, so this never touches the
    ledger or pyarrow and the deployed function stays small.
    """
    global _showcase_cache
    if _showcase_cache is None:
        from hallmark import storage

        body = storage.client().get_object(Bucket=storage.bucket(), Key=SHOWCASE_KEY)["Body"].read()
        _showcase_cache = json.loads(body)
    return _showcase_cache


@app.get("/api/showcase")
def showcase() -> JSONResponse:
    """Public data for the homepage. Contains no prompts by construction."""
    try:
        return JSONResponse(_showcase())
    except Exception as exc:  # noqa: BLE001 - the page degrades rather than breaks
        return JSONResponse({"error": str(exc), "specimens": [], "attempts": []}, status_code=503)


@app.get("/api/specimen/{modality}")
def specimen(modality: str) -> StreamingResponse:
    """Stream a stamped specimen from the private bucket.

    Proxied rather than linked, because the bucket stays private and the point
    of the page is that a visitor can download the real stamped file and check
    it themselves.
    """
    from hallmark import storage

    entry = next((s for s in _showcase().get("specimens", []) if s["modality"] == modality), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No specimen for {modality}")

    obj = storage.client().get_object(Bucket=storage.bucket(), Key=entry["key"])
    filename = Path(entry["key"]).name

    return StreamingResponse(
        obj["Body"].iter_chunks(CHUNK_BYTES),
        media_type=entry["media_type"],
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


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
