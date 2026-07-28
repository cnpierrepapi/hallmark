"""Offline integrity checking for stamped media.

Embedding a manifest changes the file. The PNG handler inserts an ``iTXt``
chunk after IHDR; the MP4 handler appends a ``uuid`` box. So a stamped file's
SHA-256 never equals the ``sha256`` its own manifest records for the generated
bytes.

Genblaze resolves this by fetching ``asset.url`` and hashing the stored copy,
which assumes the verifier can reach the bucket. A third party holding only a
downloaded file cannot do that. This module closes the gap: strip the genblaze
block, hash what remains, and compare against the manifest.

The stripping logic reads the container formats directly using the markers the
genblaze handlers write, so it stays correct without depending on private
functions inside the SDK.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from genblaze_core.media import get_handler
from genblaze_core.media.embedder import guess_mime
from genblaze_core.media.mp3 import TXXX_DESC
from genblaze_core.media.mp4 import GENBLAZE_UUID_BYTES
from genblaze_core.media.png import ITXT_KEY
from genblaze_core.models.manifest import Manifest

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class UnsupportedMediaError(Exception):
    """Raised when a file's container has no known strip routine."""


@dataclass(frozen=True)
class IntegrityReport:
    """The result of checking one stamped file against its own manifest.

    ``manifest_ok`` means the manifest is internally consistent: its canonical
    hash matches its payload and every output declares a well-formed sha256.
    ``bytes_ok`` means the media itself is unaltered since generation.

    Both must hold. A file can carry a perfectly valid manifest describing
    content that has since been edited, which is precisely the case this
    product exists to catch.
    """

    manifest_ok: bool
    bytes_ok: bool
    computed_sha256: str
    declared_sha256: str | None
    media_type: str
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.manifest_ok and self.bytes_ok


def _strip_png(data: bytes) -> bytes:
    """Return PNG bytes with the genblaze iTXt chunk removed."""
    if data[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise UnsupportedMediaError("Not a valid PNG (signature mismatch)")

    out = bytearray(PNG_SIGNATURE)
    pos = len(PNG_SIGNATURE)
    keyword = ITXT_KEY.encode("latin-1")

    while pos < len(data):
        if pos + 12 > len(data):
            raise UnsupportedMediaError("Truncated PNG chunk header")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        total = 12 + length
        if pos + total > len(data):
            raise UnsupportedMediaError("Truncated PNG chunk")

        if chunk_type == b"iTXt":
            payload = data[pos + 8 : pos + 8 + length]
            null_pos = payload.find(b"\x00")
            if 0 <= null_pos <= 79 and payload[:null_pos] == keyword:
                pos += total
                continue

        out += data[pos : pos + total]
        pos += total

    return bytes(out)


def _read_box_size(data: bytes, pos: int) -> tuple[int, int]:
    """Return ``(box_size, header_size)`` for the MP4 box starting at ``pos``.

    Handles the 64-bit ``largesize`` form and the ``size == 0`` form, which
    means the box runs to end of file.
    """
    size = struct.unpack(">I", data[pos : pos + 4])[0]
    if size == 1:
        if pos + 16 > len(data):
            raise UnsupportedMediaError("Truncated MP4 largesize box")
        return struct.unpack(">Q", data[pos + 8 : pos + 16])[0], 16
    if size == 0:
        return len(data) - pos, 8
    return size, 8


def _strip_mp4(data: bytes) -> bytes:
    """Return MP4 bytes with the genblaze uuid box removed."""
    out = bytearray()
    pos = 0

    while pos + 8 <= len(data):
        box_size, header_size = _read_box_size(data, pos)
        if box_size < 8 or pos + box_size > len(data):
            raise UnsupportedMediaError("Truncated or malformed MP4 box")

        box_type = data[pos + 4 : pos + 8]
        if box_type == b"uuid" and box_size >= header_size + 16:
            box_uuid = data[pos + header_size : pos + header_size + 16]
            if box_uuid == GENBLAZE_UUID_BYTES:
                pos += box_size
                continue

        out += data[pos : pos + box_size]
        pos += box_size

    return bytes(out)


def _embedded_json_png(data: bytes) -> str | None:
    """Return the raw genblaze iTXt text from a PNG, or None if absent."""
    if data[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise UnsupportedMediaError("Not a valid PNG (signature mismatch)")

    keyword = ITXT_KEY.encode("latin-1")
    pos = len(PNG_SIGNATURE)

    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        total = 12 + length
        if pos + total > len(data):
            break

        if chunk_type == b"iTXt":
            payload = data[pos + 8 : pos + 8 + length]
            null_pos = payload.find(b"\x00")
            if 0 <= null_pos <= 79 and payload[:null_pos] == keyword:
                # keyword NUL, compression flag, compression method,
                # language NUL, translated keyword NUL, then the text.
                cursor = null_pos + 3
                lang_end = payload.find(b"\x00", cursor)
                tkw_end = payload.find(b"\x00", lang_end + 1)
                return payload[tkw_end + 1 :].decode("utf-8")

        pos += total

    return None


def _embedded_json_mp4(data: bytes) -> str | None:
    """Return the raw genblaze uuid box payload from an MP4, or None."""
    pos = 0
    while pos + 8 <= len(data):
        box_size, header_size = _read_box_size(data, pos)
        if box_size < 8 or pos + box_size > len(data):
            break

        if data[pos + 4 : pos + 8] == b"uuid" and box_size >= header_size + 16:
            if data[pos + header_size : pos + header_size + 16] == GENBLAZE_UUID_BYTES:
                return data[pos + header_size + 16 : pos + box_size].decode("utf-8")

        pos += box_size

    return None


def _strip_mp3(data: bytes) -> bytes:
    """Return MP3 bytes with the genblaze ID3v2 TXXX frame removed.

    Unlike PNG and MP4, this cannot patch bytes in place, because the ID3
    container is rewritten by the tag library on save. Removing only our own
    frame and letting mutagen rewrite the tag is deterministic for files this
    pipeline produced, which is what the round-trip test pins down.
    """
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.mp3 import MP3

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        handle.write(data)
        tmp = Path(handle.name)

    try:
        try:
            tags = ID3(tmp)
        except ID3NoHeaderError:
            return data

        if not tags.getall(f"TXXX:{TXXX_DESC}"):
            return data

        tags.delall(f"TXXX:{TXXX_DESC}")
        if len(tags.keys()) == 0:
            # Nothing else was ever in the tag, so remove it entirely rather
            # than leave an empty ID3 header the original did not have.
            tags.delete(tmp)
            audio = MP3(tmp)
            audio.save()
        else:
            tags.save(tmp)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def _embedded_json_mp3(data: bytes) -> str | None:
    """Return the raw genblaze TXXX payload from an MP3, or None."""
    from mutagen.id3 import ID3, ID3NoHeaderError

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        handle.write(data)
        tmp = Path(handle.name)

    try:
        try:
            tags = ID3(tmp)
        except ID3NoHeaderError:
            return None
        frame = tags.get(f"TXXX:{TXXX_DESC}")
        if frame is None:
            return None
        return frame.text[0]
    finally:
        tmp.unlink(missing_ok=True)


_STRIPPERS = {
    "image/png": _strip_png,
    "video/mp4": _strip_mp4,
    "audio/mpeg": _strip_mp3,
}

_EXTRACTORS = {
    "image/png": _embedded_json_png,
    "video/mp4": _embedded_json_mp4,
    "audio/mpeg": _embedded_json_mp3,
}


def extract_embedded_json(path: Path, mime_type: str | None = None) -> dict | None:
    """Return the embedded genblaze block as a dict, without interpreting it.

    Needed because pointer mode embeds ``{schema_version, canonical_hash,
    manifest_uri}`` rather than a manifest. Handing that to ``parse_manifest``
    fails, so callers need to look at the shape before deciding what it is.
    """
    mime = mime_type or guess_mime(path)
    extractor = _EXTRACTORS.get(mime)
    if extractor is None:
        raise UnsupportedMediaError(
            f"No extractor for {mime}. Supported: {', '.join(sorted(_EXTRACTORS))}"
        )
    text = extractor(path.read_bytes())
    if text is None:
        return None
    return json.loads(text)


def canonical_bytes(path: Path, mime_type: str | None = None) -> bytes:
    """Return the media bytes as they were before any manifest was embedded."""
    mime = mime_type or guess_mime(path)
    stripper = _STRIPPERS.get(mime)
    if stripper is None:
        raise UnsupportedMediaError(
            f"No strip routine for {mime}. Supported: {', '.join(sorted(_STRIPPERS))}"
        )
    return stripper(path.read_bytes())


def canonical_sha256(path: Path, mime_type: str | None = None) -> str:
    """Hash a file with any embedded manifest removed."""
    return hashlib.sha256(canonical_bytes(path, mime_type)).hexdigest()


def _declared_sha256(manifest: Manifest, computed: str) -> str | None:
    """Find the sha256 this manifest records for the file we are holding.

    A run can produce several assets. Prefer an exact match so a multi-asset
    manifest verifies whichever of its outputs the caller happens to have,
    then fall back to the last output so a mismatch reports a real comparison
    rather than an empty one.
    """
    declared = [
        asset.sha256
        for step in manifest.run.steps
        for asset in step.assets
        if asset.sha256
    ]
    if computed in declared:
        return computed
    return declared[-1] if declared else None


def verify_file(path: Path, mime_type: str | None = None) -> IntegrityReport:
    """Check a stamped media file against the manifest embedded inside it."""
    mime = mime_type or guess_mime(path)

    handler = get_handler(mime)
    if handler is None:
        return IntegrityReport(
            manifest_ok=False,
            bytes_ok=False,
            computed_sha256="",
            declared_sha256=None,
            media_type=mime,
            reason=f"No genblaze handler for {mime}",
        )

    try:
        manifest = handler.extract(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a reason
        return IntegrityReport(
            manifest_ok=False,
            bytes_ok=False,
            computed_sha256="",
            declared_sha256=None,
            media_type=mime,
            reason=f"No manifest found: {exc}",
        )

    report = manifest.verification_report()
    computed = canonical_sha256(path, mime)
    declared = _declared_sha256(manifest, computed)

    reason = None
    if not report.hash_ok:
        reason = "Manifest hash does not match its own payload"
    elif report.unverified_sha256_ids:
        reason = "Manifest has outputs without a valid sha256"
    elif declared is None:
        reason = "Manifest declares no output hashes to compare against"
    elif declared != computed:
        reason = "Media bytes changed after generation"

    return IntegrityReport(
        manifest_ok=report.ok,
        bytes_ok=declared is not None and declared == computed,
        computed_sha256=computed,
        declared_sha256=declared,
        media_type=mime,
        reason=reason,
    )
