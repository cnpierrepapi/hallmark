"""Write provenance into a media file, in full or as a pointer.

Genblaze can only embed a pointer as a sidecar file. A sidecar does not travel
with the asset: upload it to a platform or forward it to a client and the
pointer is gone. That defeats the purpose, because pointer mode exists so an
asset can prove itself while keeping the prompt and parameters private.

So the pointer is written inline here, using the same markers the genblaze
handlers use (PNG iTXt keyword, MP4 uuid box, MP3 ID3v2 TXXX frame). Files
stamped this way are read by the stock handlers as well as by ours.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from genblaze_core.media import get_handler
from genblaze_core.media.embedder import guess_mime
from genblaze_core.media.mp3 import TXXX_DESC
from genblaze_core.media.mp4 import GENBLAZE_UUID_BYTES
from genblaze_core.media.png import ITXT_KEY
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.policy import EmbedPolicy

from hallmark.integrity import (
    PNG_SIGNATURE,
    XMP_APP1_HEADER,
    XMP_MANIFEST_CLOSE,
    XMP_MANIFEST_OPEN,
    UnsupportedMediaError,
    _is_manifest_packet,
    _jpeg_segments,
    _read_box_size,
)

# An APP segment's length field is 16 bits, so its payload cannot exceed this.
MAX_APP1_BYTES = 65533


def _png_itxt(keyword: str, text: str) -> bytes:
    body = (
        keyword.encode("latin-1")
        + b"\x00"
        + b"\x00"  # uncompressed
        + b"\x00"  # compression method, ignored when uncompressed
        + b"\x00"  # empty language tag
        + b"\x00"  # empty translated keyword
        + text.encode("utf-8")
    )
    crc = struct.pack(">I", zlib.crc32(b"iTXt" + body) & 0xFFFFFFFF)
    return struct.pack(">I", len(body)) + b"iTXt" + body + crc


def _write_png(source: Path, dest: Path, payload: str) -> None:
    data = source.read_bytes()
    if data[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise UnsupportedMediaError("Not a valid PNG (signature mismatch)")

    keyword = ITXT_KEY.encode("latin-1")
    out = bytearray(PNG_SIGNATURE)
    pos = len(PNG_SIGNATURE)
    saw_ihdr = False

    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        total = 12 + length
        if pos + total > len(data):
            raise UnsupportedMediaError("Truncated PNG chunk")

        # Drop any existing genblaze chunk so re-stamping leaves exactly one.
        if chunk_type == b"iTXt":
            block = data[pos + 8 : pos + 8 + length]
            null_pos = block.find(b"\x00")
            if 0 <= null_pos <= 79 and block[:null_pos] == keyword:
                pos += total
                continue

        out += data[pos : pos + total]
        if chunk_type == b"IHDR":
            saw_ihdr = True
            out += _png_itxt(ITXT_KEY, payload)
        pos += total

    if not saw_ihdr:
        raise UnsupportedMediaError("Not a valid PNG (no IHDR chunk)")
    dest.write_bytes(bytes(out))


def _write_mp4(source: Path, dest: Path, payload: str) -> None:
    data = source.read_bytes()
    out = bytearray()
    pos = 0

    while pos + 8 <= len(data):
        box_size, header_size = _read_box_size(data, pos)
        if box_size < 8 or pos + box_size > len(data):
            raise UnsupportedMediaError("Truncated or malformed MP4 box")

        if data[pos + 4 : pos + 8] == b"uuid" and box_size >= header_size + 16:
            if data[pos + header_size : pos + header_size + 16] == GENBLAZE_UUID_BYTES:
                pos += box_size
                continue

        out += data[pos : pos + box_size]
        pos += box_size

    body = GENBLAZE_UUID_BYTES + payload.encode("utf-8")
    out += struct.pack(">I", len(body) + 8) + b"uuid" + body
    dest.write_bytes(bytes(out))


def _write_jpeg(source: Path, dest: Path, payload: str) -> None:
    """Insert the record as an XMP packet, leaving every other byte alone.

    Genblaze's JPEG handler re-encodes the image through Pillow to write its
    XMP. That rewrites the entire file, so stripping the record afterwards
    cannot return the original bytes, and a hash taken before embedding will
    never match again. Splicing one segment in keeps the round trip exact.

    The packet is deliberately a second one. A delivered asset already carries
    an XMP packet describing it in human words, written before the file was
    hashed, and that one has to survive stripping untouched.
    """
    import html

    data = source.read_bytes()

    # Drop any record already present so re-stamping leaves exactly one.
    kept = bytearray()
    cursor = 0
    for start, end, marker, segment in _jpeg_segments(data):
        if marker == 0xE1 and _is_manifest_packet(segment):
            kept += data[cursor:start]
            cursor = end
    kept += data[cursor:]
    out = bytes(kept)

    # Sit with the other application segments, after any JFIF, EXIF or XMP
    # block the file already carries.
    insert_at = 2
    for _start, end, marker, _segment in _jpeg_segments(out):
        if not 0xE0 <= marker <= 0xEF:
            break
        insert_at = end

    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        ' xmlns:mf="https://github.com/backblaze-labs/genblaze/ns/1.0/">'
        '<rdf:Description rdf:about="">'
        f"{XMP_MANIFEST_OPEN}{html.escape(payload, quote=False)}{XMP_MANIFEST_CLOSE}"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    body = XMP_APP1_HEADER + packet.encode("utf-8")
    if len(body) + 2 > MAX_APP1_BYTES:
        raise UnsupportedMediaError(
            f"Record is {len(body)} bytes, too large for a JPEG APP1 segment. "
            "Use pointer mode."
        )

    marker = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    dest.write_bytes(out[:insert_at] + marker + out[insert_at:])


def _write_mp3(source: Path, dest: Path, payload: str) -> None:
    from mutagen.id3 import ID3, TXXX, ID3NoHeaderError

    if dest != source:
        dest.write_bytes(source.read_bytes())

    try:
        tags = ID3(dest)
    except ID3NoHeaderError:
        tags = ID3()

    tags.delall(f"TXXX:{TXXX_DESC}")
    tags.add(TXXX(encoding=3, desc=TXXX_DESC, text=[payload]))
    tags.save(dest)


_WRITERS = {
    "image/png": _write_png,
    "image/jpeg": _write_jpeg,
    "video/mp4": _write_mp4,
    "audio/mpeg": _write_mp3,
}


def stamp(
    source: Path,
    manifest: Manifest,
    dest: Path,
    *,
    policy: EmbedPolicy | None = None,
    mime_type: str | None = None,
) -> str:
    """Write provenance into a copy of ``source`` at ``dest``.

    Returns the embed mode actually used. With no policy, or an explicit
    ``full`` policy, this defers to the genblaze handler. With ``pointer`` it
    writes the pointer inline, which the SDK cannot do.
    """
    mime = mime_type or guess_mime(source)
    mode = policy.embed_mode if policy else "full"

    if mode == "none":
        if dest != source:
            dest.write_bytes(source.read_bytes())
        return "none"

    if mode == "full":
        handler = get_handler(mime)
        if handler is None:
            raise UnsupportedMediaError(f"No genblaze handler for {mime}")
        handler.embed(source, manifest, dest)
        return "full"

    if manifest.manifest_uri is None:
        raise ValueError("Pointer mode needs manifest.manifest_uri to be set first")

    writer = _WRITERS.get(mime)
    if writer is None:
        raise UnsupportedMediaError(
            f"No pointer writer for {mime}. Supported: {', '.join(sorted(_WRITERS))}"
        )

    writer(source, dest, manifest.to_embed_json(policy))
    return "pointer"
