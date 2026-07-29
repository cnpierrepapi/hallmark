"""Human-readable provenance metadata, written where file browsers look.

The provenance record proper is a hash pointer, and a hash pointer is only
meaningful to a verifier. Someone who downloads an asset and opens its
properties should also be able to read, in plain words, that it was generated,
by what, and who signed it off. Otherwise the proof only exists inside the
system that made it.

Where operating systems actually look was measured, not assumed:

* PNG carrying the same signature as ``tEXt`` chunks, as an XMP packet, and as
  an ``eXIf`` chunk showed **nothing** in the Windows Explorer properties
  dialog. Windows has no property handler that reads PNG metadata.
* The identical EXIF in a JPEG showed up immediately as Title, Subject,
  Authors, Comments, Copyright and Program name.

So the delivered asset is a JPEG, and the visible metadata is written *before*
the file is hashed. The signature is inside the bytes the proof covers rather
than in a second copy travelling alongside them. PNG still gets the same fields
written into it for the tools that do read them (exiftool, Adobe applications,
macOS Preview), because a PNG asset should not be silently poorer.

EXIF tags, and what Windows shows them as:

    0x010E ImageDescription -> Title and Subject
    0x013B Artist           -> Authors
    0x0131 Software         -> Program name
    0x8298 Copyright        -> Copyright
    0x9C9C XPComment        -> Comments
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# The IPTC code for content made by a generative model. This is the vocabulary
# the AI disclosure ecosystem reads, including C2PA tooling, so it is worth far
# more than a sentence of our own prose.
TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)

XMP_KEYWORD = "XML:com.adobe.xmp"

# Quality is high and chroma subsampling is off. The asset is the product, and
# a visible signature is not worth paying for in artefacts.
JPEG_QUALITY = 95

# Every identity this module can add to a PNG. Recorded inside the file so the
# stripper removes exactly what was added and nothing the generator wrote.
_ADDED_KEYWORD = "hallmark:added"


@dataclass(frozen=True)
class Signature:
    """The human-readable half of a provenance record.

    Deliberately free of the manifest hash. These fields are written before the
    file is hashed, so anything derived from that hash cannot appear here
    without a circular dependency. The disclosure is visible; the proof is the
    pointer, and the pointer is what a verifier reads.
    """

    approver: str
    model: str
    provider: str = "GMI Cloud"
    verify_url: str = "https://hallmark-rust.vercel.app"
    approved_at: datetime | None = None
    note: str | None = None
    rights: str | None = None
    brief: str | None = None
    # What the file is, in the words the credit uses. A clip calling itself an
    # image in its own properties undercuts the point of writing it there.
    medium: str = "image"

    @property
    def when(self) -> datetime:
        return self.approved_at or datetime.now(timezone.utc)

    @property
    def day(self) -> str:
        return self.when.strftime("%d %B %Y")

    def title(self) -> str:
        return f"AI generated, approved by {self.approver}"

    def description(self) -> str:
        """Surfaces as Title and Subject in Windows, ImageDescription elsewhere."""
        return (
            f"AI generated {self.medium}, approved by {self.approver} on {self.day}. "
            f"Carries a HALLMARK provenance record."
        )

    def comment(self) -> str:
        """Surfaces as Comments in Windows."""
        parts = [
            f"Generated with {self.model} on {self.provider}.",
            f"Signed off by {self.approver} on {self.day}.",
        ]
        if self.note:
            parts.append(f'Approval note: "{self.note.rstrip(". ")}".')
        parts.append("The prompt is withheld by the publisher.")
        parts.append(f"Check this file has not been altered at {self.verify_url}.")
        return " ".join(parts)

    def software(self) -> str:
        return f"{self.model} via {self.provider}, recorded by HALLMARK"


def _xp(value: str) -> bytes:
    """Encode a Microsoft XP* EXIF tag: UCS-2 little endian, NUL terminated."""
    return value.encode("utf-16-le") + b"\x00\x00"


def exif(signature: Signature):
    """Build the EXIF block. Returns a PIL Exif object, valid for both formats."""
    from PIL import Image

    tags = Image.Exif()
    tags[0x010E] = signature.description()  # ImageDescription
    tags[0x0131] = signature.software()  # Software
    tags[0x013B] = signature.approver  # Artist
    tags[0x0132] = signature.when.strftime("%Y:%m:%d %H:%M:%S")  # DateTime
    if signature.rights:
        tags[0x8298] = signature.rights  # Copyright
    tags[0x9C9B] = _xp(signature.title())  # XPTitle
    tags[0x9C9C] = _xp(signature.comment())  # XPComment
    tags[0x9C9D] = _xp(signature.approver)  # XPAuthor
    tags[0x9C9E] = _xp("AI generated;HALLMARK;provenance")  # XPKeywords
    return tags


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def xmp(signature: Signature) -> bytes:
    """Build an XMP packet carrying the same signature plus the IPTC AI code.

    Kept separate from the pointer packet. This one is part of the asset and is
    hashed with it; the pointer is added afterwards and stripped before hashing.
    """
    rights = (
        f"<dc:rights><rdf:Alt><rdf:li xml:lang='x-default'>"
        f"{_escape(signature.rights)}</rdf:li></rdf:Alt></dc:rights>"
        if signature.rights
        else ""
    )
    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        ' xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"'
        ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
        ' xmlns:hallmark="https://hallmark-rust.vercel.app/ns/1.0/">'
        f"<dc:title><rdf:Alt><rdf:li xml:lang='x-default'>"
        f"{_escape(signature.title())}</rdf:li></rdf:Alt></dc:title>"
        f"<dc:description><rdf:Alt><rdf:li xml:lang='x-default'>"
        f"{_escape(signature.comment())}</rdf:li></rdf:Alt></dc:description>"
        f"<dc:creator><rdf:Seq><rdf:li>{_escape(signature.approver)}</rdf:li></rdf:Seq></dc:creator>"
        f"{rights}"
        f"<xmp:CreatorTool>{_escape(signature.software())}</xmp:CreatorTool>"
        f"<xmp:CreateDate>{signature.when.replace(microsecond=0).isoformat()}</xmp:CreateDate>"
        f"<photoshop:Credit>{_escape(signature.approver)}</photoshop:Credit>"
        f"<Iptc4xmpExt:DigitalSourceType>{TRAINED_ALGORITHMIC_MEDIA}"
        f"</Iptc4xmpExt:DigitalSourceType>"
        f"<hallmark:approver>{_escape(signature.approver)}</hallmark:approver>"
        f"<hallmark:model>{_escape(signature.model)}</hallmark:model>"
        f"<hallmark:promptVisibility>withheld</hallmark:promptVisibility>"
        f"<hallmark:verifyAt>{_escape(signature.verify_url)}</hallmark:verifyAt>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


# --- JPEG ----------------------------------------------------------------


def to_jpeg(source: Path, dest: Path, signature: Signature, quality: int = JPEG_QUALITY) -> Path:
    """Re-encode an image as a JPEG carrying the visible signature.

    This runs before hashing, so the metadata is part of the asset rather than
    an attachment to it. Editing the signature out of a delivered file changes
    its hash, which is the point: the disclosure is as tamper evident as the
    pixels.
    """
    from PIL import Image

    with Image.open(source) as img:
        rgb = img.convert("RGB")
        rgb.save(
            dest,
            "JPEG",
            quality=quality,
            subsampling=0,
            exif=exif(signature),
            xmp=xmp(signature),
        )
    return dest


# --- PNG -----------------------------------------------------------------


def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    return struct.pack(">I", len(body)) + kind + body + crc


def _text(keyword: str, value: str) -> bytes:
    return _chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + value.encode("latin-1", "replace"))


def _itxt(keyword: str, value: str) -> bytes:
    body = keyword.encode("latin-1") + b"\x00\x00\x00\x00\x00" + value.encode("utf-8")
    return _chunk(b"iTXt", body)


def png_chunk_identities(data: bytes) -> list[str]:
    """List the identity of every chunk in a PNG, text chunks by keyword."""
    identities = []
    pos = len(PNG_SIGNATURE)
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        name = kind.decode("latin-1")
        if kind in (b"tEXt", b"iTXt", b"zTXt"):
            body = data[pos + 8 : pos + 8 + length]
            end = body.find(b"\x00")
            if 0 <= end <= 79:
                name = f"{name}:{body[:end].decode('latin-1')}"
        identities.append(name)
        pos += 12 + length
    return identities


def apply_png(source: Path, dest: Path, signature: Signature) -> list[str]:
    """Write the visible signature into a PNG as text, XMP and EXIF chunks.

    Windows will not show any of it, which is why the delivered asset is a
    JPEG. Everything else that reads image metadata will, so a PNG kept as a
    PNG is not left bare.

    Returns the chunk identities added. They are also recorded inside the file
    so the stripper removes exactly these and never something the generator
    wrote itself.
    """
    data = source.read_bytes()
    if data[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise ValueError("Not a valid PNG (signature mismatch)")

    existing = set(png_chunk_identities(data))
    fields = {
        "Title": signature.title(),
        "Author": signature.approver,
        "Description": signature.description(),
        "Software": signature.software(),
        "Comment": signature.comment(),
        "Creation Time": signature.when.strftime("%a, %d %b %Y %H:%M:%S GMT"),
    }
    if signature.rights:
        fields["Copyright"] = signature.rights

    added: list[str] = []
    block = b""
    for keyword, value in fields.items():
        identity = f"tEXt:{keyword}"
        if identity in existing:
            continue
        block += _text(keyword, value)
        added.append(identity)

    if f"iTXt:{XMP_KEYWORD}" not in existing:
        block += _itxt(XMP_KEYWORD, xmp(signature).decode("utf-8"))
        added.append(f"iTXt:{XMP_KEYWORD}")

    if "eXIf" not in existing:
        block += _chunk(b"eXIf", exif(signature).tobytes())
        added.append("eXIf")

    if not added:
        dest.write_bytes(data)
        return []

    # The list goes in first so a stripper reads it before anything else.
    block = _itxt(_ADDED_KEYWORD, ",".join(added)) + block
    added.insert(0, f"iTXt:{_ADDED_KEYWORD}")

    ihdr_end = len(PNG_SIGNATURE) + 12 + struct.unpack(">I", data[8:12])[0]
    dest.write_bytes(data[:ihdr_end] + block + data[ihdr_end:])
    return added


def strip_png(data: bytes) -> bytes:
    """Remove the chunks apply_png added, using the list it recorded.

    Guessing by keyword would be wrong: a generator is free to write its own
    Software or Description chunk, and removing that would change bytes this
    pipeline never touched.
    """
    added = _read_added(data)
    if not added:
        return data

    wanted = set(added)
    out = bytearray(PNG_SIGNATURE)
    pos = len(PNG_SIGNATURE)

    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        total = 12 + length
        identity = kind.decode("latin-1")
        if kind in (b"tEXt", b"iTXt", b"zTXt"):
            body = data[pos + 8 : pos + 8 + length]
            end = body.find(b"\x00")
            if 0 <= end <= 79:
                identity = f"{identity}:{body[:end].decode('latin-1')}"

        # Only the first occurrence, because apply_png never adds an identity
        # the file already had, and it inserts at the front.
        if identity in wanted:
            wanted.discard(identity)
            pos += total
            continue

        out += data[pos : pos + total]
        pos += total

    return bytes(out)


def _read_added(data: bytes) -> list[str]:
    """Read the recorded list of chunks apply_png added to this PNG."""
    if data[: len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        return []
    keyword = _ADDED_KEYWORD.encode("latin-1")
    pos = len(PNG_SIGNATURE)
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        if data[pos + 4 : pos + 8] == b"iTXt":
            body = data[pos + 8 : pos + 8 + length]
            end = body.find(b"\x00")
            if 0 <= end <= 79 and body[:end] == keyword:
                cursor = end + 3
                lang = body.find(b"\x00", cursor)
                translated = body.find(b"\x00", lang + 1)
                text = body[translated + 1 :].decode("utf-8")
                return [f"iTXt:{_ADDED_KEYWORD}"] + [p for p in text.split(",") if p]
        pos += 12 + length
    return []


# What Windows reads off an MP4. It uses the iTunes style atoms, so these are
# the four that actually surface in the Details tab: Title, Subtitle, Comments
# and Contributing artists, plus the encoder line under Program name.
MP4_ATOMS = {
    "title": "\xa9nam",
    "subtitle": "\xa9st3",
    "comment": "\xa9cmt",
    "artist": "\xa9ART",
    "encoder": "\xa9too",
}


def _source_metadata(data: bytes) -> dict[str, str]:
    """Read the generator's own key/value metadata out of an MP4.

    Clips arrive with an Apple style ``mdta`` table, where the names live in a
    ``keys`` box and the values are stored against their position in it. The
    video model writes a JSON blob of its own under ``AIGC``, naming who
    produced the render, and that is provenance. It has to survive.
    """
    try:
        keys_at = data.index(b"keys")
    except ValueError:
        return {}

    count = struct.unpack(">I", data[keys_at + 8 : keys_at + 12])[0]
    names: list[str] = []
    pos = keys_at + 12
    for _ in range(count):
        if pos + 8 > len(data):
            return {}
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        names.append(data[pos + 8 : pos + size].decode("utf-8", "replace"))
        pos += size

    from mutagen.mp4 import MP4

    found: dict[str, str] = {}
    with io.BytesIO(data) as handle:
        for key, values in MP4(handle).items():
            if len(key) == 4 and key[0] == "\x00":
                index = struct.unpack(">I", key.encode("latin-1"))[0]
                if 1 <= index <= len(names) and values:
                    found[names[index - 1]] = str(values[0])
    return found


# The container's own bookkeeping. Reproduced by any encoder and not worth
# carrying across as a claim about where the clip came from.
_BOILERPLATE = {"major_brand", "minor_version", "compatible_brands"}


def to_mp4(source: Path, dest: Path, signature: Signature) -> Path:
    """Copy a clip and write the signature where a file browser will show it.

    A still has carried its credit in EXIF since the delivery format changed to
    JPEG, and a clip carried nothing: right click, properties, and the file a
    marking record describes said nothing about itself at all.

    Windows only reads the iTunes style table, and these clips arrive with the
    Apple ``mdta`` one, which it ignores. Measured, not assumed: the same tags
    under ``mdta`` show nothing in the Details tab and under ``mdir`` show
    Title, Comments, Authors and Contributing artists immediately. So the old
    table is replaced, and everything the generator had written into it is
    carried across first, because destroying the render's own provenance to
    make room for ours would be the exact failure this product exists to stop.

    Written before the clip is hashed, exactly as the stills are, so editing the
    credit back out breaks verification in the same way editing the picture
    does. mutagen fixes the chunk offsets as it rewrites, which is safe here
    because nothing has been hashed yet.
    """
    from mutagen.mp4 import MP4, MP4FreeForm

    data = source.read_bytes()
    carried = _source_metadata(data)
    dest.write_bytes(_strip_udta(data))

    clip = MP4(dest)
    clip[MP4_ATOMS["title"]] = [signature.title()]
    clip[MP4_ATOMS["subtitle"]] = [signature.description()]
    clip[MP4_ATOMS["comment"]] = [signature.comment()]
    clip[MP4_ATOMS["artist"]] = [signature.approver]
    clip[MP4_ATOMS["encoder"]] = [signature.software()]
    for name, value in carried.items():
        if name in _BOILERPLATE:
            continue
        clip[f"----:com.hallmark:generator.{name}"] = [
            MP4FreeForm(value.encode("utf-8"))
        ]
    clip.save()
    return dest


def _strip_udta(data: bytes) -> bytes:
    """Remove the user data box, so a fresh iTunes style one can be written.

    Only when the index sits after the media, which is where these clips put
    it. Shrinking a box ahead of the media would move every sample and leave
    the offset table pointing at the wrong bytes, so in that layout the old
    table is left alone: worse properties beat a broken file.
    """
    if data.find(b"moov") < data.rfind(b"mdat"):
        return data

    out = bytearray()
    pos = 0
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        if size == 0:
            size = len(data) - pos
        if size < 8 or pos + size > len(data):
            return data

        if data[pos + 4 : pos + 8] == b"moov":
            inner = _drop_child(data[pos + header : pos + size], b"udta")
            out += struct.pack(">I", len(inner) + 8) + b"moov" + inner
        else:
            out += data[pos : pos + size]
        pos += size
    return bytes(out)


def _drop_child(body: bytes, kind: bytes) -> bytes:
    kept = bytearray()
    pos = 0
    while pos + 8 <= len(body):
        size = struct.unpack(">I", body[pos : pos + 4])[0]
        if size < 8 or pos + size > len(body):
            return body
        if body[pos + 4 : pos + 8] != kind:
            kept += body[pos : pos + size]
        pos += size
    return bytes(kept)


def _read_mp4_visible(path: Path) -> dict[str, str]:
    from mutagen.mp4 import MP4

    clip = MP4(path)
    fields: dict[str, str] = {}
    for name, atom in MP4_ATOMS.items():
        values = clip.get(atom) or []
        if values:
            fields[name] = str(values[0])
    return fields


def read_visible(path: Path) -> dict[str, str]:
    """Read back the fields a file browser would show. Used by the checks."""
    from PIL import Image

    if path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        return _read_mp4_visible(path)

    fields: dict[str, str] = {}
    with Image.open(path) as img:
        tags = img.getexif()
        for tag, name in ((0x010E, "description"), (0x0131, "software"), (0x013B, "artist")):
            if tags.get(tag):
                fields[name] = str(tags[tag])
        for tag, name in ((0x9C9B, "title"), (0x9C9C, "comment"), (0x9C9D, "author")):
            raw = tags.get(tag)
            if raw:
                value = raw.decode("utf-16-le") if isinstance(raw, bytes) else str(raw)
                fields[name] = value.rstrip("\x00")
        for key in ("Title", "Author", "Description", "Comment", "Software"):
            if key in img.info:
                fields.setdefault(key.lower(), img.info[key])
    return fields
