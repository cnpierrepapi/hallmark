"""The signature has to be visible in the file, not only in our system.

A record a file browser cannot show is a record most people will never see. So
the delivered asset carries the approval as ordinary image metadata, written
before the file is hashed, which pins two properties worth testing:

  * the fields a properties dialog reads are actually present, and
  * editing them out changes the hash, exactly like editing the picture does.

The format choice is not cosmetic either. Windows shows nothing at all for PNG
metadata, measured by writing the same signature into a PNG as text chunks, as
XMP and as EXIF and reading it back through the shell property system. That is
why delivery is JPEG, and why these tests hold the JPEG path to a higher bar
than the PNG one.
"""

from __future__ import annotations

import hashlib
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from hallmark import metadata
from hallmark.metadata import Signature


@pytest.fixture
def signature() -> Signature:
    return Signature(
        approver="Ama",
        model="gpt-image-2-generate",
        approved_at=datetime(2026, 7, 28, 9, 30, tzinfo=timezone.utc),
        note="it has a good stance with very vivid features",
        rights="Onenept Studios",
        verify_url="https://hallmark-rust.vercel.app",
    )


def _png(path: Path) -> bytes:
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for x in range(64):
        for y in range(64):
            pixels[x, y] = ((x * 5) % 256, (y * 9) % 256, ((x * y) % 256))
    image.save(path, "PNG")
    return path.read_bytes()


class TestJpegDelivery:
    def test_the_fields_a_properties_dialog_reads_are_present(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        source = tmp_path / "raw.png"
        _png(source)
        delivered = metadata.to_jpeg(source, tmp_path / "asset.jpg", signature)

        visible = metadata.read_visible(delivered)
        # Windows maps these to Title, Comments, Authors and Program name.
        assert "Ama" in visible["title"]
        assert "Ama" in visible["artist"]
        assert signature.note in visible["comment"]
        assert "gpt-image-2-generate" in visible["software"]

    def test_the_approval_note_survives_into_the_file(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        source = tmp_path / "raw.png"
        _png(source)
        delivered = metadata.to_jpeg(source, tmp_path / "asset.jpg", signature)

        raw = delivered.read_bytes()
        assert b"trainedAlgorithmicMedia" in raw, "the IPTC AI source code must be written"
        assert b"withheld" in raw
        assert signature.note.encode() in raw

    def test_editing_the_signature_out_changes_the_hash(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        """The visible credit is inside the bytes the record covers.

        Written after hashing it would be a caption anyone could rewrite. This
        is the test that says it was written before.
        """
        source = tmp_path / "raw.png"
        _png(source)
        delivered = metadata.to_jpeg(source, tmp_path / "asset.jpg", signature)
        signed_hash = hashlib.sha256(delivered.read_bytes()).hexdigest()

        forged = metadata.to_jpeg(
            source,
            tmp_path / "forged.jpg",
            Signature(approver="Someone Else", model=signature.model,
                      approved_at=signature.approved_at),
        )
        assert hashlib.sha256(forged.read_bytes()).hexdigest() != signed_hash

    def test_a_plain_jpeg_carries_no_signature(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.jpg"
        Image.new("RGB", (64, 64), (30, 30, 30)).save(plain, "JPEG")
        assert metadata.read_visible(plain) == {}


class TestPngAdditions:
    def test_stripping_returns_the_generator_bytes_exactly(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        """Nothing the model wrote may be disturbed.

        Generated PNGs arrive carrying a signed C2PA box from the provider.
        Removing hallmark's own chunks has to leave that untouched, which is
        why the added chunks are recorded in the file rather than guessed at
        by keyword.
        """
        source = tmp_path / "raw.png"
        original = _png(source)

        dest = tmp_path / "visible.png"
        added = metadata.apply_png(source, dest, signature)

        assert "eXIf" in added
        assert f"iTXt:{metadata.XMP_KEYWORD}" in added
        assert dest.read_bytes() != original
        assert metadata.strip_png(dest.read_bytes()) == original

    def test_a_chunk_the_generator_already_wrote_is_left_alone(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        source = tmp_path / "raw.png"
        from PIL import PngImagePlugin

        image = Image.new("RGB", (64, 64), (90, 40, 20))
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "some other pipeline")
        image.save(source, "PNG", pnginfo=info)
        original = source.read_bytes()

        dest = tmp_path / "visible.png"
        added = metadata.apply_png(source, dest, signature)

        assert "tEXt:Software" not in added, "must not add a second Software chunk"
        assert metadata.strip_png(dest.read_bytes()) == original
        assert b"some other pipeline" in dest.read_bytes()

    def test_applying_twice_is_stable(self, tmp_path: Path, signature: Signature) -> None:
        source = tmp_path / "raw.png"
        original = _png(source)

        once = tmp_path / "once.png"
        metadata.apply_png(source, once, signature)
        twice = tmp_path / "twice.png"
        metadata.apply_png(once, twice, signature)

        # The second pass finds every identity present and adds nothing, so
        # the file is unchanged and still strips back to the original.
        assert twice.read_bytes() == once.read_bytes()
        assert metadata.strip_png(twice.read_bytes()) == original


NUL = b"\x00"


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body) + 8) + kind + body


def _clip(path: Path, entries: dict[str, str] | None = None) -> bytes:
    """A tiny MP4 carrying the Apple style table these clips arrive with.

    Built rather than recorded, because the point of the conversion is that the
    names live in a keys box and the values are stored against their position
    in it. A fixture written the iTunes way would pass the test without the
    code under test doing anything.
    """
    entries = entries or {"encoder": "Lavf60.16.100"}
    fixture = Path(__file__).parent / "fixtures" / "tiny.mp4"
    data = fixture.read_bytes()

    keys = b"".join(_box(b"mdta", name.encode()) for name in entries)
    keys = _box(b"keys", struct.pack(">II", 0, len(entries)) + keys)
    values = b"".join(
        _box(struct.pack(">I", i), _box(b"data", struct.pack(">II", 1, 0) + v.encode()))
        for i, v in enumerate(entries.values(), start=1)
    )
    meta = _box(b"meta", NUL * 4
                + _box(b"hdlr", NUL * 8 + b"mdta" + NUL * 12)
                + keys + _box(b"ilst", values))

    moov_at = data.rindex(b"moov") - 4
    size = struct.unpack(">I", data[moov_at:moov_at + 4])[0]
    body = data[moov_at + 8:moov_at + size]
    # The old table goes, so what is left is only the one built here.
    rebuilt = _box(b"moov", _drop(body, b"udta") + _box(b"udta", meta))

    data = data[:moov_at] + rebuilt + data[moov_at + size:]
    path.write_bytes(data)
    return data


def _drop(body: bytes, kind: bytes) -> bytes:
    out, pos = bytearray(), 0
    while pos + 8 <= len(body):
        size = struct.unpack(">I", body[pos:pos + 4])[0]
        if size < 8:
            break
        if body[pos + 4:pos + 8] != kind:
            out += body[pos:pos + size]
        pos += size
    return bytes(out)


class TestMp4:
    def test_windows_readable_fields_are_written(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        """Windows reads the iTunes table only, which is measured, not assumed.

        The same tags under the Apple table these clips arrive with show nothing
        at all in the properties dialog.
        """
        source = tmp_path / "raw.mp4"
        _clip(source)

        dest = tmp_path / "delivered.mp4"
        metadata.to_mp4(source, dest, signature)

        body = dest.read_bytes()
        table = body.rindex(b"meta")
        assert b"mdir" in body[table : table + 48], "Windows ignores the mdta table"
        assert b"mdta" not in body[table:], "the old table must not be left behind"

        fields = metadata.read_visible(dest)
        assert signature.approver in fields["artist"]
        assert signature.model in fields["comment"]
        assert "AI generated" in fields["title"]

    def test_the_generators_own_metadata_survives(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        """Replacing the table must not throw away what the model wrote in it.

        The video model records who produced the render under its own key.
        Destroying that to make room for our credit would be the exact failure
        this product exists to stop.
        """
        from mutagen.mp4 import MP4

        blob = '{"ContentProducer":"001191330106","ProduceID":"R-lnPp61"}'
        source = tmp_path / "raw.mp4"
        _clip(source, {"AIGC": blob, "encoder": "Lavf60.16.100"})

        dest = tmp_path / "delivered.mp4"
        metadata.to_mp4(source, dest, signature)

        clip = MP4(dest)
        carried = clip.get("----:com.hallmark:generator.AIGC")
        assert carried, "the generator's own record was dropped"
        assert bytes(carried[0]).decode() == blob
        assert clip["\xa9nam"], "our own credit must be there as well"

    def test_editing_the_credit_changes_the_file(
        self, tmp_path: Path, signature: Signature
    ) -> None:
        source = tmp_path / "raw.mp4"
        _clip(source)
        dest = tmp_path / "delivered.mp4"
        metadata.to_mp4(source, dest, signature)

        before = hashlib.sha256(dest.read_bytes()).hexdigest()
        edited = dest.read_bytes().replace(
            signature.approver.encode(), b"X" * len(signature.approver)
        )
        assert hashlib.sha256(edited).hexdigest() != before
