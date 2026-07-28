"""Tests for offline integrity checking.

These build real PNG and MP4 containers, stamp them through the genblaze
handlers, and assert that a clean file verifies while a modified one does not.
No provider credits required: the container logic is what is under test.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from genblaze_core.media import get_handler
from genblaze_core.media.mp4 import GENBLAZE_UUID_BYTES
from genblaze_core.models.asset import Asset
from genblaze_core.models.enums import Modality, StepType
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.run import Run
from genblaze_core.models.step import Step
from PIL import Image

from hallmark.integrity import (
    UnsupportedMediaError,
    canonical_bytes,
    canonical_sha256,
    verify_file,
)


def _make_png(path: Path, colour: tuple[int, int, int] = (200, 120, 40)) -> bytes:
    Image.new("RGB", (64, 48), colour).save(path, "PNG")
    return path.read_bytes()


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def _make_mp4(path: Path, payload: bytes = b"\x00" * 512) -> bytes:
    data = (
        _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
        + _box(b"free", b"")
        + _box(b"mdat", payload)
    )
    path.write_bytes(data)
    return data


def _manifest_for(sha256: str, media_type: str) -> Manifest:
    step = Step(
        step_type=StepType.GENERATE,
        modality=Modality.IMAGE if media_type.startswith("image/") else Modality.VIDEO,
        provider="test",
        model="test-model",
        prompt="a test asset",
        assets=[Asset(url="https://example.test/asset", media_type=media_type, sha256=sha256)],
    )
    return Manifest.from_run(Run(name="integrity-test", steps=[step]))


class TestPng:
    def test_clean_file_verifies(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.png"
        original = _make_png(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "image/png")

        stamped = tmp_path / "stamped.png"
        get_handler("image/png").embed(source, manifest, stamped)

        assert stamped.read_bytes() != original, "stamping must change the file"

        report = verify_file(stamped)
        assert report.ok
        assert report.manifest_ok
        assert report.bytes_ok
        assert report.reason is None

    def test_stripping_recovers_the_original_bytes(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.png"
        original = _make_png(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "image/png")

        stamped = tmp_path / "stamped.png"
        get_handler("image/png").embed(source, manifest, stamped)

        assert canonical_bytes(stamped) == original
        assert canonical_sha256(stamped) == hashlib.sha256(original).hexdigest()

    def test_tampered_pixels_are_caught(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.png"
        original = _make_png(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "image/png")

        stamped = tmp_path / "stamped.png"
        get_handler("image/png").embed(source, manifest, stamped)

        # Re-encode different pixel data, then stamp the same manifest onto it.
        # This is the realistic attack: a valid-looking manifest attached to
        # content that is not what the manifest describes.
        forged_source = tmp_path / "forged.png"
        _make_png(forged_source, colour=(10, 10, 10))
        forged = tmp_path / "forged_stamped.png"
        get_handler("image/png").embed(forged_source, manifest, forged)

        report = verify_file(forged)
        assert not report.ok
        assert report.manifest_ok, "the manifest itself is still internally valid"
        assert not report.bytes_ok
        assert report.reason == "Media bytes changed after generation"

    def test_unstamped_file_reports_no_manifest(self, tmp_path: Path) -> None:
        source = tmp_path / "bare.png"
        _make_png(source)

        report = verify_file(source)
        assert not report.ok
        assert report.reason is not None
        assert "No manifest" in report.reason

    def test_restamping_is_idempotent(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.png"
        original = _make_png(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "image/png")

        handler = get_handler("image/png")
        once = tmp_path / "once.png"
        handler.embed(source, manifest, once)
        twice = tmp_path / "twice.png"
        handler.embed(once, manifest, twice)

        assert canonical_bytes(twice) == original
        assert verify_file(twice).ok


class TestMp4:
    def test_clean_file_verifies(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.mp4"
        original = _make_mp4(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "video/mp4")

        stamped = tmp_path / "stamped.mp4"
        get_handler("video/mp4").embed(source, manifest, stamped)

        assert GENBLAZE_UUID_BYTES in stamped.read_bytes()
        assert canonical_bytes(stamped) == original

        report = verify_file(stamped)
        assert report.ok, report.reason

    def test_tampered_payload_is_caught(self, tmp_path: Path) -> None:
        source = tmp_path / "clean.mp4"
        original = _make_mp4(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest(), "video/mp4")

        stamped = tmp_path / "stamped.mp4"
        get_handler("video/mp4").embed(source, manifest, stamped)

        forged_source = tmp_path / "forged.mp4"
        _make_mp4(forged_source, payload=b"\x01" * 512)
        forged = tmp_path / "forged_stamped.mp4"
        get_handler("video/mp4").embed(forged_source, manifest, forged)

        report = verify_file(forged)
        assert not report.ok
        assert not report.bytes_ok
        assert report.reason == "Media bytes changed after generation"


class TestUnsupported:
    def test_unknown_container_raises(self, tmp_path: Path) -> None:
        odd = tmp_path / "notes.bin"
        odd.write_bytes(b"just some bytes")

        with pytest.raises(UnsupportedMediaError):
            canonical_bytes(odd)
