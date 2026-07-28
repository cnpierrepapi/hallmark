"""Round-trip tests for stamping, across every container we ship.

The property that matters: stamping a file and then stripping it must return
exactly the original bytes. If that fails, every clean asset would verify as
altered, so it is pinned per format rather than assumed.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from genblaze_core.models.asset import Asset
from genblaze_core.models.enums import Modality, PromptVisibility, StepType
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.policy import EmbedPolicy
from genblaze_core.models.run import Run
from genblaze_core.models.step import Step
from PIL import Image

from hallmark.integrity import canonical_bytes, extract_embedded_json
from hallmark.stamp import stamp
from hallmark.verify import BROKEN, VERIFIED, verify

# MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding: 417 bytes per frame.
_MP3_HEADER = b"\xff\xfb\x90\x64"
_MP3_FRAME = _MP3_HEADER + b"\x00" * 413


def _png(path: Path, colour=(200, 120, 40)) -> bytes:
    Image.new("RGB", (48, 32), colour).save(path, "PNG")
    return path.read_bytes()


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def _mp4(path: Path, payload: bytes = b"\x00" * 256) -> bytes:
    data = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41") + _box(b"mdat", payload)
    path.write_bytes(data)
    return data


def _mp3(path: Path, frames: int = 8) -> bytes:
    path.write_bytes(_MP3_FRAME * frames)
    return path.read_bytes()


def _manifest(sha256: str, media_type: str, modality: Modality) -> Manifest:
    step = Step(
        step_type=StepType.GENERATE,
        modality=modality,
        provider="gmicloud",
        model="test-model",
        prompt="a brand new coffee brand launch",
        prompt_visibility=PromptVisibility.PRIVATE,
        assets=[Asset(url="https://example.test/a", media_type=media_type, sha256=sha256)],
    )
    manifest = Manifest.from_run(Run(name="campaign", steps=[step]))
    manifest.manifest_uri = "s3://hackathon-press/manifests/campaign.json"
    return manifest


CASES = [
    ("png", "image/png", Modality.IMAGE, _png),
    ("mp4", "video/mp4", Modality.VIDEO, _mp4),
    ("mp3", "audio/mpeg", Modality.AUDIO, _mp3),
]


@pytest.mark.parametrize("ext,media_type,modality,make", CASES)
class TestPointerRoundTrip:
    def test_strip_recovers_original_bytes(
        self, tmp_path: Path, ext, media_type, modality, make
    ) -> None:
        source = tmp_path / f"src.{ext}"
        original = make(source)
        manifest = _manifest(hashlib.sha256(original).hexdigest(), media_type, modality)

        dest = tmp_path / f"stamped.{ext}"
        mode = stamp(source, manifest, dest, policy=EmbedPolicy(embed_mode="pointer"))

        assert mode == "pointer"
        assert dest.read_bytes() != original, "stamping must change the file"
        assert canonical_bytes(dest) == original, "stripping must restore the original exactly"

    def test_pointer_payload_withholds_the_prompt(
        self, tmp_path: Path, ext, media_type, modality, make
    ) -> None:
        source = tmp_path / f"src.{ext}"
        original = make(source)
        manifest = _manifest(hashlib.sha256(original).hexdigest(), media_type, modality)

        dest = tmp_path / f"stamped.{ext}"
        stamp(source, manifest, dest, policy=EmbedPolicy(embed_mode="pointer"))

        block = extract_embedded_json(dest)
        assert block is not None
        assert set(block) == {"schema_version", "canonical_hash", "manifest_uri"}
        assert "coffee brand launch" not in str(block)

    def test_restamping_stays_idempotent(
        self, tmp_path: Path, ext, media_type, modality, make
    ) -> None:
        source = tmp_path / f"src.{ext}"
        original = make(source)
        manifest = _manifest(hashlib.sha256(original).hexdigest(), media_type, modality)

        once = tmp_path / f"once.{ext}"
        twice = tmp_path / f"twice.{ext}"
        stamp(source, manifest, once, policy=EmbedPolicy(embed_mode="pointer"))
        stamp(once, manifest, twice, policy=EmbedPolicy(embed_mode="pointer"))

        assert canonical_bytes(twice) == original


class TestPointerVerification:
    def test_pointer_file_verifies_when_manifest_resolves(self, tmp_path: Path, monkeypatch) -> None:
        from hallmark import verify as verify_mod

        source = tmp_path / "src.png"
        original = _png(source)
        manifest = _manifest(hashlib.sha256(original).hexdigest(), "image/png", Modality.IMAGE)

        dest = tmp_path / "stamped.png"
        stamp(source, manifest, dest, policy=EmbedPolicy(embed_mode="pointer"))

        monkeypatch.setattr(verify_mod, "_fetch_manifest", lambda uri: manifest)

        result = verify(dest)
        assert result.verdict == VERIFIED
        assert result.manifest_source.startswith("fetched:")
        assert result.steps[0].prompt_withheld
        assert result.steps[0].prompt is None

    def test_pointer_to_a_different_manifest_is_rejected(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from hallmark import verify as verify_mod

        source = tmp_path / "src.png"
        original = _png(source)
        honest = _manifest(hashlib.sha256(original).hexdigest(), "image/png", Modality.IMAGE)

        dest = tmp_path / "stamped.png"
        stamp(source, honest, dest, policy=EmbedPolicy(embed_mode="pointer"))

        # Storage returns a valid but different manifest.
        other = _manifest("b" * 64, "image/png", Modality.IMAGE)
        monkeypatch.setattr(verify_mod, "_fetch_manifest", lambda uri: other)

        result = verify(dest)
        assert result.verdict == BROKEN
        assert "Pointer hash does not match" in (result.reason or "")


class TestFullMode:
    def test_full_mode_still_uses_the_sdk_handler(self, tmp_path: Path) -> None:
        source = tmp_path / "src.png"
        original = _png(source)
        manifest = _manifest(hashlib.sha256(original).hexdigest(), "image/png", Modality.IMAGE)

        dest = tmp_path / "full.png"
        mode = stamp(source, manifest, dest, policy=EmbedPolicy(embed_mode="full"))

        assert mode == "full"
        block = extract_embedded_json(dest)
        assert "run" in block
        assert canonical_bytes(dest) == original
