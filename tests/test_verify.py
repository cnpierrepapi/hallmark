"""Tests for the verification layer and the HTTP service.

Covers the four verdicts and the pointer-rebinding attack: a forger who
repoints an embedded pointer at a manifest that legitimately describes their
altered file must not get a pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from genblaze_core.media import get_handler
from genblaze_core.models.asset import Asset
from genblaze_core.models.enums import Modality, PromptVisibility, StepType
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.policy import EmbedPolicy
from genblaze_core.models.run import Run
from genblaze_core.models.step import Step
from PIL import Image

from hallmark import verify as verify_mod
from hallmark.api import app
from hallmark.verify import ALTERED, BROKEN, UNSIGNED, VERIFIED, verify


def _make_png(path: Path, colour: tuple[int, int, int] = (200, 120, 40)) -> bytes:
    Image.new("RGB", (64, 48), colour).save(path, "PNG")
    return path.read_bytes()


def _manifest_for(
    sha256: str,
    *,
    prompt: str = "a ceramic cup on a windowsill",
    visibility: PromptVisibility = PromptVisibility.PUBLIC,
) -> Manifest:
    step = Step(
        step_type=StepType.GENERATE,
        modality=Modality.IMAGE,
        provider="gmicloud-image",
        model="gpt-image-2-generate",
        prompt=prompt,
        prompt_visibility=visibility,
        assets=[Asset(url="https://example.test/a.png", media_type="image/png", sha256=sha256)],
    )
    return Manifest.from_run(Run(name="campaign-test", steps=[step]))


def _stamp(tmp_path: Path, colour=(200, 120, 40)) -> tuple[Path, Manifest]:
    source = tmp_path / "src.png"
    original = _make_png(source, colour)
    manifest = _manifest_for(hashlib.sha256(original).hexdigest())
    stamped = tmp_path / "stamped.png"
    get_handler("image/png").embed(source, manifest, stamped)
    return stamped, manifest


class TestVerdicts:
    def test_verified(self, tmp_path: Path) -> None:
        stamped, manifest = _stamp(tmp_path)

        result = verify(stamped)
        assert result.verdict == VERIFIED
        assert result.ok
        assert result.reason is None
        assert result.manifest_source == "embedded"
        assert result.computed_sha256 == result.declared_sha256
        assert result.run_name == "campaign-test"
        assert result.canonical_hash == manifest.canonical_hash

    def test_altered(self, tmp_path: Path) -> None:
        _, manifest = _stamp(tmp_path)

        forged_src = tmp_path / "forged.png"
        _make_png(forged_src, (9, 9, 9))
        forged = tmp_path / "forged_stamped.png"
        get_handler("image/png").embed(forged_src, manifest, forged)

        result = verify(forged)
        assert result.verdict == ALTERED
        assert not result.ok
        assert result.computed_sha256 != result.declared_sha256

    def test_unsigned(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.png"
        _make_png(bare)

        result = verify(bare)
        assert result.verdict == UNSIGNED
        assert result.steps == []

    def test_broken_when_record_is_edited(self, tmp_path: Path) -> None:
        stamped, manifest = _stamp(tmp_path)

        # Rewrite the prompt inside the embedded record but keep the old hash.
        data = manifest.model_dump(mode="python")
        data["run"]["steps"][0]["prompt"] = "something the publisher never asked for"
        tampered = tmp_path / "tampered_record.png"
        source = tmp_path / "src2.png"
        _make_png(source)
        text = json.dumps(data, default=str)
        raw = source.read_bytes()
        from genblaze_core.media.png import _build_itxt, _walk_chunks

        out = bytearray(raw[:8])
        for pos, chunk_type, total in _walk_chunks(raw):
            out += raw[pos : pos + total]
            if chunk_type == b"IHDR":
                out += _build_itxt("genblaze:manifest", text)
        tampered.write_bytes(bytes(out))

        result = verify(tampered)
        assert result.verdict == BROKEN
        assert "does not match its own contents" in (result.reason or "")


class TestPromptPrivacy:
    def test_private_prompt_is_withheld(self, tmp_path: Path) -> None:
        source = tmp_path / "src.png"
        original = _make_png(source)
        manifest = _manifest_for(
            hashlib.sha256(original).hexdigest(),
            prompt="our proprietary brand prompt",
            visibility=PromptVisibility.PRIVATE,
        )
        stamped = tmp_path / "stamped.png"
        get_handler("image/png").embed(source, manifest, stamped)

        result = verify(stamped)
        assert result.verdict == VERIFIED
        assert result.steps[0].prompt_withheld
        assert result.steps[0].prompt is None

        # And the secret must not leak through the serialized payload.
        assert "proprietary" not in json.dumps(result.to_dict())


class TestPointerMode:
    def test_pointer_is_resolved_from_storage(self, tmp_path: Path, monkeypatch) -> None:
        source = tmp_path / "src.png"
        original = _make_png(source)
        manifest = _manifest_for(hashlib.sha256(original).hexdigest())
        manifest.manifest_uri = "s3://hackathon-press/manifests/abc.json"

        # PngHandler.embed always writes the full manifest, so the pointer
        # block is built directly here. Manifest resolution is what is under
        # test, not the handler's choice of payload.
        stamped = tmp_path / "pointer.png"
        pointer = json.loads(manifest.to_embed_json(EmbedPolicy(embed_mode="pointer")))
        _embed_raw(source, stamped, json.dumps(pointer))

        monkeypatch.setattr(verify_mod, "_fetch_manifest", lambda uri: manifest)

        result = verify(stamped)
        assert result.verdict == VERIFIED
        assert result.manifest_source.startswith("fetched:")

    def test_repointed_manifest_is_rejected(self, tmp_path: Path, monkeypatch) -> None:
        """A pointer must not be able to name a manifest describing other bytes."""
        forged_src = tmp_path / "forged.png"
        forged_bytes = _make_png(forged_src, (3, 3, 3))

        # A manifest that honestly describes the forger's own file.
        attacker_manifest = _manifest_for(hashlib.sha256(forged_bytes).hexdigest())

        # The pointer stamped into the file still claims the original hash.
        honest = _manifest_for("a" * 64)
        honest.manifest_uri = "s3://hackathon-press/manifests/honest.json"
        pointer = json.loads(honest.to_embed_json(EmbedPolicy(embed_mode="pointer")))

        stamped = tmp_path / "repointed.png"
        _embed_raw(forged_src, stamped, json.dumps(pointer))

        monkeypatch.setattr(verify_mod, "_fetch_manifest", lambda uri: attacker_manifest)

        result = verify(stamped)
        assert result.verdict == BROKEN
        assert "Pointer hash does not match" in (result.reason or "")


def _embed_raw(source: Path, dest: Path, text: str) -> None:
    """Write an arbitrary genblaze iTXt payload into a PNG."""
    from genblaze_core.media.png import _build_itxt, _walk_chunks

    raw = source.read_bytes()
    out = bytearray(raw[:8])
    for pos, chunk_type, total in _walk_chunks(raw):
        out += raw[pos : pos + total]
        if chunk_type == b"IHDR":
            out += _build_itxt("genblaze:manifest", text)
    dest.write_bytes(bytes(out))


class TestApi:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_index_serves(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Hallmark" in response.text

    def test_verify_endpoint_passes_clean_file(self, client: TestClient, tmp_path: Path) -> None:
        stamped, _ = _stamp(tmp_path)
        response = client.post(
            "/api/verify",
            files={"file": ("stamped.png", stamped.read_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == VERIFIED
        assert body["filename"] == "stamped.png"
        assert body["steps"][0]["model"] == "gpt-image-2-generate"

    def test_verify_endpoint_catches_alteration(self, client: TestClient, tmp_path: Path) -> None:
        _, manifest = _stamp(tmp_path)
        forged_src = tmp_path / "f.png"
        _make_png(forged_src, (1, 2, 3))
        forged = tmp_path / "f_stamped.png"
        get_handler("image/png").embed(forged_src, manifest, forged)

        response = client.post(
            "/api/verify",
            files={"file": ("f_stamped.png", forged.read_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["verdict"] == ALTERED

    def test_empty_upload_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/verify", files={"file": ("x.png", b"", "image/png")})
        assert response.status_code == 400
