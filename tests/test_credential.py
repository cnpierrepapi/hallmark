"""Attaching a credential without destroying everything already in the file.

Three things have to survive the same bytes: the visible credit a file browser
shows, our own pointer, and the provider's signature on the render it came
from. The last one was being thrown away entirely by the conversion to JPEG,
and the first two are easy to lose while fixing that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hallmark import credential, integrity, metadata, provenance


@pytest.fixture
def identity():
    """Signing needs a real key, which lives in the environment and not here."""
    if not credential.available():
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not credential.available():
        pytest.skip("no signing identity configured in this environment")


@pytest.fixture
def delivered() -> Path:
    path = Path("out/gallery/still-coldbrew-can.jpg")
    if not path.exists():
        pytest.skip("no delivered asset on disk")
    return path


@pytest.fixture
def parent() -> Path:
    path = Path("out/gallery/still-coldbrew-can_raw.png")
    if not path.exists():
        pytest.skip("no source render on disk")
    return path


class TestSigningIsAdditive:
    def test_the_credential_carries_the_render_it_came_from(
        self, identity, delivered: Path, parent: Path, tmp_path: Path
    ):
        out = tmp_path / "signed.jpg"
        assert credential.sign(delivered, out, parent=parent,
                               model="gpt-image-2-generate", approver="Studio Lead")

        found = provenance.read_c2pa(out)
        assert found.present and found.signed
        assert found.detail["ingredients"] == 1, "the parent render is recorded"
        assert found.detail["digital_source_type"] == "trainedAlgorithmicMedia"
        assert found.detail["generator"] == "gpt-image-2-generate", (
            "the model that made it, not the tool that signed it"
        )

    def test_the_pointer_and_the_visible_credit_both_survive(
        self, identity, delivered: Path, parent: Path, tmp_path: Path
    ):
        out = tmp_path / "signed.jpg"
        assert credential.sign(delivered, out, parent=parent, model="m", approver="a")
        assert integrity.extract_embedded_json(out, "image/jpeg"), "pointer lost"
        assert len(metadata.read_visible(out)) == len(metadata.read_visible(delivered))

    def test_signing_does_not_change_the_bytes_our_record_covers(
        self, identity, delivered: Path, parent: Path, tmp_path: Path
    ):
        """The credential is written after hashing, so the hash must ignore it.

        Checked on the canonical bytes rather than by verifying, because
        verifying resolves a pointer against the bucket and the suite is not
        allowed near it. This is the property that makes verification survive.
        """
        out = tmp_path / "signed.jpg"
        assert credential.sign(delivered, out, parent=parent, model="m", approver="a")
        assert integrity.canonical_sha256(out, "image/jpeg") == integrity.canonical_sha256(
            delivered, "image/jpeg"
        ), "signing changed the bytes our record is taken over"


class TestMissingIdentityIsNotAFailure:
    def test_no_key_configured_leaves_delivery_alone(
        self, monkeypatch, delivered: Path, tmp_path: Path
    ):
        monkeypatch.delenv(credential.CERT_ENV, raising=False)
        monkeypatch.delenv(credential.KEY_ENV, raising=False)
        out = tmp_path / "unsigned.jpg"
        assert credential.sign(delivered, out) is False
        assert not out.exists(), "a missing key must not leave a half written file"

    def test_a_broken_key_does_not_raise(self, monkeypatch, delivered: Path, tmp_path: Path):
        monkeypatch.setenv(credential.CERT_ENV, "-----BEGIN CERTIFICATE-----\nnope\n")
        monkeypatch.setenv(credential.KEY_ENV, "-----BEGIN PRIVATE KEY-----\nnope\n")
        assert credential.sign(delivered, tmp_path / "x.jpg") is False


class TestEnvironmentMangling:
    def test_escaped_newlines_are_accepted(self, monkeypatch):
        """A hosting dashboard and an env file disagree about multi line values."""
        monkeypatch.setenv(credential.CERT_ENV, "-----BEGIN CERTIFICATE-----\\nAAA\\n-----END CERTIFICATE-----")
        pem = credential._pem(credential.CERT_ENV)
        assert pem is not None
        assert b"\\n" not in pem and pem.count(b"\n") >= 2
