"""Reading a file that came from somewhere else.

The checker's hard case is a file with no relationship to us at all. These
pin the two things that were wrong about the old answer: that a credential
from another signer was ignored entirely, and that "no record of ours" was
reported in language a reader would take to mean "nothing here".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallmark import provenance
from hallmark.provenance import Finding


def _ours(verdict: str) -> Finding:
    return Finding(
        source="hallmark",
        present=verdict != "unsigned",
        signed=verdict in ("verified", "altered"),
        trusted=verdict == "verified",
        says="",
        detail={"verdict": verdict},
    )


def _credential(source_type: str | None, issuer: str = "Adobe Inc.",
                generator: str = "GPT Image 2", failures: list | None = None) -> Finding:
    return Finding(
        source="c2pa",
        present=True,
        signed=True,
        trusted=None,
        says="",
        detail={
            "issuer": issuer,
            "generator": generator,
            "digital_source_type": source_type,
            "failures": failures or [],
        },
    )


class TestReadingSomeoneElsesCredential:
    def test_a_signed_generative_credential_is_reported_as_such(self):
        summary = provenance.summarise(
            [_ours("unsigned"), _credential("trainedAlgorithmicMedia")]
        )
        assert summary["generated_by_signed_credential"] is True
        assert "Adobe Inc." in summary["headline"]
        assert "GPT Image 2" in summary["headline"], (
            "the model named in the action, not the application that signed it"
        )

    def test_our_own_record_takes_the_headline(self):
        summary = provenance.summarise(
            [_ours("verified"), _credential("trainedAlgorithmicMedia")]
        )
        assert "this pipeline" in summary["headline"]

    def test_a_credential_that_fails_its_checks_is_not_read_as_proof(self):
        summary = provenance.summarise(
            [_ours("unsigned"),
             _credential("trainedAlgorithmicMedia", failures=["assertion.hashedURI.mismatch"])]
        )
        assert summary["generated_by_signed_credential"] is False
        assert "fails its own checks" in summary["headline"]

    def test_an_unsigned_claim_is_never_promoted_to_signed(self):
        claims = Finding(
            source="claims",
            present=True,
            signed=False,
            detail={"declared_source_type": "trainedAlgorithmicMedia"},
        )
        summary = provenance.summarise([_ours("unsigned"), claims])
        assert summary["generated_by_signed_credential"] is False
        assert summary["generated_by_unsigned_claim"] is True
        assert "nothing signs the claim" in summary["headline"]

    def test_a_file_with_nothing_says_nothing_rather_than_clearing_it(self):
        summary = provenance.summarise(
            [_ours("unsigned"), Finding("c2pa", present=False), Finding("claims", present=False)]
        )
        assert summary["headline"] == "Nothing in this file says how it was made."
        assert "genuine" in summary["caveat"], (
            "every answer has to say that a missing credential proves nothing"
        )

    def test_the_stripping_caveat_is_on_every_answer(self):
        for findings in ([_ours("verified")], [_ours("unsigned")],
                         [_ours("unsigned"), _credential("trainedAlgorithmicMedia")]):
            assert "strip" in provenance.summarise(findings)["caveat"]


class TestAgainstRealFiles:
    """Reading actual bytes, not a fixture describing them."""

    @pytest.fixture
    def firefly_png(self) -> Path:
        path = Path("out/gallery/still-coldbrew-can_raw.png")
        if not path.exists():
            pytest.skip("no generated render on disk to read")
        return path

    def test_the_providers_credential_is_found_and_named(self, firefly_png: Path):
        found = provenance.read_c2pa(firefly_png)
        assert found.present and found.signed
        assert found.detail["issuer"] == "Adobe Inc."
        assert found.detail["digital_source_type"] == "trainedAlgorithmicMedia"
        assert found.detail["generator"] == "GPT Image 2"

    def test_the_shipped_trust_list_vouches_for_a_real_signer(self, firefly_png: Path):
        """Without the list every credential reads untrusted, Adobe's included."""
        found = provenance.read_c2pa(firefly_png)
        assert found.trusted is True
        assert found.detail["untrusted"] is False
        assert found.detail["failures"] == []

    def test_the_trust_list_is_vendored_rather_than_fetched(self):
        """Verification must not depend on reaching another host mid-request."""
        for name in ("anchors.pem", "allowed.sha256.txt", "store.cfg"):
            assert (provenance.TRUST_DIR / name).exists(), f"{name} is not shipped"

    def test_a_file_with_no_credential_reads_cleanly(self, tmp_path: Path):
        plain = tmp_path / "plain.jpg"
        from PIL import Image

        Image.new("RGB", (32, 32), (120, 90, 60)).save(plain, "JPEG")
        found = provenance.read_c2pa(plain)
        assert found.present is False
        assert "No Content Credential" in found.says
