"""Tests for the quality gate and the approval record."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from genblaze_core.models.asset import Asset
from genblaze_core.models.enums import Modality, PromptVisibility, StepType
from genblaze_core.models.manifest import Manifest, parse_manifest
from genblaze_core.models.run import Run
from genblaze_core.models.step import Step
from PIL import Image

from hallmark import approval
from hallmark.approval import Approval
from hallmark.evaluate import evaluate_image, evaluate_video


def _manifest() -> Manifest:
    step = Step(
        step_type=StepType.GENERATE,
        modality=Modality.IMAGE,
        provider="gmicloud-image",
        model="gpt-image-2-generate",
        prompt="a coffee ad",
        prompt_visibility=PromptVisibility.PRIVATE,
        assets=[Asset(url="https://example.test/a.png", media_type="image/png", sha256="a" * 64)],
    )
    return Manifest.from_run(Run(name="campaign", steps=[step]))


class TestQualityChecks:
    def test_blank_image_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.png"
        Image.new("RGB", (512, 512), (128, 128, 128)).save(path, "PNG")

        result = evaluate_image(path)
        assert not result.passed
        assert any(c.name == "not_blank" and not c.passed for c in result.checks)
        assert "pixel spread" in (result.reason or "")

    def test_detailed_image_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "busy.png"
        img = Image.new("RGB", (512, 512))
        img.putdata([((x * 7) % 256, (y * 11) % 256, (x + y) % 256) for y in range(512) for x in range(512)])
        img.save(path, "PNG")

        result = evaluate_image(path)
        assert result.passed
        assert result.score == 1.0

    def test_tiny_image_fails_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / "small.png"
        img = Image.new("RGB", (64, 64))
        img.putdata([((x * 9) % 256, (y * 13) % 256, 200) for y in range(64) for x in range(64)])
        img.save(path, "PNG")

        result = evaluate_image(path)
        assert not result.passed
        assert any(c.name == "resolution" and not c.passed for c in result.checks)

    def test_video_without_media_data_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.mp4"
        header = struct.pack(">I", 20) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isom"
        path.write_bytes(header)

        result = evaluate_video(path)
        assert not result.passed
        assert any(c.name == "has_media_data" and not c.passed for c in result.checks)

    def test_non_video_fails_container_check(self, tmp_path: Path) -> None:
        path = tmp_path / "notvideo.mp4"
        path.write_bytes(b"this is not an mp4 at all")

        result = evaluate_video(path)
        assert not result.passed
        assert result.checks[0].name == "container"


class TestApprovalRecord:
    def test_approval_changes_the_hash(self) -> None:
        manifest = _manifest()
        before = manifest.canonical_hash

        approval.apply(manifest, Approval(approver="Ada", decision="approved"))

        assert manifest.canonical_hash != before
        assert manifest.verify_hash()
        assert approval.read(manifest)["approver"] == "Ada"

    def test_forged_approver_breaks_verification(self) -> None:
        """The point of putting sign-off in the hash: it cannot be edited later."""
        manifest = _manifest()
        approval.apply(manifest, Approval(approver="Ada", decision="approved"))

        payload = manifest.model_dump(mode="json")
        payload["run"]["metadata"]["approval"]["approver"] = "Someone Else"

        forged = parse_manifest(payload)
        assert not forged.verify_hash()
        assert not forged.verify()

    def test_approval_note_and_timestamp_are_recorded(self) -> None:
        manifest = _manifest()
        approval.apply(
            manifest, Approval(approver="Ada", decision="approved", note="Cleared for paid social")
        )

        record = approval.read(manifest)
        assert record["decision"] == "approved"
        assert record["note"] == "Cleared for paid social"
        assert record["approved_at"].startswith("20")

    def test_existing_metadata_is_preserved(self) -> None:
        manifest = _manifest()
        manifest.run.metadata = {"client": "Acme"}

        approval.apply(manifest, Approval(approver="Ada", decision="approved"))

        assert manifest.run.metadata["client"] == "Acme"
        assert manifest.run.metadata["approval"]["approver"] == "Ada"


class TestLedgerSchema:
    def test_attempt_rows_match_the_schema(self) -> None:
        from hallmark.ledger import SCHEMA, Attempt, _table

        attempts = [
            Attempt(
                run_id="r1",
                campaign="coffee",
                modality="image",
                model="gpt-image-2-generate",
                provider="gmicloud",
                accepted=True,
                score=1.0,
                latency_seconds=3.2,
                cost_usd=0.01,
                sha256="a" * 64,
                size_bytes=1024,
                media_type="image/png",
                checks="[]",
            ),
            Attempt(
                run_id="r1",
                campaign="coffee",
                modality="image",
                model="gpt-image-2-generate",
                provider="gmicloud",
                accepted=False,
                score=0.66,
                latency_seconds=3.9,
                reject_reason="not_blank: pixel spread 2.1",
                cost_usd=0.01,
                sha256="b" * 64,
                size_bytes=900,
                media_type="image/png",
                checks="[]",
            ),
        ]

        table = _table(attempts)
        assert table.schema == SCHEMA
        assert table.num_rows == 2
        # Rejects are kept, which is the entire point of the ledger.
        assert table.column("accepted").to_pylist() == [True, False]
        assert table.column("reject_reason").to_pylist()[1].startswith("not_blank")
