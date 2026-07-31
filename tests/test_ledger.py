"""The route an attempt takes from the live page into the ledger.

The deployed function cannot write Parquet, so attempts made in public land in
storage as JSON and the publish step folds them in. That handover is the part
worth pinning: before it existed, the acceptance rate on the page counted only
the runs started from a terminal, while claiming to keep every attempt.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from hallmark import attempts, demo, ledger, storage

# The finished three-candidate session, already stubbed out for the selection
# tests. Reused rather than rebuilt so both files describe the same run.
from test_demo import fake_run  # noqa: F401


class FakeS3:
    """Enough of the S3 surface for a write, a listing and a delete."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {
                    "Contents": [
                        {"Key": key} for key in sorted(objects) if key.startswith(Prefix)
                    ]
                }

        return Paginator()


@pytest.fixture
def fake_storage(monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(storage, "client", lambda: s3)
    monkeypatch.setattr(storage, "bucket", lambda: "test-bucket")
    return s3


def _row(accepted: bool, score: float) -> dict:
    return {
        "run_id": "run-1",
        "campaign": "demo",
        "modality": "image",
        "model": "gpt-image-2-generate",
        "provider": "gmicloud",
        "accepted": accepted,
        "score": score,
        "latency_seconds": 4.0,
        "reject_reason": None if accepted else "the pose was better in the other one",
        "cost_usd": None,
        "sha256": "a" * 64,
        "size_bytes": 1234,
        "media_type": "image/jpeg" if accepted else "image/png",
        "checks": json.dumps([]),
    }


class TestPendingAttempts:
    def test_records_land_under_the_pending_prefix(self, fake_storage):
        key = attempts.record([_row(True, 1.0)])
        assert key is not None
        assert key.startswith(attempts.PENDING_PREFIX)
        assert list(fake_storage.objects) == [key]

    def test_an_empty_batch_writes_nothing(self, fake_storage):
        assert attempts.record([]) is None
        assert fake_storage.objects == {}

    def test_bookkeeping_never_breaks_the_caller(self, monkeypatch):
        """A visitor who has just approved an asset must not see a stack trace."""

        def boom():
            raise RuntimeError("storage is down")

        monkeypatch.setattr(storage, "client", boom)
        assert attempts.record([_row(True, 1.0)]) is None

    def test_draining_folds_them_into_parquet_and_clears_the_pending_copy(
        self, fake_storage
    ):
        attempts.record([_row(True, 1.0), _row(False, 0.4)])
        assert ledger.drain_pending() == 2

        keys = list(fake_storage.objects)
        assert not [k for k in keys if k.startswith(attempts.PENDING_PREFIX)]
        assert [k for k in keys if k.startswith(ledger.LEDGER_PREFIX)]

        rows = ledger.summary()
        assert len(rows) == 1
        assert rows[0]["attempts"] == 2
        assert rows[0]["accepted"] == 1

    def test_draining_nothing_is_not_an_error(self, fake_storage):
        assert ledger.drain_pending() == 0


class TestTheDemoRecordsWhatItGenerated:
    def test_every_candidate_reaches_the_ledger_including_the_rejects(
        self, tmp_path: Path, monkeypatch, fake_run
    ):
        recorded: list[list[dict]] = []
        monkeypatch.setattr(demo.attempts, "record", lambda rows: recorded.append(rows))
        monkeypatch.setattr(
            demo,
            "_rationales",
            lambda s, p, r: (demo._fallback_rationales(s, p, r), demo.REASON_FROM_TEMPLATE),
        )

        demo.select_candidate("sess-1", 1, "the light is better", tmp_path / "w",
                              signer="Ama")

        assert len(recorded) == 1
        rows = recorded[0]
        assert len(rows) == 3, "all three attempts, not just the one that won"
        assert [r["accepted"] for r in sorted(rows, key=lambda r: r["sha256"])].count(True) == 1

        chosen = [r for r in rows if r["accepted"]][0]
        rejects = [r for r in rows if not r["accepted"]]
        assert chosen["media_type"] == "image/jpeg"
        assert chosen["reject_reason"] is None
        assert all(r["reject_reason"] for r in rejects), "a reject records why it lost"
        assert all(r["campaign"] == "demo" for r in rows)
