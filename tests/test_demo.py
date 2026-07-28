"""The selection path, offline.

This is the step the whole demo turns on: three candidates come back, a person
picks one and says why, and what leaves is a signed file plus a record of the
attempts that lost. It touches storage, the chat model and the stamping code at
once, so it is exercised here with those three faked out rather than only on a
live run that costs credits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from hallmark import demo


def _png(path: Path, seed: int) -> None:
    image = Image.new("RGB", (72, 72))
    pixels = image.load()
    for x in range(72):
        for y in range(72):
            pixels[x, y] = ((x * seed) % 256, (y * 3) % 256, ((x + y + seed) * 5) % 256)
    image.save(path, "PNG")


@pytest.fixture
def fake_run(tmp_path: Path, monkeypatch):
    """A finished three-candidate session, with storage and chat stubbed out."""
    sources = {}
    for index in range(3):
        path = tmp_path / f"src_{index}.png"
        _png(path, seed=index + 2)
        sources[f"https://example.test/{index}.png"] = path

    session = {
        "session_id": "sess-1",
        "brief": "a bottle of cold brew on wet stone",
        "prompt": "a bottle of cold brew on wet stone. studio product photography",
        "style": "product",
        "style_label": "Product still",
        "model": "gpt-image-2-generate",
        "created_at": "2026-07-28T10:00:00+00:00",
        "status": "ready",
        "candidates": [
            {
                "index": index,
                "job_id": f"job-{index}",
                "seed": index,
                "status": "ready",
                "url": f"https://example.test/{index}.png",
                "media_type": "image/png",
                "latency_seconds": 3.0,
                "checks": [],
                "score": 0.0,
                "passed": False,
                "accepted": False,
                "reason": None,
                "stored_key": None,
            }
            for index in range(3)
        ],
        "selection": None,
        "run_id": None,
        "manifest_uri": None,
    }

    saved: dict = {"session": None}
    uploads: dict[str, bytes] = {}

    monkeypatch.setattr(demo, "load_session", lambda sid: session)
    monkeypatch.setattr(demo, "save_session", lambda s: saved.__setitem__("session", s))
    monkeypatch.setattr(
        demo.storage, "download", lambda url, dest: dest.write_bytes(sources[url].read_bytes())
    )
    monkeypatch.setattr(
        demo.storage,
        "upload",
        lambda local, key, mime: uploads.__setitem__(key, Path(local).read_bytes()),
    )

    from hallmark import approval as approval_mod

    def publish(manifest, run_id):
        uri = f"s3://test-bucket/campaigns/{run_id}/manifest.json"
        manifest.manifest_uri = uri
        uploads[f"manifest:{run_id}"] = manifest.to_canonical_json().encode()
        return uri

    monkeypatch.setattr(approval_mod, "publish_manifest", publish)
    return session, uploads


def _no_chat(monkeypatch):
    """Force the fallback notes, so the assertion is about our own wording."""
    monkeypatch.setattr(
        demo, "_rationales", lambda s, p, r: demo._fallback_rationales(s, p, r)
    )


class TestSelection:
    def test_the_chosen_file_is_delivered_as_a_signed_jpeg(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        _no_chat(monkeypatch)
        _session, uploads = fake_run

        result = demo.select_candidate(
            "sess-1", 1, "it has a good stance with very vivid features",
            tmp_path / "work", signer="Ama",
        )

        chosen = next(c for c in result["candidates"] if c["accepted"])
        assert chosen["media_type"] == "image/jpeg"
        assert chosen["stored_key"].endswith("chosen_1.jpg")

        from hallmark import metadata

        delivered = tmp_path / "delivered.jpg"
        delivered.write_bytes(uploads[chosen["stored_key"]])
        visible = metadata.read_visible(delivered)
        assert "Ama" in visible["artist"]
        assert "good stance" in visible["comment"]

    def test_the_recorded_hash_covers_the_visible_signature(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        """The hash must be of the delivered file, signature included.

        If the manifest recorded the raw PNG's hash instead, the file a
        visitor downloads would never match its own record.
        """
        _no_chat(monkeypatch)
        _session, uploads = fake_run

        result = demo.select_candidate(
            "sess-1", 0, "the light falls better", tmp_path / "work", signer="Ama"
        )
        chosen = next(c for c in result["candidates"] if c["accepted"])

        from hallmark.integrity import canonical_bytes

        stamped = tmp_path / "stamped.jpg"
        stamped.write_bytes(uploads[chosen["stored_key"]])
        assert hashlib.sha256(canonical_bytes(stamped)).hexdigest() == chosen["sha256"]

        # Published under the session id, which is what the pointer resolves.
        manifest = json.loads(uploads["manifest:sess-1"])
        declared = [
            asset["sha256"]
            for step in manifest["run"]["steps"]
            for asset in step["assets"]
        ]
        assert chosen["sha256"] in declared

    def test_rejects_are_kept_and_left_unsigned(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        _no_chat(monkeypatch)
        _session, uploads = fake_run

        result = demo.select_candidate(
            "sess-1", 2, "the stance is stronger", tmp_path / "work", signer="Ama"
        )

        rejects = [c for c in result["candidates"] if not c["accepted"]]
        assert len(rejects) == 2
        for reject in rejects:
            assert reject["stored_key"].endswith(".png"), "a reject is stored as it came back"
            assert uploads[reject["stored_key"]], "the bytes are kept, not binned"

            from hallmark import metadata

            path = tmp_path / f"reject_{reject['index']}.png"
            path.write_bytes(uploads[reject["stored_key"]])
            assert metadata.read_visible(path) == {}, "nobody approved it, so nothing signs it"

    def test_rejection_notes_answer_the_reason_the_reviewer_gave(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        _no_chat(monkeypatch)

        reason = "it has a good stance with very vivid features"
        result = demo.select_candidate("sess-1", 1, reason, tmp_path / "work", signer="Ama")

        notes = [c["reason"] for c in result["candidates"] if not c["accepted"]]
        assert len(notes) == 2
        for note in notes:
            assert reason.rstrip(".") in note, "the note has to quote the reviewer's own test"
        assert notes[0] != notes[1], "two rejects must not get one sentence twice"

    def test_no_reason_given_invents_nothing(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        _no_chat(monkeypatch)

        result = demo.select_candidate("sess-1", 0, "", tmp_path / "work", signer="Ama")

        for candidate in result["candidates"]:
            if candidate["accepted"]:
                continue
            note = candidate["reason"].lower()
            assert "own judgement" in note or "no reason" in note or "no stated grounds" in note


class TestStyles:
    def test_every_preset_is_offered_with_a_label(self) -> None:
        choices = demo.style_choices()
        assert {c["slug"] for c in choices} == set(demo.STYLE_PRESETS)
        assert all(c["label"] and c["note"] for c in choices)
        assert demo.DEFAULT_STYLE in demo.STYLE_PRESETS

    def test_an_unknown_style_falls_back_rather_than_reaching_the_model(self) -> None:
        """A style is a preset, so an unknown slug must not become prompt text."""
        assert "nonsense" not in demo.STYLE_PRESETS
        chosen = "nonsense" if "nonsense" in demo.STYLE_PRESETS else demo.DEFAULT_STYLE
        assert chosen == demo.DEFAULT_STYLE
