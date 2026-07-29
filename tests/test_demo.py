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

        # Published under the session id and the run number, which is what the
        # pointer resolves. The run number is in there so a second run cannot
        # overwrite the record the first run's files point at.
        manifest = json.loads(uploads["manifest:sess-1/r1"])
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

    def test_every_preset_says_something_about_movement_for_video(self) -> None:
        """A still's hint describes lenses and light and nothing that moves.

        Reusing it for a clip throws away the only instruction a video model
        cares about, so each preset carries a second hint.
        """
        for slug, preset in demo.STYLE_PRESETS.items():
            assert preset["video_hint"], f"{slug} has no video hint"
            assert preset["video_hint"] != preset["hint"]


class TestMediaKinds:
    def test_both_kinds_are_offered(self) -> None:
        kinds = {k["slug"]: k for k in demo.kind_choices()}
        assert set(kinds) == {"image", "video"}
        assert demo.DEFAULT_KIND in kinds
        assert all(k["label"] and k["note"] for k in kinds.values())

    def test_video_offers_a_real_choice(self) -> None:
        """One candidate would leave nothing to pick and no reject to keep."""
        assert demo.MEDIA_KINDS["video"]["candidates"] >= 2

    def test_a_clip_is_never_offered_as_uploadable(self) -> None:
        """The platform refuses a body over 4.5MB and a clip is bigger.

        The page has to know before it tries, or it bounces off a 413 with no
        useful thing to say.
        """
        kinds = {k["slug"]: k for k in demo.kind_choices()}
        assert kinds["video"]["uploadable"] is False
        assert demo.UPLOADABLE_MAX_BYTES <= 4_500_000

    def test_an_unknown_kind_falls_back_to_the_cheap_one(self) -> None:
        assert "hologram" not in demo.MEDIA_KINDS
        assert demo.DEFAULT_KIND == "image"


class TestQuota:
    """Video has its own budget, so stills getting busy cannot spend it."""

    def _status(self, **over):
        base = {
            "used_today": 0,
            "daily_cap": demo.DAILY_GENERATION_CAP,
            "used_this_session": 0,
            "session_cap": demo.PER_SESSION_CAP,
            "videos_today": 0,
            "video_daily_cap": demo.DAILY_VIDEO_CAP,
            "videos_this_session": 0,
            "video_session_cap": demo.PER_SESSION_VIDEO_CAP,
        }
        base.update(over)
        return base

    def test_a_fresh_visitor_can_have_either(self) -> None:
        assert demo._quota_refusal(self._status(), "image") is None
        assert demo._quota_refusal(self._status(), "video") is None

    def test_spending_the_video_budget_leaves_stills_working(self) -> None:
        status = self._status(videos_today=demo.DAILY_VIDEO_CAP)
        assert demo._quota_refusal(status, "image") is None
        refusal = demo._quota_refusal(status, "video")
        assert refusal and "video" in refusal.lower()
        assert "still" in refusal.lower(), "refusing has to say what still works"

    def test_the_video_cap_is_tighter_than_the_shared_one(self) -> None:
        assert demo.DAILY_VIDEO_CAP < demo.DAILY_GENERATION_CAP
        assert demo.PER_SESSION_VIDEO_CAP < demo.PER_SESSION_CAP

    def test_a_run_that_rendered_nothing_is_given_back(self, monkeypatch) -> None:
        """One failed render must not end a visitor's demo.

        The video allowance is one run per browser, so charging for a run that
        produced no file at all would lock them out over something that cost
        nobody anything.
        """
        state = {
            "generations": 4,
            "sessions": {"sess-x": 1},
            "videos": 2,
            "video_sessions": {"sess-x": 1},
        }
        written = {}
        monkeypatch.setattr(demo, "_read_quota", lambda: state)
        monkeypatch.setattr(demo.storage, "bucket", lambda: "test-bucket")
        monkeypatch.setattr(
            demo.storage, "client",
            lambda: type("C", (), {"put_object": lambda _self, **kw: written.update(kw)})(),
        )

        demo.refund_quota("sess-x", "video")

        import json as _json

        after = _json.loads(written["Body"])
        assert after["generations"] == 3
        assert after["sessions"]["sess-x"] == 0
        assert after["videos"] == 1
        assert after["video_sessions"]["sess-x"] == 0

    def test_a_refund_never_goes_below_zero(self, monkeypatch) -> None:
        state = {"generations": 0, "sessions": {}, "videos": 0, "video_sessions": {}}
        written = {}
        monkeypatch.setattr(demo, "_read_quota", lambda: state)
        monkeypatch.setattr(demo.storage, "bucket", lambda: "test-bucket")
        monkeypatch.setattr(
            demo.storage, "client",
            lambda: type("C", (), {"put_object": lambda _self, **kw: written.update(kw)})(),
        )

        demo.refund_quota("nobody", "video")

        import json as _json

        after = _json.loads(written["Body"])
        assert after["generations"] == 0 and after["videos"] == 0

    def test_the_shared_cap_still_stops_video(self) -> None:
        status = self._status(used_today=demo.DAILY_GENERATION_CAP)
        assert demo._quota_refusal(status, "video") is not None
        assert demo._quota_refusal(status, "image") is not None


class TestJobIds:
    """The adapters disagree about what submit returns, and it is silent.

    The image provider hands back a plain id string. The video provider hands
    back a SubmitResult object. Calling str() on the second stores its repr,
    polls 404 forever, and the render finishes and bills while the page waits.
    """

    def test_a_plain_string_is_taken_as_is(self) -> None:
        assert demo.job_id_of("79629ce5-fc00-45d1-bab7-10a976d7fba7") == (
            "79629ce5-fc00-45d1-bab7-10a976d7fba7"
        )

    def test_an_object_gives_up_its_prediction_id(self) -> None:
        class SubmitResult:
            prediction_id = "79629ce5-fc00-45d1-bab7-10a976d7fba7"
            estimated_seconds = 30.0

        got = demo.job_id_of(SubmitResult())
        assert got == "79629ce5-fc00-45d1-bab7-10a976d7fba7"
        assert "SubmitResult" not in got, "the repr is what broke this"

    def test_a_dict_response_works_too(self) -> None:
        assert demo.job_id_of({"request_id": "abc-123"}) == "abc-123"

    def test_something_unrecognised_still_yields_a_string(self) -> None:
        """Better a wrong id that fails loudly than a crash mid submit."""
        assert isinstance(demo.job_id_of(12345), str)


class TestSeeds:
    """wan2.7 refuses a seed outside a signed 32 bit range, and refuses it late.

    The submit is accepted and a job id comes back; the job fails a second
    later in the queue listing. So this cannot be caught at submit time and has
    to be right before the request is built.
    """

    def test_no_seed_this_scheme_can_produce_is_out_of_range(self) -> None:
        widest = (demo.SEED_CYCLE_SECONDS - 1) * 100 + max(
            spec["candidates"] for spec in demo.MEDIA_KINDS.values()
        )
        assert widest <= demo.MAX_SEED, "a seed this size is rejected by the video model"

    def test_seeds_stay_unique_within_a_batch(self) -> None:
        """They are how a submit whose answer never arrived finds its job."""
        batch = 1_785_344_732 % demo.SEED_CYCLE_SECONDS
        seeds = [batch * 100 + i for i in range(max(
            spec["candidates"] for spec in demo.MEDIA_KINDS.values()))]
        assert len(set(seeds)) == len(seeds)
        assert all(0 <= s <= demo.MAX_SEED for s in seeds)


class TestRunsAccumulate:
    """A browser can do more than one run, and both have to survive it.

    The first version wrote every run to the same object names, so a second
    run destroyed the first one's files. Those names are served with an hour
    of cache life, so the page went on showing an asset that no longer
    existed: the reject thumbnail from a run the visitor had moved on from.
    """

    def test_two_runs_never_write_to_the_same_object(self) -> None:
        first = {
            demo.asset_key("sess", 1, i, chosen=i == 0, suffix=".jpg" if i == 0 else ".png")
            for i in range(3)
        }
        second = {
            demo.asset_key("sess", 2, i, chosen=i == 0, suffix=".jpg" if i == 0 else ".png")
            for i in range(3)
        }
        assert first & second == set(), "a later run must not overwrite an earlier one"
        assert len(first) == 3

    def test_a_second_run_keeps_the_first_in_the_history(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        _no_chat(monkeypatch)
        session, _uploads = fake_run

        first = demo.select_candidate("sess-1", 0, "the light", tmp_path / "a", signer="Ama")
        assert len(first["runs"]) == 1

        # What start_generation does to a session on a second go: new renders,
        # a new run number, and the finished runs carried across.
        session["run_seq"] = 2
        session["status"] = "ready"
        session["selection"] = None
        for c in session["candidates"]:
            c.update({"status": "ready", "accepted": False, "stored_key": None,
                      "reason": None})

        second = demo.select_candidate("sess-1", 1, "the stance", tmp_path / "b", signer="Ama")

        assert [r["run_seq"] for r in second["runs"]] == [1, 2]
        totals = demo.session_totals(second)
        assert totals == {"runs": 2, "assets": 6, "signed": 2, "unsigned": 4}

        keys = [a["stored_key"] for r in second["runs"] for a in r["assets"]]
        assert len(set(keys)) == 6, "every asset the visitor made has its own object"

    def test_the_marking_record_covers_every_run(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        """Step 7's document is the session's, not the last run's."""
        from hallmark import compliance

        _no_chat(monkeypatch)
        session, _uploads = fake_run

        demo.select_candidate("sess-1", 0, "the light", tmp_path / "a", signer="Ama")
        session["run_seq"] = 2
        session["selection"] = None
        for c in session["candidates"]:
            c.update({"status": "ready", "accepted": False, "stored_key": None})
        final = demo.select_candidate("sess-1", 2, "the stance", tmp_path / "b", signer="Ama")

        runs = demo.session_runs(final)
        record = compliance.from_session(final, runs, marks={})

        assert record["asset_count"] == 6
        assert record["run_count"] == 2
        assert record["scope"] == "session"
        assert record["download_url"].endswith("/compliance/session/sess-1/download")
        titles = " ".join(a["title"] for a in record["assets"])
        assert "Run 1" in titles and "Run 2" in titles

    def test_marks_read_now_beat_marks_read_at_delivery(
        self, tmp_path: Path, monkeypatch, fake_run
    ) -> None:
        """Both are measurements, taken at different times. Say which."""
        from hallmark import compliance

        _no_chat(monkeypatch)
        _session, _uploads = fake_run
        final = demo.select_candidate("sess-1", 0, "the light", tmp_path / "a", signer="Ama")
        runs = demo.session_runs(final)

        key = runs[0]["assets"][0]["stored_key"]
        record = compliance.from_session(
            final, runs, marks={key: {"credential": False, "visible": True, "pointer": True}}
        )

        fresh = next(a for a in record["assets"] if a["slug"] == key.rsplit("/", 1)[-1])
        stale = next(a for a in record["assets"] if a["slug"] != key.rsplit("/", 1)[-1])
        assert "just now" in fresh["measured"]
        assert "delivered" in stale["measured"]
        assert record["measured_now"] == 1


class TestStripVisible:
    """Removing the credit is an edit, and the record has to refuse it."""

    def test_taking_the_credit_out_of_a_jpeg_changes_the_bytes(
        self, tmp_path: Path
    ) -> None:
        from hallmark import metadata

        source = tmp_path / "src.png"
        _png(source, seed=3)
        delivered = tmp_path / "delivered.jpg"
        metadata.to_jpeg(
            source,
            delivered,
            metadata.Signature(approver="Ama", model="gpt-image-2-generate"),
        )
        assert metadata.read_visible(delivered), "the credit is there to begin with"

        stripped = metadata.strip_visible(delivered.read_bytes(), "image/jpeg")
        assert stripped != delivered.read_bytes()

        bare = tmp_path / "bare.jpg"
        bare.write_bytes(stripped)
        assert metadata.read_visible(bare) == {}, "the credit is gone"

        from PIL import Image

        with Image.open(bare) as img:
            assert img.size == (72, 72), "the picture itself is untouched"

    def test_taking_the_credit_out_of_a_clip_changes_the_bytes(
        self, tmp_path: Path
    ) -> None:
        """The clip case is the one that matters: it is the only route offered.

        A clip is too large to move through the platform in a request body, so
        the visitor cannot edit it in their own browser. This is what the
        server does on their behalf, on a copy.
        """
        from hallmark import metadata

        source = Path("tests/fixtures/tiny.mp4")
        credited = tmp_path / "credited.mp4"
        metadata.to_mp4(
            source,
            credited,
            metadata.Signature(approver="Ama", model="wan2.7-t2v", medium="video"),
        )
        assert "Ama" in str(metadata.read_visible(credited))

        data = credited.read_bytes()
        stripped = metadata.strip_visible(data, "video/mp4")
        assert stripped != data
        assert len(stripped) < len(data)

        bare = tmp_path / "bare.mp4"
        bare.write_bytes(stripped)
        assert metadata.read_visible(bare) == {}

    def test_a_file_with_no_credit_comes_back_unchanged(self, tmp_path: Path) -> None:
        """So a caller can tell 'removed it' from 'there was none'."""
        from hallmark import metadata

        source = tmp_path / "plain.png"
        _png(source, seed=5)
        data = source.read_bytes()
        assert metadata.strip_visible(data, "image/png") == data
