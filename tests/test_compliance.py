"""The marking record, and what it says about the attempts nobody approved.

The sheet is the one artefact here likely to be printed or attached to an
email, so what it leaves out matters as much as what it shows. A record
listing only the assets that shipped describes a pipeline with no misses,
which is never the pipeline anyone actually ran.
"""

from __future__ import annotations


class TestTheSheetShowsWhatWasRefused:
    """A record listing only what shipped reads as a pipeline that never misses."""

    def test_a_campaign_sheet_lists_the_refused_attempts_and_why(self) -> None:
        from hallmark import compliance, sheet

        record = compliance.build(
            {
                "run_id": "run-9",
                "product": "cold brew",
                "gallery": [],
                "attempts": [
                    {"modality": "image", "model": "gpt-image-2-generate",
                     "accepted": True, "score": 0.9, "reject_reason": None,
                     "sha256": "a" * 64, "size_bytes": 1024, "checks": []},
                    {"modality": "image", "model": "gpt-image-2-generate",
                     "accepted": False, "score": 0.4,
                     "reject_reason": "The reviewer wanted a warmer light.",
                     "sha256": "b" * 64, "size_bytes": 2048,
                     "checks": [{"name": "contrast", "passed": False, "detail": "flat"}]},
                ],
            }
        )

        assert record["reject_count"] == 1
        html = sheet.render(record)
        assert "Refused attempts (1)" in html
        assert "The reviewer wanted a warmer light." in html
        assert "contrast" in html, "a failed check is part of why it lost"

    def test_a_sheet_with_nothing_refused_says_so_rather_than_going_quiet(self) -> None:
        from hallmark import compliance, sheet

        html = sheet.render(compliance.build({"run_id": "run-0", "gallery": [],
                                              "attempts": []}))
        assert "Refused attempts (0)" in html
        assert "Nothing was refused" in html

    def test_a_session_sheet_carries_the_note_on_every_asset(self) -> None:
        from hallmark import compliance, sheet

        record = compliance.from_session(
            {"session_id": "sess-1"},
            [
                {
                    "run_seq": 1,
                    "kind": "image",
                    "kind_label": "Still image",
                    "model": "gpt-image-2-generate",
                    "selection": {"human_reason": "the stance is stronger",
                                  "signer": "Ama"},
                    "assets": [
                        {"index": 0, "accepted": True, "stored_key": "s/r1_chosen_0.jpg",
                         "media_type": "image/jpeg", "sha256": "a" * 64,
                         "size_bytes": 10, "marks": {}},
                        {"index": 1, "accepted": False, "stored_key": "s/r1_reject_1.png",
                         "media_type": "image/png", "sha256": "b" * 64, "size_bytes": 10,
                         "reason": "Lost on the reviewer's own test: the stance.",
                         "reason_source": "reviewer's reason, worded from a fixed template",
                         "marks": {}},
                    ],
                }
            ],
            {},
        )

        assert record["reject_count"] == 1
        refused = [a for a in record["assets"] if a["decision"] == "not selected"]
        assert refused and refused[0]["reason"].startswith("Lost on")

        html = sheet.render(record)
        # Apostrophes come back escaped, so the fragments checked here avoid them.
        assert "Lost on the reviewer" in html
        assert "worded from a fixed template" in html, "who wrote the note is part of it"
        # The rows are already in the asset table, so they are not printed twice.
        assert html.count("Lost on the reviewer") == 1
