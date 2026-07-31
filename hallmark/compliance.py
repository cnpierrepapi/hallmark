"""A campaign's marking record, in a form a person can audit.

From 2 August 2026, Article 50 of the EU AI Act requires providers of systems
generating synthetic image, audio, video or text to mark those outputs in a
machine readable way. A line of small print is not a mark. It does not survive
a re-upload, it cannot be checked by a machine, and it is not attached to the
thing it describes.

So this sheet does not assert compliance. It sets out, per asset, exactly which
marks were applied, what each one is, and where to check it, then leaves the
reading to whoever is doing the reading. Every hash on the page can be
recomputed from a file, and every asset links to the checker so nothing here
has to be taken on trust.

Three marks are applied, and they fail in different ways on purpose:

    credential   a signed C2PA manifest, carrying the parent it came from.
                 The strongest of the three and the one the regulation is
                 converging on. Stripped by most social platforms.
    visible      EXIF and XMP fields an operating system will display, so the
                 disclosure survives being handed to a person rather than a
                 verifier. Written before hashing, so editing it is detectable.
    pointer      our own record, resolved against the bucket, which is what
                 proves the bytes have not moved since sign off.

No single one of them is sufficient, which is why all three are written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# What the sheet says about the pipeline, in the words the vendors use.
PIPELINE = [
    ("Generation", "GMI Cloud", "The models that produced the assets."),
    ("Orchestration", "Genblaze", "Records each step and the manifest covering it."),
    ("Storage", "Backblaze B2", "Holds the assets and the records they resolve against."),
]

MARK_LABELS = {
    "credential": "C2PA Content Credential",
    "visible": "Embedded file properties",
    "pointer": "HALLMARK provenance record",
}

STATEMENT = (
    "From 2 August 2026, providers of systems that generate synthetic audio, image, "
    "video or text must mark those outputs in a machine readable way under Article 50 "
    "of the EU AI Act. A line of small print is not a mark, and it does not survive a "
    "re-upload. Generated through GMI Cloud, orchestrated with Genblaze, stored on "
    "Backblaze B2."
)

# Said plainly, because a sheet that overclaims is worse than no sheet.
LIMITS = [
    "This is a record of the marks applied to these files. It is not a legal opinion "
    "and it does not certify compliance on anyone's behalf.",
    "Marks describe how a file was made. They cannot establish that what it shows is "
    "true, and nothing here has examined the content of any asset.",
    "The judgement behind every refusal is the reviewer's. Where a note says it was "
    "worded by a language model, the model was given the reviewer's stated reason and "
    "the measured checks, and has never seen the asset it is writing about.",
    "Metadata is removed by most platforms when a file is uploaded. An asset that "
    "loses its marks downstream is still covered by the record it was published with, "
    "which is why the hashes below are the durable part.",
]


def marks_for(path: Path, media_type: str) -> dict[str, bool]:
    """Which marks a file actually carries, read off the file itself.

    Measured, never assumed. A sheet reporting what the pipeline was supposed
    to do is worth nothing: the clips shipped without credentials once already,
    silently, because a parent was offered to the signer under the wrong media
    type. One definition, shared by the publisher and by the live sheet, so the
    two can never report different marks for the same asset.
    """
    from hallmark import integrity, metadata, provenance

    def safe(fn: Any, default: bool = False) -> bool:
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 - an unreadable mark is a missing mark
            return default

    return {
        "credential": safe(lambda: provenance.read_c2pa(path).present),
        # Clips carry a readable credit too now, so this is asked of the file
        # rather than decided from its type. Deciding from the type is how the
        # sheet went on reporting nine assets with six credits after every one
        # of them had been written.
        "visible": safe(lambda: bool(metadata.read_visible(path))),
        "pointer": safe(lambda: bool(integrity.extract_embedded_json(path, media_type))),
    }


def asset_row(tile: dict[str, Any]) -> dict[str, Any]:
    """One asset, reduced to what an auditor would want to see."""
    marks = tile.get("marks") or {}
    return {
        "slug": tile["slug"],
        "title": tile.get("title") or tile["slug"],
        "kind": tile.get("kind", "image"),
        "media_type": tile.get("media_type", ""),
        "model": tile.get("model", ""),
        "sha256": tile.get("sha256", ""),
        "size_bytes": tile.get("size_bytes", 0),
        "manifest_uri": tile.get("manifest_uri", ""),
        # Where a reader goes to check this particular asset for themselves.
        # Campaign assets get re-checked in place; a visitor's own assets are
        # offered back to them, since they are the one holding the copy.
        "check": tile.get("check", ""),
        "check_label": tile.get("check_label", ""),
        # What review decided about this file and why. An asset row that says
        # "not selected" and stops there tells a reader the pipeline had a
        # choice and hides the only part that explains it.
        "decision": tile.get("decision", ""),
        "reason": tile.get("reason", ""),
        "reason_source": tile.get("reason_source", ""),
        "marks": [MARK_LABELS[k] for k in ("credential", "visible", "pointer") if marks.get(k)],
        "missing": [MARK_LABELS[k] for k in ("credential", "visible", "pointer")
                    if k in marks and not marks.get(k)],
    }


def reject_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The attempts nobody approved, with the note filed against each.

    A campaign's asset table lists what shipped, so on its own it reads as a
    pipeline that never misses. The refused attempts are the half of the record
    that shows the review was real, and the reason is the whole of their value:
    a row saying only "not selected" is a count, not an audit trail.
    """
    rows = []
    for attempt in attempts:
        if attempt.get("accepted"):
            continue
        failed = [
            check.get("name", "")
            for check in attempt.get("checks") or []
            if not check.get("passed")
        ]
        rows.append(
            {
                "modality": attempt.get("modality", ""),
                "model": attempt.get("model", ""),
                "reason": attempt.get("reject_reason") or "No note recorded.",
                "score": attempt.get("score"),
                "sha256": attempt.get("sha256") or "",
                "size_bytes": attempt.get("size_bytes") or 0,
                "failed_checks": failed,
            }
        )
    return rows


def build(showcase: dict[str, Any]) -> dict[str, Any]:
    """Assemble the record for one campaign from its published payload."""
    gallery = showcase.get("gallery") or []
    rows = [asset_row(t) for t in gallery]
    refused = reject_rows(showcase.get("attempts") or [])

    counted: dict[str, int] = {}
    for key, label in MARK_LABELS.items():
        counted[label] = sum(1 for t in gallery if (t.get("marks") or {}).get(key))

    return {
        "run_id": showcase.get("run_id", ""),
        "product": showcase.get("product", ""),
        "audience": showcase.get("audience", ""),
        "approval": showcase.get("approval") or {},
        "manifest_uri": showcase.get("manifest_uri", ""),
        "canonical_hash": showcase.get("canonical_hash", ""),
        "statement": STATEMENT,
        "pipeline": PIPELINE,
        "assets": rows,
        "asset_count": len(rows),
        "marks_applied": counted,
        "rejected": refused,
        "reject_count": len(refused),
        "ledger": showcase.get("ledger") or [],
        "limits": LIMITS,
    }


def from_session(
    session: dict[str, Any],
    runs: list[dict[str, Any]],
    marks: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """Assemble the record for everything one browser has generated.

    Same document as a campaign gets, built from what the visitor actually
    generated, so the sheet is not something they have to take our word for
    from a run they never saw. The assets nobody picked are on it too, marked
    as carrying nothing, because they were never signed off and the record
    should say so rather than quietly leaving them out.

    The scope is the browser session, not one run. Somebody who generates
    stills, then a clip, then stills again is holding six or nine files and
    needs one document covering all of them; a record that silently dropped
    everything but the last run would be worse than none, because it would look
    complete.

    ``marks`` holds what was measured by reading the files during this request.
    Anything absent from it falls back to what was measured at delivery, and
    the row says which of the two it is rather than presenting them as the same
    thing. Re-reading every file of every run inside a function that dies at 60
    seconds is not something a clip-sized asset survives.
    """
    rows = []
    measured_now = 0
    for run in runs:
        seq = run.get("run_seq")
        label = run.get("kind_label") or ("Video clip" if run.get("kind") == "video"
                                          else "Still image")
        for asset in run.get("assets") or []:
            key = asset.get("stored_key")
            if not key:
                continue
            name = key.rsplit("/", 1)[-1]
            chosen = bool(asset.get("accepted"))
            fresh = key in marks
            measured_now += 1 if fresh else 0
            row = asset_row(
                {
                    "slug": name,
                    "title": (
                        f"Run {seq} · {label} · "
                        + ("approved" if chosen else "not selected")
                        + f", candidate {asset.get('index')}"
                    ),
                    "kind": run.get("kind", "image"),
                    "media_type": asset.get("media_type", ""),
                    "model": run.get("model", "") or session.get("model", ""),
                    "sha256": asset.get("sha256", ""),
                    "size_bytes": asset.get("size_bytes", 0),
                    "manifest_uri": run.get("manifest_uri", "") if chosen else "",
                    "check": f"/api/demo/asset/{session.get('session_id')}/{name}?download=1",
                    "check_label": "download this file",
                    "marks": marks.get(key) or asset.get("marks") or {},
                    "decision": "approved" if chosen else "not selected",
                    "reason": (
                        (run.get("selection") or {}).get("human_reason") or ""
                        if chosen
                        else asset.get("reason") or ""
                    ),
                    "reason_source": (
                        "written by the reviewer" if chosen
                        else asset.get("reason_source") or ""
                    ),
                }
            )
            row["measured"] = "read from the file just now" if fresh else (
                "read from the file when it was delivered"
            )
            rows.append(row)

    counted: dict[str, int] = {}
    for key, mark_label in MARK_LABELS.items():
        counted[mark_label] = sum(
            1
            for run in runs
            for asset in run.get("assets") or []
            if (marks.get(asset.get("stored_key")) or asset.get("marks") or {}).get(key)
        )

    # The newest run is the one the page just walked through, so its approval
    # is the one to head the document with.
    latest = runs[-1] if runs else {}
    selection = latest.get("selection") or {}
    briefs = [r.get("brief", "") for r in runs if r.get("brief")]

    return {
        "run_id": session.get("session_id", ""),
        "scope": "session",
        "run_count": len(runs),
        "measured_now": measured_now,
        "product": "; ".join(dict.fromkeys(briefs)) or session.get("brief", ""),
        "audience": latest.get("style_label", "") or session.get("style_label", ""),
        "approval": {
            "approver": selection.get("signer"),
            "approved_at": selection.get("decided_at"),
            "note": selection.get("human_reason"),
            "decision": "approved" if selection.get("picked") is not None else None,
        },
        "manifest_uri": latest.get("manifest_uri", ""),
        "canonical_hash": latest.get("canonical_hash", ""),
        "download_url": f"/compliance/session/{session.get('session_id', '')}/download",
        "statement": STATEMENT,
        "pipeline": PIPELINE,
        "assets": rows,
        "asset_count": len(rows),
        "marks_applied": counted,
        # A session's refused attempts are rows in the asset table already,
        # each carrying its own note, so repeating them below would be the same
        # information twice. The count still heads the section.
        "rejected": [],
        "reject_count": sum(1 for row in rows if row.get("decision") == "not selected"),
        "ledger": [],
        "limits": LIMITS,
    }
