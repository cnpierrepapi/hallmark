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
        "visible": safe(lambda: bool(metadata.read_visible(path)))
        if media_type.startswith("image/")
        else False,
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
        "marks": [MARK_LABELS[k] for k in ("credential", "visible", "pointer") if marks.get(k)],
        "missing": [MARK_LABELS[k] for k in ("credential", "visible", "pointer")
                    if k in marks and not marks.get(k)],
    }


def build(showcase: dict[str, Any]) -> dict[str, Any]:
    """Assemble the record for one campaign from its published payload."""
    gallery = showcase.get("gallery") or []
    rows = [asset_row(t) for t in gallery]

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
        "ledger": showcase.get("ledger") or [],
        "limits": LIMITS,
    }


def from_session(session: dict[str, Any], marks: dict[str, dict[str, bool]]) -> dict[str, Any]:
    """Assemble the record for one visitor's own run on the demo.

    Same document as a campaign gets, built from what the visitor actually
    generated, so the sheet is not something they have to take our word for
    from a run they never saw. The assets nobody picked are on it too, marked
    as carrying nothing, because they were never signed off and the record
    should say so rather than quietly leaving them out.
    """
    picked = (session.get("selection") or {}).get("picked")
    rows = []
    for c in session.get("candidates") or []:
        if not c.get("stored_key"):
            continue
        chosen = c.get("index") == picked
        rows.append(
            asset_row(
                {
                    "slug": c["stored_key"].rsplit("/", 1)[-1],
                    "title": ("Approved asset" if chosen else "Not selected")
                    + f", candidate {c.get('index')}",
                    "kind": "image",
                    "media_type": c.get("media_type", ""),
                    "model": c.get("model") or session.get("model", ""),
                    "sha256": c.get("sha256", ""),
                    "size_bytes": c.get("size_bytes", 0),
                    "manifest_uri": session.get("manifest_uri", "") if chosen else "",
                    "check": f"/api/demo/asset/{session.get('session_id')}/"
                             f"{c['stored_key'].rsplit('/', 1)[-1]}?download=1",
                    "check_label": "download this file",
                    "marks": marks.get(c["stored_key"], {}),
                }
            )
        )

    counted: dict[str, int] = {}
    for key, label in MARK_LABELS.items():
        counted[label] = sum(1 for m in marks.values() if m.get(key))

    selection = session.get("selection") or {}
    return {
        "run_id": session.get("run_id") or session.get("session_id", ""),
        "product": session.get("brief", ""),
        "audience": session.get("style_label", ""),
        "approval": {
            "approver": selection.get("signer"),
            "approved_at": selection.get("decided_at"),
            "note": selection.get("human_reason"),
            "decision": "approved" if picked is not None else None,
        },
        "manifest_uri": session.get("manifest_uri", ""),
        "canonical_hash": session.get("canonical_hash", ""),
        "statement": STATEMENT,
        "pipeline": PIPELINE,
        "assets": rows,
        "asset_count": len(rows),
        "marks_applied": counted,
        "ledger": [],
        "limits": LIMITS,
    }
