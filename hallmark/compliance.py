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
