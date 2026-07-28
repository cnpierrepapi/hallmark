"""The human sign-off, recorded so it cannot be forged.

Nothing is stamped or published until a person approves it. The approval is
written into the run metadata and the manifest hash is then recomputed.

That ordering is the whole point. Run metadata is inside the canonical hash by
design, described in genblaze as user-supplied provenance rather than
operational noise. So an approval cannot be bolted onto a finished record
afterwards: change the approver, the timestamp or the note and the hash no
longer matches, and the file fails verification.

The published asset therefore carries proof of two separate things: which model
made it, and which human signed it off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from genblaze_core.models.manifest import Manifest

from hallmark import storage

APPROVAL_KEY = "approval"


@dataclass
class Approval:
    approver: str
    decision: str  # "approved" or "rejected"
    note: str | None = None
    approved_at: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "decision": self.decision,
            "note": self.note,
            "approved_at": self.approved_at
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }


def apply(manifest: Manifest, approval: Approval) -> Manifest:
    """Record an approval and rehash, so the sign-off is covered by the hash."""
    metadata = dict(manifest.run.metadata or {})
    metadata[APPROVAL_KEY] = approval.as_metadata()
    manifest.run.metadata = metadata
    manifest.compute_hash()
    return manifest


def read(manifest: Manifest) -> dict[str, Any] | None:
    return (manifest.run.metadata or {}).get(APPROVAL_KEY)


def publish_manifest(manifest: Manifest, run_id: str) -> str:
    """Store the approved manifest and point the record at it.

    Written under our own key rather than the one the sink used, because the
    hash changed when the approval was recorded and the sink's copy describes
    the state before sign-off.
    """
    key = f"campaigns/{run_id}/manifest.json"
    payload = manifest.to_canonical_json().encode("utf-8")

    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=key,
        Body=payload,
        ContentType="application/json",
    )

    uri = f"{storage.endpoint()}/{storage.bucket()}/{key}"
    manifest.manifest_uri = uri
    return uri


def load_published(run_id: str) -> Manifest | None:
    """Read back an approved manifest, for audit rather than verification."""
    from genblaze_core.models.manifest import parse_manifest

    key = f"campaigns/{run_id}/manifest.json"
    try:
        body = storage.client().get_object(Bucket=storage.bucket(), Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 - missing is a normal answer here
        return None
    return parse_manifest(json.loads(body))
