"""Turn a media file into a provenance verdict a human can act on.

Two things have to be resolved before a verdict exists.

First, what is embedded. A full manifest carries everything. A pointer carries
only ``{schema_version, canonical_hash, manifest_uri}``, keeping the prompt and
parameters off the file so an asset can be published without publishing the
creative process. Pointers are resolved by fetching the manifest from storage.

Second, whether the pointer can be trusted. A pointer names its own manifest,
so a forger could rewrite it to reference a manifest that does describe their
altered file. The fetched manifest's canonical hash is therefore checked
against the hash embedded in the file, and a mismatch is rejected outright.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from genblaze_core.media.embedder import guess_mime
from genblaze_core.models.enums import PromptVisibility
from genblaze_core.models.manifest import Manifest, parse_manifest

from hallmark.integrity import (
    UnsupportedMediaError,
    canonical_sha256,
    extract_embedded_json,
)

VERIFIED = "verified"
ALTERED = "altered"
UNSIGNED = "unsigned"
BROKEN = "broken"

_VERDICT_SUMMARY = {
    VERIFIED: "This file is unaltered since it was generated.",
    ALTERED: "This file carries a valid record, but the media has been changed since.",
    UNSIGNED: "This file carries no provenance record.",
    BROKEN: "This file's provenance record failed its own integrity check.",
}


class ManifestResolutionError(Exception):
    """Raised when a pointer manifest cannot be resolved or fails its check."""


@dataclass
class ProvenanceStep:
    provider: str | None
    model: str
    modality: str
    step_type: str
    prompt: str | None
    prompt_withheld: bool
    seed: int | None
    params: dict[str, Any]


@dataclass
class VerificationResult:
    verdict: str
    summary: str
    media_type: str
    computed_sha256: str
    declared_sha256: str | None
    manifest_source: str
    reason: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    created_at: datetime | None = None
    canonical_hash: str | None = None
    schema_version: str | None = None
    steps: list[ProvenanceStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "reason": self.reason,
            "media_type": self.media_type,
            "computed_sha256": self.computed_sha256,
            "declared_sha256": self.declared_sha256,
            "manifest_source": self.manifest_source,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "canonical_hash": self.canonical_hash,
            "schema_version": self.schema_version,
            "steps": [
                {
                    "provider": step.provider,
                    "model": step.model,
                    "modality": step.modality,
                    "step_type": step.step_type,
                    "prompt": step.prompt,
                    "prompt_withheld": step.prompt_withheld,
                    "seed": step.seed,
                    "params": step.params,
                }
                for step in self.steps
            ],
        }


def _s3_client():
    """Reuse the shared, retrying client rather than building one per call."""
    from hallmark import storage

    return storage.client()


def _fetch_manifest(manifest_uri: str) -> Manifest:
    """Fetch a manifest from B2 by URI.

    Accepts both ``s3://bucket/key`` and the https endpoint form, since
    genblaze records whichever the storage backend produced.
    """
    if manifest_uri.startswith("s3://"):
        without_scheme = manifest_uri[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
    else:
        endpoint = os.environ.get("B2_ENDPOINT", "")
        if not endpoint or not manifest_uri.startswith(endpoint):
            raise ManifestResolutionError(
                f"Manifest URI is not in a location this verifier can read: {manifest_uri}"
            )
        remainder = manifest_uri[len(endpoint) :].lstrip("/")
        bucket, _, key = remainder.partition("/")

    if not bucket or not key:
        raise ManifestResolutionError(f"Could not parse a bucket and key from {manifest_uri}")

    try:
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        raise ManifestResolutionError(f"Could not fetch manifest: {exc}") from exc

    import json

    return parse_manifest(json.loads(body))


def _resolve(path: Path, mime: str) -> tuple[Manifest | None, str]:
    """Return the manifest for a file and where it came from."""
    embedded = extract_embedded_json(path, mime)
    if embedded is None:
        return None, "none"

    if "run" in embedded:
        return parse_manifest(embedded), "embedded"

    manifest_uri = embedded.get("manifest_uri")
    if not manifest_uri:
        raise ManifestResolutionError(
            "Embedded block is neither a manifest nor a pointer with a manifest_uri"
        )

    manifest = _fetch_manifest(manifest_uri)

    # The pointer names its own manifest, so an attacker could repoint it at a
    # manifest that legitimately describes their altered file. Binding the
    # fetched manifest to the hash embedded in the media closes that.
    pointer_hash = embedded.get("canonical_hash")
    if pointer_hash and manifest.canonical_hash != pointer_hash:
        raise ManifestResolutionError(
            "Pointer hash does not match the manifest it references. "
            f"File says {pointer_hash[:16]}..., manifest is {manifest.canonical_hash[:16]}..."
        )

    return manifest, f"fetched:{manifest_uri}"


def _steps_from(manifest: Manifest) -> list[ProvenanceStep]:
    steps = []
    for step in manifest.run.steps:
        withheld = step.prompt_visibility != PromptVisibility.PUBLIC or step.prompt is None
        steps.append(
            ProvenanceStep(
                provider=step.provider,
                model=step.model,
                modality=step.modality.value,
                step_type=step.step_type.value,
                prompt=None if withheld else step.prompt,
                prompt_withheld=withheld,
                seed=step.seed,
                params=step.params or {},
            )
        )
    return steps


def verify(path: Path, mime_type: str | None = None) -> VerificationResult:
    """Produce a full verification result for a media file."""
    mime = mime_type or guess_mime(path)

    def unsigned(reason: str) -> VerificationResult:
        return VerificationResult(
            verdict=UNSIGNED,
            summary=_VERDICT_SUMMARY[UNSIGNED],
            reason=reason,
            media_type=mime,
            computed_sha256="",
            declared_sha256=None,
            manifest_source="none",
        )

    try:
        manifest, source = _resolve(path, mime)
    except UnsupportedMediaError as exc:
        return unsigned(str(exc))
    except ManifestResolutionError as exc:
        return VerificationResult(
            verdict=BROKEN,
            summary=_VERDICT_SUMMARY[BROKEN],
            reason=str(exc),
            media_type=mime,
            computed_sha256="",
            declared_sha256=None,
            manifest_source="unresolved",
        )
    except Exception as exc:  # noqa: BLE001 - malformed input must not 500
        return unsigned(f"Could not read a provenance record: {exc}")

    if manifest is None:
        return unsigned("No provenance record is embedded in this file")

    computed = canonical_sha256(path, mime)
    report = manifest.verification_report()

    declared = [
        asset.sha256
        for step in manifest.run.steps
        for asset in step.assets
        if asset.sha256
    ]
    matched = computed if computed in declared else (declared[-1] if declared else None)

    if not report.hash_ok:
        verdict = BROKEN
        reason = "The provenance record does not match its own contents"
    elif report.unverified_sha256_ids:
        verdict = BROKEN
        reason = "The provenance record contains outputs with no valid hash"
    elif matched is None:
        verdict = BROKEN
        reason = "The provenance record declares no output hashes to check against"
    elif matched != computed:
        verdict = ALTERED
        reason = "The media does not match the hash recorded when it was generated"
    else:
        verdict = VERIFIED
        reason = None

    return VerificationResult(
        verdict=verdict,
        summary=_VERDICT_SUMMARY[verdict],
        reason=reason,
        media_type=mime,
        computed_sha256=computed,
        declared_sha256=matched,
        manifest_source=source,
        run_id=manifest.run.run_id,
        run_name=manifest.run.name,
        created_at=manifest.run.created_at,
        canonical_hash=manifest.canonical_hash,
        schema_version=manifest.schema_version,
        steps=_steps_from(manifest),
    )
