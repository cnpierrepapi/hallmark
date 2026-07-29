"""Everything a file will say about how it was made, from every source we can read.

The public checker used to answer one question: is there a HALLMARK record in
here. For a file we made that is a real answer. For a file handed to a newsroom
by a stranger it is close to useless, because the only verdict it can ever
reach is "unsigned", which reads as "nothing here" when the truth is often
"nothing of ours here".

That was measurably wrong rather than merely narrow. The images this pipeline
generates arrive carrying a signed Adobe credential naming the model that made
them, and the checker walked straight past it and called them unsigned.

So a file is now read three ways, and each is reported on its own terms:

    our record   the pointer we embed, resolved against the bucket
    C2PA         a signed credential from whoever made or edited the file
    claims       IPTC and XMP fields asserting a source type, with no signature

They are never collapsed into a single "this is AI" answer, because no honest
tool can produce one. A signature proves who vouched for a file and that the
bytes still match what they vouched for. It cannot prove the vouching party was
truthful, and its absence proves nothing at all: every social platform strips
this metadata on upload, so the most common reason a real photograph has no
credential is that someone posted it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The IPTC vocabulary the AI disclosure rules are written against. Anything
# claiming one of these is claiming the content was machine generated.
SYNTHETIC_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeSynthetic",
    "algorithmicMedia",
}

# C2PA reports each check by code. These are the ones worth saying out loud.
_UNTRUSTED = "signingCredential.untrusted"
_TRUSTED = "signingCredential.trusted"

# The C2PA trust list, vendored so verification does not depend on reaching
# another host mid-request. Without it every credential comes back untrusted,
# including Adobe's, because nothing on the machine says whose certificates
# count. Refreshed from contentcredentials.org/trust.
TRUST_DIR = Path(__file__).resolve().parent / "trust"

_trust_loaded = False


def _load_trust() -> bool:
    """Point the verifier at the published trust list, once per process."""
    global _trust_loaded
    if _trust_loaded:
        return True
    try:
        import warnings

        from c2pa import load_settings

        settings = {
            "trust": {
                "trust_anchors": (TRUST_DIR / "anchors.pem").read_text(encoding="utf-8"),
                "allowed_list": (TRUST_DIR / "allowed.sha256.txt").read_text(encoding="utf-8"),
                "trust_config": (TRUST_DIR / "store.cfg").read_text(encoding="utf-8"),
            },
            "verify": {"verify_trust": True},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_settings(json.dumps(settings))
        _trust_loaded = True
    except Exception:  # noqa: BLE001 - unverified trust beats no answer at all
        return False
    return True


@dataclass
class Finding:
    """What one source of provenance says about a file."""

    source: str
    present: bool
    signed: bool = False
    trusted: bool | None = None
    says: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "present": self.present,
            "signed": self.signed,
            "trusted": self.trusted,
            "says": self.says,
            "detail": self.detail,
        }


def _actions(manifest: dict) -> list[dict]:
    """Flatten the action assertions, whichever schema version wrote them."""
    out: list[dict] = []
    for assertion in manifest.get("assertions") or []:
        if not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        out.extend((assertion.get("data") or {}).get("actions") or [])
    return out


def _source_type(actions: list[dict]) -> str | None:
    """The IPTC digital source type, wherever the writer chose to put it.

    Version 2 of the actions assertion puts it on the action itself. Adobe
    writes it as a vendor parameter instead. Reading only one of the two means
    missing the declaration on roughly half of what arrives.
    """
    for action in actions:
        candidates = dict(action.get("parameters") or {})
        candidates.update({k: v for k, v in action.items() if k != "parameters"})
        for key, value in candidates.items():
            if "digitalsourcetype" in key.lower() and isinstance(value, str):
                return value.rstrip("/").rsplit("/", 1)[-1]
    return None


def _generator(manifest: dict, actions: list[dict]) -> str | None:
    """Prefer the specific model named in an action over the host application.

    Firefly signs images generated elsewhere, so claim_generator says Adobe
    while the assertion says GPT Image 2. Reporting the first and not the
    second would name the wrong party as the maker.
    """
    # The model named on the creating action, which is the v2 way to say it.
    for action in actions:
        if action.get("action") != "c2pa.created":
            continue
        agent = action.get("softwareAgent")
        name = agent.get("name") if isinstance(agent, dict) else agent
        if isinstance(name, str) and name:
            return name

    # Adobe's vendor parameter, which is how credentials in the wild say it.
    for action in actions:
        params = action.get("parameters") or {}
        for key, value in params.items():
            if key.lower().endswith("details") and isinstance(value, str) and value:
                return value

    info = manifest.get("claim_generator_info") or []
    if info and isinstance(info, list) and info[0].get("name"):
        return str(info[0]["name"])
    return manifest.get("claim_generator")


def read_c2pa(path: Path) -> Finding:
    """Read a Content Credential, if the file carries one.

    Deliberately reports what the library reports rather than reducing it to a
    pass or fail. A credential can be cryptographically sound and still signed
    by a certificate nothing vouches for, which is a different thing from a
    broken signature and has to read differently.
    """
    try:
        from c2pa import Reader
    except Exception:  # noqa: BLE001 - the checker still works without it
        return Finding("c2pa", present=False, says="Content Credentials could not be read here.")

    _load_trust()

    try:
        with Reader(str(path)) as reader:
            report = json.loads(reader.json())
    except Exception:  # noqa: BLE001 - no credential is the common case
        return Finding(
            "c2pa",
            present=False,
            says="No Content Credential is attached to this file.",
        )

    active = report.get("active_manifest")
    manifest = (report.get("manifests") or {}).get(active) or {}
    codes = [s.get("code") for s in (report.get("validation_status") or [])]
    signature = manifest.get("signature_info") or {}
    actions = _actions(manifest)
    source_type = _source_type(actions)
    generator = _generator(manifest, actions)
    issuer = signature.get("issuer")

    # An untrusted credential is not a failed one. It means the signature holds
    # while nothing on the list vouches for the certificate that made it, which
    # has to read as unknown rather than as bad.
    other_failures = [c for c in codes if c != _UNTRUSTED]
    results = (report.get("validation_results") or {}).get("activeManifest") or {}
    successes = {s.get("code") for s in results.get("success", [])}
    if other_failures:
        trusted = False
    elif _TRUSTED in successes:
        trusted = True
    elif _UNTRUSTED in codes:
        trusted = None
    else:
        trusted = None

    who = issuer or "Someone"
    if other_failures:
        says = "This file carries a Content Credential that does not pass its own checks."
    elif source_type in SYNTHETIC_SOURCE_TYPES:
        made_by = generator or "a generative model"
        says = f"{who} signed a credential saying this was made by {made_by}."
        if trusted is None:
            says += " Nothing on our list vouches for that signer."
    else:
        says = f"{who} signed a credential describing how this file was made."

    return Finding(
        "c2pa",
        present=True,
        signed=True,
        trusted=trusted,
        says=says,
        detail={
            "issuer": issuer,
            "signed_at": signature.get("time"),
            "generator": generator,
            "claim_generator": manifest.get("claim_generator"),
            "digital_source_type": source_type,
            "actions": [a.get("action") for a in actions if a.get("action")],
            "ingredients": len(manifest.get("ingredients") or []),
            "failures": other_failures,
            "untrusted": _UNTRUSTED in codes,
        },
    )


def read_claims(path: Path, media_type: str) -> Finding:
    """Unsigned assertions written into the file's own metadata.

    Anyone can write these with a text editor, so they are evidence of intent
    and nothing more. Worth surfacing, never worth trusting, and the wording
    has to keep those apart.
    """
    if not media_type.startswith("image/"):
        return Finding("claims", present=False, says="No readable metadata claims.")

    try:
        from hallmark import metadata

        visible = metadata.read_visible(path)
    except Exception:  # noqa: BLE001 - a verdict must not hinge on metadata
        return Finding("claims", present=False, says="No readable metadata claims.")

    blob = json.dumps(visible).lower()
    declared = next((s for s in SYNTHETIC_SOURCE_TYPES if s.lower() in blob), None)

    if not visible:
        return Finding(
            "claims",
            present=False,
            says="Nothing is written into this file's own metadata.",
        )

    if declared:
        says = "The file's metadata says it was machine generated. Nothing signs that claim."
    else:
        says = "The file carries metadata, none of it declaring how the content was made."

    return Finding(
        "claims",
        present=True,
        signed=False,
        trusted=False,
        says=says,
        detail={"declared_source_type": declared, "fields": visible},
    )


def summarise(findings: list[Finding]) -> dict[str, Any]:
    """State only what the findings support, and name what is still unknown."""
    by = {f.source: f for f in findings}
    ours = by.get("hallmark")
    cred = by.get("c2pa")
    claims = by.get("claims")

    signed_synthetic = bool(
        cred
        and cred.present
        and not cred.detail.get("failures")
        and cred.detail.get("digital_source_type") in SYNTHETIC_SOURCE_TYPES
    )
    claimed_synthetic = bool(claims and claims.detail.get("declared_source_type"))

    if ours and ours.present:
        headline = "This file came from this pipeline and carries our record."
    elif signed_synthetic:
        who = (cred.detail.get("generator") or "a generative model") if cred else ""
        headline = f"Signed by {cred.detail.get('issuer')} as generated by {who}."
    elif cred and cred.present and cred.detail.get("failures"):
        headline = "This file carries a credential that fails its own checks."
    elif cred and cred.present:
        headline = "This file carries a signed credential that does not say it was generated."
    elif claimed_synthetic:
        headline = "This file claims to be machine generated, but nothing signs the claim."
    else:
        headline = "Nothing in this file says how it was made."

    return {
        "headline": headline,
        "generated_by_signed_credential": signed_synthetic,
        "generated_by_unsigned_claim": claimed_synthetic and not signed_synthetic,
        "sources_found": [f.source for f in findings if f.present],
        # Said on every answer, because it is true on every answer and it is the
        # thing a reader is most likely to get wrong.
        "caveat": (
            "Absence of a credential is not evidence a file is genuine. Most platforms "
            "strip this metadata when an image is uploaded."
        ),
    }


def inspect(path: Path, media_type: str, ours: Finding | None = None) -> dict[str, Any]:
    """Read a file every way we can and report each source separately."""
    findings = [f for f in (ours, read_c2pa(path), read_claims(path, media_type)) if f]
    return {
        "findings": [f.to_dict() for f in findings],
        "summary": summarise(findings),
    }
