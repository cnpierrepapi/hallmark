"""Attach a Content Credential to a delivered asset, carrying its parent with it.

The images this pipeline generates arrive with a credential signed by Adobe
naming the model that made them. Converting a PNG to the JPEG we deliver threw
that away: the encoder rewrites the file and cannot carry a JUMBF box, so a
provenance product was quietly destroying the strongest piece of provenance in
the file it was handed.

Copying the box across verbatim would be worse than losing it. A C2PA manifest
is hard bound to the exact bytes it was signed over, so the same box on a
re-encoded file describes a file that no longer exists and fails its own check.
A credential that fails is a louder wrong answer than no credential at all.

The format's own answer to this is an ingredient. The delivered JPEG is a new
asset that declares the original as its parent, and the parent keeps its own
signature inside the new manifest. So a reader sees two things at once: that
HALLMARK signed what was delivered, and that Adobe signed what it came from.
The trace survives the conversion instead of being asserted by us second hand.

Ordering matters and is not negotiable. The credential is written last, after
the visible signature and after our own pointer, because its signature has to
cover everything else in the file. Our hash therefore ignores credential boxes
entirely, which is why integrity.strip removes them.

The signing key is read from the environment and never from the repository. A
self issued certificate is not on any public trust list, so our own credential
reads as untrusted while the parent's still reads as trusted. That is honest:
nobody has vouched for us, and the page should not pretend otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CERT_ENV = "HALLMARK_C2PA_CERT"
KEY_ENV = "HALLMARK_C2PA_KEY"

# A credential with no trusted timestamp stops verifying the day the signing
# certificate expires. The signing call requires one.
TIMESTAMP_AUTHORITY = os.environ.get(
    "HALLMARK_C2PA_TSA", "http://timestamp.digicert.com"
).encode()

SYNTHETIC = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"

# What the format calls the containers we deliver.
MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _pem(name: str) -> bytes | None:
    """Read a PEM out of the environment, however the host mangled the newlines.

    An env file and a hosting dashboard disagree about what a multi line value
    is, so the same key arrives either with real newlines or with the two
    characters a backslash and an n. PEM parsing fails on the second, silently
    turning into "no credential attached" rather than an error anyone would see.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    return raw.replace("\\n", "\n").strip().encode() + b"\n"


def available() -> bool:
    """Whether a signing identity is configured at all."""
    return bool(_pem(CERT_ENV) and _pem(KEY_ENV))


def _manifest(model: str | None, approver: str | None, note: str | None) -> dict:
    """The claim we are willing to sign our name to.

    Only what we actually know: which model produced it, that it is machine
    generated, and who signed it off. No claim about the picture itself, which
    nothing here has ever looked at.
    """
    actions: list[dict] = [
        {
            "action": "c2pa.created",
            "digitalSourceType": SYNTHETIC,
            "softwareAgent": {"name": model or "a generative model"},
        },
        {"action": "c2pa.converted", "softwareAgent": {"name": "HALLMARK"}},
    ]

    manifest: dict = {
        "claim_generator_info": [{"name": "HALLMARK", "version": "1.0"}],
        "title": "Approved campaign asset",
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
    }

    if approver:
        manifest["assertions"].append(
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "creator": [{"@type": "Person", "name": approver}],
                    **({"reviewBody": note} if note else {}),
                },
            }
        )

    return manifest


def sign(
    source: Path,
    dest: Path,
    *,
    parent: Path | None = None,
    model: str | None = None,
    approver: str | None = None,
    note: str | None = None,
) -> bool:
    """Write source to dest with a Content Credential attached.

    Returns False and leaves dest untouched when no identity is configured or
    the library is unavailable, so a missing key degrades delivery to what it
    was rather than failing the campaign. Never raises for that reason: an
    asset without a credential is still a signed, verifiable asset here.
    """
    if not available():
        return False

    try:
        from c2pa import Builder, C2paSignerInfo, Signer
    except Exception:  # noqa: BLE001 - delivery must not hinge on the extra
        return False

    try:
        signer = Signer.from_info(
            C2paSignerInfo(
                alg=b"es256",
                sign_cert=_pem(CERT_ENV),
                private_key=_pem(KEY_ENV),
                ta_url=TIMESTAMP_AUTHORITY,
            )
        )

        builder = Builder(_manifest(model, approver, note))

        if parent and parent.exists():
            parent_mime = MIME_BY_SUFFIX.get(parent.suffix.lower(), "image/png")
            with open(parent, "rb") as handle:
                builder.add_ingredient(
                    json.dumps({"title": parent.name, "relationship": "parentOf"}),
                    parent_mime,
                    handle,
                )

        dest.unlink(missing_ok=True)
        builder.sign_file(source, dest, signer)
    except Exception:  # noqa: BLE001 - see the docstring
        dest.unlink(missing_ok=True)
        return False

    return dest.exists()
