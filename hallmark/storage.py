"""Backblaze B2 access.

Assets live in a private bucket, so nothing here works anonymously. That is
deliberate: campaign material is usually unreleased, and the provider's own
URLs are public to anyone holding the link.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from genblaze_core import KeyStrategy, ObjectStorageSink
from genblaze_s3 import S3StorageBackend

_CLIENT = None

# Every boto3 client opens its own connection pool and re-resolves
# credentials. Building one per call exhausts sockets partway through a
# campaign, which shows up as "Could not connect to the endpoint URL" on the
# later assets rather than the first. One cached client, with retries so a
# blip does not lose media we have already paid to generate.
_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    max_pool_connections=25,
    connect_timeout=15,
    read_timeout=120,
)


def bucket() -> str:
    return os.environ["B2_BUCKET"]


def endpoint() -> str:
    return os.environ["B2_ENDPOINT"]


def client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint(),
            region_name=os.environ.get("B2_REGION"),
            aws_access_key_id=os.environ["B2_KEY_ID"],
            aws_secret_access_key=os.environ["B2_APP_KEY"],
            config=_CONFIG,
        )
    return _CLIENT


def sink() -> ObjectStorageSink:
    """A genblaze sink that stores assets and manifests in B2.

    The sink is what makes provenance possible at all: GMI returns a URL but
    no hash, and this is the step that hashes each asset as it streams in.
    """
    return ObjectStorageSink(
        S3StorageBackend.for_backblaze(
            bucket(),
            region=os.environ.get("B2_REGION"),
            key_id=os.environ["B2_KEY_ID"],
            app_key=os.environ["B2_APP_KEY"],
        ),
        key_strategy=KeyStrategy.HIERARCHICAL,
    )


def key_for(url: str) -> str | None:
    """Return the bucket key for a URL that points at our own bucket."""
    base = endpoint()
    if url.startswith(base):
        remainder = url[len(base) :].lstrip("/")
        prefix = f"{bucket()}/"
        return remainder[len(prefix) :] if remainder.startswith(prefix) else remainder
    if url.startswith("s3://"):
        _, _, rest = url.partition("s3://")
        _, _, key = rest.partition("/")
        return key or None
    return None


def download(url: str, dest: Path) -> None:
    """Fetch an asset to disk, using credentials when it is ours."""
    key = key_for(url)
    if key:
        client().download_file(bucket(), key, str(dest))
        return

    import httpx

    with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def upload(path: Path, key: str, content_type: str) -> str:
    """Store a file and return the URL recorded for it."""
    client().put_object(
        Bucket=bucket(),
        Key=key,
        Body=path.read_bytes(),
        ContentType=content_type,
    )
    return f"{endpoint()}/{bucket()}/{key}"


def presigned(url_or_key: str, expires_seconds: int = 3600) -> str:
    """A time-limited link so a private asset can be shown without opening it up."""
    key = key_for(url_or_key) or url_or_key
    return client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket(), "Key": key},
        ExpiresIn=expires_seconds,
    )


def is_ours(url: str) -> bool:
    return key_for(url) is not None


def host_of(url: str) -> str:
    return urlparse(url).netloc
