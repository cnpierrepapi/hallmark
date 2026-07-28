"""Confirm the B2 bucket is reachable and writable before building on it.

Does a real round trip: put an object, read it back, compare bytes, then
delete it. A green result here means storage is genuinely wired, not just
that credentials parsed.

    python scripts/check_b2.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_REGIONS = ("us-west-005", "us-west-004", "us-west-002", "us-east-005", "eu-central-003")
PROBE_KEY = "_hallmark/connectivity-probe.txt"
PROBE_BODY = b"hallmark storage probe"


def _client(endpoint: str, region: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APP_KEY"],
    )


def main() -> int:
    load_dotenv(ROOT / ".env")

    missing = [k for k in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET") if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return 1

    bucket = os.environ["B2_BUCKET"]
    configured = os.environ.get("B2_ENDPOINT", "").strip()

    endpoints = []
    if configured:
        region = configured.split("s3.")[-1].split(".")[0]
        endpoints.append((configured, region))
    for region in CANDIDATE_REGIONS:
        candidate = (f"https://s3.{region}.backblazeb2.com", region)
        if candidate not in endpoints:
            endpoints.append(candidate)

    for endpoint, region in endpoints:
        print(f"[..] trying {endpoint}")
        try:
            client = _client(endpoint, region)
            client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "?")
            print(f"     no: {code}")
            continue
        except Exception as exc:  # noqa: BLE001 - probe should never hard fail
            print(f"     no: {type(exc).__name__}: {exc}")
            continue

        print(f"[ok] bucket '{bucket}' reachable at {endpoint}")

        client.put_object(Bucket=bucket, Key=PROBE_KEY, Body=PROBE_BODY)
        print("[ok] wrote probe object")

        body = client.get_object(Bucket=bucket, Key=PROBE_KEY)["Body"].read()
        if body != PROBE_BODY:
            print("[!!] read back did not match what was written")
            return 1
        print("[ok] read back matches")

        client.delete_object(Bucket=bucket, Key=PROBE_KEY)
        print("[ok] cleaned up probe object")

        print(f"\nPASS  set B2_ENDPOINT={endpoint}")
        return 0

    print("\nFAIL  no endpoint accepted these credentials for this bucket.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
