"""Attempt records that can be written from inside the deployed function.

The ledger itself is Parquet, and pyarrow is a columnar engine far too heavy to
ship inside a serverless function that never runs a query. Importing it there
would cost every cold start for a capability the request path does not use.

So attempts made in public get dropped here as small JSON objects, and the
local publish step folds them into the Parquet ledger before it reports. The
figures on the page end up counting the traffic the page itself generated,
which is the whole claim section 07 makes, while the function stays light.

One object per batch, never a rewrite of a shared file. Object storage has no
append, so a read-modify-write cycle would silently drop whichever visitor
happened to be selecting at the same moment.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from hallmark import storage

PENDING_PREFIX = "ledger/pending"


def record(rows: list[dict[str, Any]]) -> str | None:
    """Store a batch of attempts for the ledger to pick up. Returns the key.

    Best effort by design: a visitor who has just approved an asset should not
    see an error because bookkeeping failed behind them. A dropped batch costs
    a row in a report, which is worth less than the selection it would break.
    """
    if not rows:
        return None

    now = datetime.now(timezone.utc)
    for row in rows:
        row.setdefault("attempt_id", str(uuid.uuid4()))
        row.setdefault("created_at", now.isoformat())

    key = f"{PENDING_PREFIX}/date={now.strftime('%Y-%m-%d')}/{uuid.uuid4()}.json"
    try:
        storage.client().put_object(
            Bucket=storage.bucket(),
            Key=key,
            Body=json.dumps(rows).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:  # noqa: BLE001 - never break a selection over a record
        return None
    return key
