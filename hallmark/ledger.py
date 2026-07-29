"""A durable record of every generation attempt, including the rejects.

Generation is non-deterministic and paid for by the attempt. Most pipelines
keep the winner and throw away everything else, which means nobody can answer
the questions that actually matter after a month of use: how often does this
model fail, what does an accepted asset really cost once failures are counted,
and did last week's prompt change make things better or worse.

So every attempt is appended here, accepted or not, with its score, cost,
latency and the reason it was rejected. Parquet on B2, partitioned by day, so
it can be queried directly without standing up a database.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from hallmark import attempts, storage

LEDGER_PREFIX = "ledger/attempts"

SCHEMA = pa.schema(
    [
        ("attempt_id", pa.string()),
        ("run_id", pa.string()),
        ("campaign", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("modality", pa.string()),
        ("model", pa.string()),
        ("provider", pa.string()),
        ("accepted", pa.bool_()),
        ("score", pa.float64()),
        ("reject_reason", pa.string()),
        ("latency_seconds", pa.float64()),
        ("cost_usd", pa.float64()),
        ("sha256", pa.string()),
        ("size_bytes", pa.int64()),
        ("media_type", pa.string()),
        ("checks", pa.string()),
    ]
)


@dataclass
class Attempt:
    run_id: str
    campaign: str
    modality: str
    model: str
    provider: str
    accepted: bool
    score: float
    latency_seconds: float
    reject_reason: str | None = None
    cost_usd: float | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    checks: str | None = None
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["cost_usd"] = float(row["cost_usd"]) if row["cost_usd"] is not None else None
        row["size_bytes"] = int(row["size_bytes"]) if row["size_bytes"] is not None else None
        return row


def _table(attempts: list[Attempt]) -> pa.Table:
    rows = [a.as_row() for a in attempts]
    columns = {name: [row.get(name) for row in rows] for name in SCHEMA.names}
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def write(attempts: list[Attempt]) -> str | None:
    """Append a batch of attempts to the ledger. Returns the object key.

    Written as one immutable object per batch rather than by rewriting a single
    file. Object storage has no append, and a read-modify-write cycle would
    lose records whenever two campaigns run at once.
    """
    if not attempts:
        return None

    day = attempts[0].created_at.strftime("%Y-%m-%d")
    key = f"{LEDGER_PREFIX}/date={day}/{uuid.uuid4()}.parquet"

    buffer = io.BytesIO()
    pq.write_table(_table(attempts), buffer, compression="snappy")
    buffer.seek(0)

    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    return key


def drain_pending() -> int:
    """Fold attempts recorded by the live demo into the Parquet ledger.

    The deployed function cannot write Parquet, so it leaves JSON behind (see
    hallmark/attempts.py). This converts whatever has accumulated, writes it as
    a normal batch, and removes the JSON only once the batch is safely stored.
    Crashing in between costs a duplicate import, never a lost attempt.
    """
    client = storage.client()
    paginator = client.get_paginator("list_objects_v2")

    keys: list[str] = []
    rows: list[Attempt] = []
    for page in paginator.paginate(Bucket=storage.bucket(), Prefix=attempts.PENDING_PREFIX):
        for item in page.get("Contents", []):
            if not item["Key"].endswith(".json"):
                continue
            body = client.get_object(Bucket=storage.bucket(), Key=item["Key"])["Body"].read()
            for row in json.loads(body):
                row["created_at"] = datetime.fromisoformat(row["created_at"])
                rows.append(Attempt(**row))
            keys.append(item["Key"])

    if not rows:
        return 0

    write(rows)
    for key in keys:
        client.delete_object(Bucket=storage.bucket(), Key=key)
    return len(rows)


def read_all() -> pa.Table | None:
    """Load the whole ledger. Fine at hackathon scale, and it proves the point."""
    client = storage.client()
    paginator = client.get_paginator("list_objects_v2")

    tables = []
    for page in paginator.paginate(Bucket=storage.bucket(), Prefix=LEDGER_PREFIX):
        for item in page.get("Contents", []):
            if not item["Key"].endswith(".parquet"):
                continue
            body = client.get_object(Bucket=storage.bucket(), Key=item["Key"])["Body"].read()
            tables.append(pq.read_table(io.BytesIO(body)))

    if not tables:
        return None
    return pa.concat_tables(tables)


def summary() -> list[dict[str, Any]]:
    """Per model: attempts, acceptance rate, and true cost per accepted asset."""
    table = read_all()
    if table is None:
        return []

    rows = table.to_pylist()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        key = (row["modality"], row["model"])
        bucket = grouped.setdefault(
            key,
            {
                "modality": row["modality"],
                "model": row["model"],
                "attempts": 0,
                "accepted": 0,
                "total_cost_usd": 0.0,
                "total_latency": 0.0,
            },
        )
        bucket["attempts"] += 1
        bucket["accepted"] += 1 if row["accepted"] else 0
        bucket["total_cost_usd"] += row["cost_usd"] or 0.0
        bucket["total_latency"] += row["latency_seconds"] or 0.0

    out = []
    for bucket in grouped.values():
        attempts = bucket["attempts"]
        accepted = bucket["accepted"]
        out.append(
            {
                "modality": bucket["modality"],
                "model": bucket["model"],
                "attempts": attempts,
                "accepted": accepted,
                "acceptance_rate": round(accepted / attempts, 3) if attempts else 0.0,
                "avg_latency_seconds": round(bucket["total_latency"] / attempts, 2),
                "total_cost_usd": round(bucket["total_cost_usd"], 4),
                # The number that matters: failures are paid for too, so the
                # real price of a usable asset includes everything discarded.
                "cost_per_accepted_usd": (
                    round(bucket["total_cost_usd"] / accepted, 4) if accepted else None
                ),
            }
        )
    return sorted(out, key=lambda r: (r["modality"], r["model"]))
