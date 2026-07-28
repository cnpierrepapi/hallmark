"""The interactive demo: generate, pick, store, prove.

Serverless functions here cap out at 60 seconds and image generation runs
closer to 90, so nothing can block on a finished render. Genblaze's providers
expose submit, poll and fetch_output separately, so the work is split across
requests: submit returns job ids immediately, the browser polls, and selection
happens once the renders exist.

Candidates are submitted in parallel. Run seven one after another and a visitor
waits ten minutes; submitted together they wait once.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from genblaze_core.models.enums import Modality, PromptVisibility, StepType
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.run import Run
from genblaze_core.models.step import Step

from hallmark import storage

CANDIDATES = 7
SESSION_PREFIX = "sessions"
QUOTA_PREFIX = "quota"

# Public generation is paid for out of a small balance, so it is capped. Past
# the cap the page replays a recorded run rather than showing an error.
DAILY_GENERATION_CAP = 12
PER_SESSION_CAP = 3

# Short, because the function itself dies at 60 seconds. A submit that has not
# answered by now is one GMI is holding open, and those are recovered from the
# queue listing rather than treated as failures.
SUBMIT_TIMEOUT = 18.0

STYLE_HINT = (
    "3D character render, glossy and colourful, dramatic studio lighting, "
    "dark seamless background, high detail"
)


class QuotaExceeded(Exception):
    """Raised when the public generation budget for today is spent."""


@dataclass
class Candidate:
    index: int
    job_id: str
    seed: int = 0
    status: str = "queued"
    url: str | None = None
    sha256: str | None = None
    media_type: str = "image/png"
    latency_seconds: float = 0.0
    checks: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    passed: bool = False
    accepted: bool = False
    reason: str | None = None
    stored_key: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "url": self.url,
            "sha256": self.sha256,
            "latency_seconds": round(self.latency_seconds, 1),
            "checks": self.checks,
            "score": self.score,
            "passed": self.passed,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _session_key(session_id: str) -> str:
    return f"{SESSION_PREFIX}/{session_id}.json"


def load_session(session_id: str) -> dict | None:
    try:
        body = storage.client().get_object(
            Bucket=storage.bucket(), Key=_session_key(session_id)
        )["Body"].read()
    except Exception:  # noqa: BLE001 - a missing session is a normal answer
        return None
    return json.loads(body)


def save_session(session: dict) -> None:
    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=_session_key(session["session_id"]),
        Body=json.dumps(session, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _quota_key() -> str:
    return f"{QUOTA_PREFIX}/{date.today().isoformat()}.json"


def _read_quota() -> dict:
    try:
        body = storage.client().get_object(Bucket=storage.bucket(), Key=_quota_key())[
            "Body"
        ].read()
        return json.loads(body)
    except Exception:  # noqa: BLE001 - no file yet means nothing spent today
        return {"generations": 0, "sessions": {}}


def _bump_quota(session_id: str) -> None:
    """Count a generation against today's budget.

    Not transactional. Object storage has no compare-and-set, so two requests
    landing in the same instant can both read the same count. The cap is a
    spend guard rather than a security boundary, and being off by one costs
    pennies, so a simple read and write is the right trade here.
    """
    quota = _read_quota()
    quota["generations"] = quota.get("generations", 0) + 1
    quota["sessions"][session_id] = quota["sessions"].get(session_id, 0) + 1
    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=_quota_key(),
        Body=json.dumps(quota).encode("utf-8"),
        ContentType="application/json",
    )


def quota_status(session_id: str | None = None) -> dict:
    quota = _read_quota()
    used = quota.get("generations", 0)
    mine = quota["sessions"].get(session_id, 0) if session_id else 0
    return {
        "used_today": used,
        "daily_cap": DAILY_GENERATION_CAP,
        "remaining_today": max(0, DAILY_GENERATION_CAP - used),
        "used_this_session": mine,
        "session_cap": PER_SESSION_CAP,
        "can_generate": used < DAILY_GENERATION_CAP and mine < PER_SESSION_CAP,
    }


def _recover_unresolved(candidates: list[Candidate]) -> None:
    """Find job ids for submits whose response never came back.

    GMI sometimes holds the connection on POST /requests until the render
    finishes rather than returning a job id. The work still runs and still
    bills, so treating a client timeout as a failure would throw away
    something already paid for.

    Every candidate carries a unique seed and GMI echoes the payload on the
    queue listing, so an unresolved submit can be matched back to its job.
    """
    import os

    import httpx

    missing = [c for c in candidates if c.status == "unresolved"]
    if not missing:
        return

    try:
        with httpx.Client(
            base_url="https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey",
            headers={"Authorization": f"Bearer {os.environ['GMI_API_KEY']}"},
            timeout=20.0,
        ) as client:
            body = client.get("/requests").json()
    except Exception:  # noqa: BLE001 - recovery is best effort
        return

    rows = body if isinstance(body, list) else body.get("requests") or body.get("data") or []
    by_seed = {}
    for row in rows:
        seed = (row.get("payload") or {}).get("seed")
        job_id = row.get("request_id") or row.get("id")
        if seed is not None and job_id:
            by_seed[int(seed)] = str(job_id)

    for candidate in missing:
        job_id = by_seed.get(candidate.seed)
        if job_id:
            candidate.job_id = job_id
            candidate.status = "queued"
            candidate.reason = None
        else:
            candidate.status = "failed"


def start_generation(brief: str, session_id: str | None = None) -> dict:
    """Submit the candidates and return immediately with their job ids."""
    from hallmark.providers import IMAGE_MODEL, image_provider

    session_id = session_id or str(uuid.uuid4())
    status = quota_status(session_id)
    if not status["can_generate"]:
        raise QuotaExceeded(
            f"Daily generation cap reached ({status['used_today']}/{status['daily_cap']})"
            if status["used_today"] >= status["daily_cap"]
            else f"This browser has used its {status['session_cap']} generations"
        )

    prompt = f"{brief.strip()}. {STYLE_HINT}"
    provider = image_provider(http_timeout=SUBMIT_TIMEOUT)
    batch = int(time.time())

    def submit(index: int) -> Candidate:
        # Same prompt, different seed. The seed doubles as a marker: GMI
        # echoes the payload back on the queue listing, so a submit whose
        # response never arrives can still be matched to its job.
        seed = batch * 100 + index
        step = Step(
            step_type=StepType.GENERATE,
            modality=Modality.IMAGE,
            provider="gmicloud-image",
            model=IMAGE_MODEL,
            prompt=prompt,
            prompt_visibility=PromptVisibility.PRIVATE,
            seed=seed,
        )
        try:
            job_id = provider.submit(step)
            return Candidate(index=index, job_id=str(job_id), seed=seed)
        except Exception as exc:  # noqa: BLE001 - recovered below, not lost
            return Candidate(
                index=index, job_id="", seed=seed, status="unresolved",
                reason=str(exc)[:160],
            )

    with ThreadPoolExecutor(max_workers=CANDIDATES) as pool:
        candidates = list(pool.map(submit, range(CANDIDATES)))

    _recover_unresolved(candidates)
    _bump_quota(session_id)

    session = {
        "session_id": session_id,
        "brief": brief.strip(),
        "prompt": prompt,
        "model": IMAGE_MODEL,
        "created_at": _now(),
        "status": "generating",
        "candidates": [c.__dict__ for c in candidates],
        "selection": None,
        "run_id": None,
        "manifest_uri": None,
    }
    save_session(session)
    return session


def poll_generation(session_id: str) -> dict:
    """Check each outstanding job and record any that finished."""
    from hallmark.providers import image_provider

    session = load_session(session_id)
    if session is None:
        raise KeyError(session_id)
    if session["status"] not in ("generating",):
        return session

    provider = image_provider()

    def check(raw: dict) -> dict:
        if raw.get("status") in ("ready", "failed"):
            return raw
        try:
            done = provider.poll(raw["job_id"])
        except Exception as exc:  # noqa: BLE001 - a poll blip is not a failure
            raw["status"] = "queued"
            raw["reason"] = str(exc)[:200]
            return raw
        if not done:
            raw["status"] = "running"
            return raw

        step = Step(
            step_type=StepType.GENERATE,
            modality=Modality.IMAGE,
            provider="gmicloud-image",
            model=session["model"],
            prompt=session["prompt"],
            prompt_visibility=PromptVisibility.PRIVATE,
        )
        try:
            filled = provider.fetch_output(raw["job_id"], step)
        except Exception as exc:  # noqa: BLE001 - surfaced per candidate
            raw["status"] = "failed"
            raw["reason"] = str(exc)[:200]
            return raw

        if not filled.assets:
            raw["status"] = "failed"
            raw["reason"] = filled.error or "no image returned"
            return raw

        asset = filled.assets[0]
        raw["status"] = "ready"
        raw["url"] = str(asset.url)
        raw["media_type"] = asset.media_type or "image/png"
        return raw

    with ThreadPoolExecutor(max_workers=CANDIDATES) as pool:
        session["candidates"] = list(pool.map(check, session["candidates"]))

    states = {c["status"] for c in session["candidates"]}
    if states <= {"ready", "failed"}:
        session["status"] = "ready" if "ready" in states else "failed"

    save_session(session)
    return session


RATIONALE_SYSTEM = """You write short QA notes for a creative team.

You have NOT seen the images. You are given the reviewer's own stated reason for
picking one, plus measured technical facts about each candidate. Write one short
line per rejected candidate saying why it was not chosen.

Rules:
- Ground every line in the reviewer's stated reason or in the measured facts.
- Never describe what an image looks like. Never invent visual detail.
- Each line must be DIFFERENT from the others. Do not repeat one sentence
  across candidates.
- Where a candidate differs on a measured fact (score, a failed check, render
  time, file size), lead with that fact and quote the number.
- Where nothing separates it, say so plainly: it was close, and the reviewer
  went the other way on their own grounds. Vary how you say it.
- Keep each line under 18 words.

Reply with JSON only: {"1": "line", "2": "line"} keyed by candidate index."""


def _rationales(session: dict, picked: int, human_reason: str) -> dict[str, str]:
    """Ask a text model to explain the rejections from the human's reason.

    The model is told plainly that it has not seen the images, and it is given
    only the reviewer's words and the measured checks. That keeps the output
    honest: no model on this account can see an image, so anything describing
    what a picture looks like would be invented.
    """
    from genblaze_gmicloud import chat

    from hallmark.providers import CHAT_MODEL

    facts = []
    for c in session["candidates"]:
        if c["status"] != "ready":
            facts.append(f'candidate {c["index"]}: failed to generate')
            continue
        checks = ", ".join(
            f'{k["name"]}={k["detail"]}' for k in (c.get("checks") or [])
        )
        facts.append(
            f'candidate {c["index"]}: '
            f'{"CHOSEN by reviewer" if c["index"] == picked else "not chosen"}, '
            f'score {c.get("score")}, {c.get("latency_seconds")}s, {checks}'
        )

    prompt = (
        f'Brief: {session["brief"]}\n'
        f'Reviewer picked candidate {picked}.\n'
        f'Reviewer said: "{human_reason.strip() or "no reason given"}"\n\n'
        + "\n".join(facts)
    )

    try:
        response = chat(model=CHAT_MODEL, prompt=prompt, system=RATIONALE_SYSTEM,
                        temperature=0.4, max_tokens=700)
        text = response.text or ""
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return {str(k): str(v) for k, v in json.loads(text[start : end + 1]).items()}
    except Exception:  # noqa: BLE001 - the demo must not hinge on the planner
        pass

    reason = human_reason.strip() or "the reviewer preferred another candidate"
    return {
        str(c["index"]): f"Not chosen: {reason}"
        for c in session["candidates"]
        if c["index"] != picked
    }


def select_candidate(session_id: str, picked: int, human_reason: str, workdir: Path) -> dict:
    """Store every candidate, stamp the chosen one, and record the decision."""
    from genblaze_core.models.asset import Asset

    from hallmark import approval
    from hallmark.approval import Approval
    from hallmark.evaluate import evaluate
    from hallmark.stamp import stamp
    from genblaze_core.models.policy import EmbedPolicy

    session = load_session(session_id)
    if session is None:
        raise KeyError(session_id)

    ready = [c for c in session["candidates"] if c["status"] == "ready"]
    if not any(c["index"] == picked for c in ready):
        raise ValueError(f"Candidate {picked} is not available to pick")

    workdir.mkdir(parents=True, exist_ok=True)

    def fetch(raw: dict) -> dict:
        """Pull the bytes, hash them, and run the technical checks."""
        import hashlib

        local = workdir / f"cand_{raw['index']}.png"
        storage.download(raw["url"], local)
        data = local.read_bytes()

        raw["sha256"] = hashlib.sha256(data).hexdigest()
        raw["size_bytes"] = len(data)
        check = evaluate(local, "image")
        raw["checks"] = check.to_dict()["checks"]
        raw["score"] = check.score
        raw["passed"] = check.passed
        raw["local_path"] = str(local)
        return raw

    with ThreadPoolExecutor(max_workers=CANDIDATES) as pool:
        fetched = list(pool.map(fetch, ready))

    by_index = {c["index"]: c for c in fetched}

    # One record covering the whole run, so the rejected attempts are inside
    # the provenance rather than thrown away.
    steps = []
    for c in fetched:
        steps.append(
            Step(
                step_type=StepType.GENERATE,
                modality=Modality.IMAGE,
                provider="gmicloud-image",
                model=session["model"],
                prompt=session["prompt"],
                prompt_visibility=PromptVisibility.PRIVATE,
                assets=[
                    Asset(
                        url=c["url"],
                        media_type="image/png",
                        sha256=c["sha256"],
                        size_bytes=c["size_bytes"],
                    )
                ],
                metadata={"candidate": c["index"], "chosen": c["index"] == picked},
            )
        )

    manifest = Manifest.from_run(Run(name=f"demo/{session['brief'][:48]}", steps=steps))
    approval.apply(
        manifest,
        Approval(approver="Visitor", decision="approved",
                 note=(human_reason.strip() or "picked on the page")[:400]),
    )
    manifest_uri = approval.publish_manifest(manifest, session_id)

    notes = _rationales(session, picked, human_reason)

    # Store every candidate, chosen or not. The rejects are the inventory.
    def store(c: dict) -> dict:
        local = Path(c["local_path"])
        chosen = c["index"] == picked
        if chosen:
            stamped = workdir / f"chosen_{c['index']}.png"
            stamp(
                local,
                manifest,
                stamped,
                policy=EmbedPolicy(embed_mode="pointer",
                                   prompt_visibility=PromptVisibility.PRIVATE),
                mime_type="image/png",
            )
            local = stamped
        key = f"{SESSION_PREFIX}/{session_id}/{'chosen' if chosen else 'reject'}_{c['index']}.png"
        storage.upload(local, key, "image/png")
        c["stored_key"] = key
        c["accepted"] = chosen
        c["reason"] = None if chosen else notes.get(str(c["index"]))
        c.pop("local_path", None)
        return c

    with ThreadPoolExecutor(max_workers=CANDIDATES) as pool:
        stored = list(pool.map(store, fetched))

    for c in session["candidates"]:
        if c["index"] in by_index:
            c.update(next(s for s in stored if s["index"] == c["index"]))

    session["status"] = "selected"
    session["selection"] = {
        "picked": picked,
        "human_reason": human_reason.strip(),
        "decided_at": _now(),
    }
    session["run_id"] = manifest.run.run_id
    session["manifest_uri"] = manifest_uri
    session["canonical_hash"] = manifest.canonical_hash
    save_session(session)
    return session
