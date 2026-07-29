"""The interactive demo: generate, pick, store, prove.

Serverless functions here cap out at 60 seconds and image generation runs
closer to 90, so nothing can block on a finished render. Genblaze's providers
expose submit, poll and fetch_output separately, so the work is split across
requests: submit returns job ids immediately, the browser polls, and selection
happens once the renders exist.

Candidates are submitted in parallel. Run them one after another and a visitor
waits minutes; submitted together they wait once.

A browser session can hold more than one run. Each run keeps its own numbered
prefix in storage, because the first version of this wrote every run to the
same keys and a second run destroyed the first one's files. Worse, those keys
are served with a long cache lifetime, so the page went on showing an asset
that had already been overwritten. Runs accumulate in ``runs`` and the marking
record covers all of them; the page shows the current one.
"""

from __future__ import annotations

import hashlib
import json
import os
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

from hallmark import attempts, storage

SESSION_PREFIX = "sessions"
QUOTA_PREFIX = "quota"

# Public generation is paid for out of a small balance, so it is capped.
#
# Video has its own counters and they are deliberately much tighter. A clip off
# wan2.7 is 1080p30 and takes over two minutes, so one visitor picking video
# costs more than several picking stills. Sharing one counter across both would
# let a handful of people spend the whole day's budget on the expensive half.
DAILY_GENERATION_CAP = 12
PER_SESSION_CAP = 3
DAILY_VIDEO_CAP = 3
PER_SESSION_VIDEO_CAP = 1

# Short, because the function itself dies at 60 seconds. A submit that has not
# answered by now is one GMI is holding open, and those are recovered from the
# queue listing rather than treated as failures.
SUBMIT_TIMEOUT = 18.0

# MEASURED: wan2.7 refuses a seed outside a signed 32 bit range with
#
#     seed must be in [0,2147483647] (code: InvalidParameter)
#
# and it refuses it ASYNCHRONOUSLY. The submit is accepted, a job id comes back,
# and the job only fails a second later in the queue listing, so nothing at
# submit time can catch it. gpt-image-2 takes the same seed happily, which is
# why a unix timestamp times a hundred worked for months and then failed the
# moment video was offered.
#
# The seed still has to be unique per candidate, because it is what a submit
# whose response never arrived is matched back to its job by. Wrapping the
# clock keeps it inside the range while staying unique for months at a time:
# 20,000,000 seconds is about 231 days, and the widest seed is 2,000,000,099.
SEED_CYCLE_SECONDS = 20_000_000
MAX_SEED = 2_147_483_647

# A style is a preset, not free text. The brief is the visitor's; the styling
# is ours, so the look stays consistent across a campaign and the expanded
# prompt stays something we can withhold without withholding their own words.
#
# Each preset carries a second hint for video. A still's hint talks about
# lenses and lighting and says nothing about what moves, so reusing it for a
# clip wastes the only instruction that matters to a video model.
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "product": {
        "label": "Product still",
        "note": "clean commercial shot, seamless sweep, controlled light",
        "hint": (
            "studio product photography, seamless sweep background, controlled "
            "softbox lighting, crisp reflections, sharp focus, high detail"
        ),
        "video_hint": (
            "studio product shot, slow controlled turntable move, seamless "
            "sweep background, softbox lighting, crisp reflections, locked "
            "focus, no camera shake"
        ),
    },
    "character": {
        "label": "3D character",
        "note": "glossy render, dramatic key light, dark ground",
        "hint": (
            "3D character render, glossy and colourful, dramatic studio lighting, "
            "dark seamless background, high detail"
        ),
        "video_hint": (
            "3D character animation, glossy and colourful, slow orbit around the "
            "subject, dramatic studio key light, dark seamless background"
        ),
    },
    "editorial": {
        "label": "Editorial photo",
        "note": "natural light, shallow depth of field, filmic colour",
        "hint": (
            "editorial photograph, 50mm lens, natural window light, shallow depth "
            "of field, filmic colour grade, candid framing"
        ),
        "video_hint": (
            "editorial footage, 50mm lens, gentle handheld drift, natural window "
            "light, shallow depth of field, filmic colour grade"
        ),
    },
    "poster": {
        "label": "Graphic poster",
        "note": "flat vector shapes, bold palette, print ready",
        "hint": (
            "flat vector poster illustration, bold geometric shapes, limited high "
            "contrast palette, generous negative space, print ready"
        ),
        "video_hint": (
            "flat graphic motion design, bold geometric shapes sliding into place, "
            "limited high contrast palette, generous negative space"
        ),
    },
}

DEFAULT_STYLE = "product"

# What a visitor can ask for, and what it costs to give them.
#
# The delivered still is a JPEG because Windows shows no metadata whatsoever
# for a PNG, so a PNG download could never display its own signature in a file
# browser. Measured, not assumed: see hallmark/metadata.py. A clip stays an
# MP4 and gets its property table rewritten in place.
MEDIA_KINDS: dict[str, dict[str, Any]] = {
    "image": {
        "label": "Still image",
        "note": "three candidates, about a minute",
        "candidates": 3,
        "raw_suffix": ".png",
        "raw_mime": "image/png",
        "delivery_suffix": ".jpg",
        "delivery_mime": "image/jpeg",
        "modality": "image",
    },
    "video": {
        "label": "Video clip",
        "note": "two clips, about three minutes",
        # Two, not one. A single candidate would leave nothing to choose
        # between and no reject to keep, which is most of what the demo is
        # for. Two is the smallest number that still makes it a decision.
        "candidates": 2,
        "raw_suffix": ".mp4",
        "raw_mime": "video/mp4",
        "delivery_suffix": ".mp4",
        "delivery_mime": "video/mp4",
        "modality": "video",
    },
}

DEFAULT_KIND = "image"

# A clip is 7 to 18MB and the platform refuses any request body over 4.5MB, so
# a visitor can never upload one back for checking. Anything at or above this
# is checked where it lies instead, with the browser hashing its own copy to
# confirm the checker read the same bytes.
UPLOADABLE_MAX_BYTES = 4 * 1024 * 1024

VERIFY_BASE = os.environ.get("HALLMARK_VERIFY_BASE", "https://hallmark-rust.vercel.app")

# Kept for callers that predate the media kinds.
CANDIDATES = MEDIA_KINDS["image"]["candidates"]
DELIVERY_MIME = MEDIA_KINDS["image"]["delivery_mime"]
DELIVERY_SUFFIX = MEDIA_KINDS["image"]["delivery_suffix"]


def style_choices() -> list[dict[str, str]]:
    """The presets, for the page to render as options."""
    return [
        {"slug": slug, "label": preset["label"], "note": preset["note"]}
        for slug, preset in STYLE_PRESETS.items()
    ]


def kind_choices() -> list[dict[str, Any]]:
    """The media kinds, for the page to render as options."""
    return [
        {
            "slug": slug,
            "label": spec["label"],
            "note": spec["note"],
            "candidates": spec["candidates"],
            "uploadable": slug != "video",
        }
        for slug, spec in MEDIA_KINDS.items()
    ]


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


def _kind_of(session: dict) -> dict[str, Any]:
    return MEDIA_KINDS.get(session.get("kind") or DEFAULT_KIND, MEDIA_KINDS[DEFAULT_KIND])


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


def _bump_quota(session_id: str, kind: str) -> None:
    """Count a generation against today's budget.

    Not transactional. Object storage has no compare-and-set, so two requests
    landing in the same instant can both read the same count. The cap is a
    spend guard rather than a security boundary, and being off by one costs
    pennies, so a simple read and write is the right trade here.

    Video is counted twice over: once against the shared budget, because it is
    still a generation, and once against its own, because it is the expensive
    one and needs a limit that does not move when stills get busy.
    """
    quota = _read_quota()
    quota["generations"] = quota.get("generations", 0) + 1
    quota["sessions"][session_id] = quota["sessions"].get(session_id, 0) + 1
    if kind == "video":
        quota["videos"] = quota.get("videos", 0) + 1
        videos = quota.setdefault("video_sessions", {})
        videos[session_id] = videos.get(session_id, 0) + 1
    storage.client().put_object(
        Bucket=storage.bucket(),
        Key=_quota_key(),
        Body=json.dumps(quota).encode("utf-8"),
        ContentType="application/json",
    )


def refund_quota(session_id: str, kind: str) -> None:
    """Give a generation back when the run produced nothing at all.

    The budget is spent at submit, because a submit can bill even when the
    client never hears back. But a run where every candidate failed rendered
    nothing and charged nothing, and holding a charge for it is not a spend
    guard, it is a locked door: the video allowance is one run per browser, so
    a single failed render would otherwise end that visitor's demo for good.

    Only ever called once per run, and never below zero.
    """
    quota = _read_quota()
    quota["generations"] = max(0, quota.get("generations", 0) - 1)
    if quota["sessions"].get(session_id):
        quota["sessions"][session_id] = max(0, quota["sessions"][session_id] - 1)
    if kind == "video":
        quota["videos"] = max(0, quota.get("videos", 0) - 1)
        videos = quota.setdefault("video_sessions", {})
        if videos.get(session_id):
            videos[session_id] = max(0, videos[session_id] - 1)
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
    videos = quota.get("videos", 0)
    my_videos = (quota.get("video_sessions") or {}).get(session_id, 0) if session_id else 0

    overall = used < DAILY_GENERATION_CAP and mine < PER_SESSION_CAP
    return {
        "used_today": used,
        "daily_cap": DAILY_GENERATION_CAP,
        "remaining_today": max(0, DAILY_GENERATION_CAP - used),
        "used_this_session": mine,
        "session_cap": PER_SESSION_CAP,
        "can_generate": overall,
        "videos_today": videos,
        "video_daily_cap": DAILY_VIDEO_CAP,
        "videos_remaining_today": max(0, DAILY_VIDEO_CAP - videos),
        "videos_this_session": my_videos,
        "video_session_cap": PER_SESSION_VIDEO_CAP,
        "can_generate_video": (
            overall and videos < DAILY_VIDEO_CAP and my_videos < PER_SESSION_VIDEO_CAP
        ),
    }


def _quota_refusal(status: dict, kind: str) -> str | None:
    """Why this visitor cannot generate right now, in words they can act on."""
    if status["used_today"] >= DAILY_GENERATION_CAP:
        return (
            f"Today's generation budget is spent "
            f"({status['used_today']}/{status['daily_cap']})"
        )
    if status["used_this_session"] >= PER_SESSION_CAP:
        return f"This browser has used its {status['session_cap']} generations"
    if kind != "video":
        return None
    if status["videos_today"] >= DAILY_VIDEO_CAP:
        return (
            f"Today's video budget is spent "
            f"({status['videos_today']}/{status['video_daily_cap']}). "
            "Stills are still available"
        )
    if status["videos_this_session"] >= PER_SESSION_VIDEO_CAP:
        return (
            f"This browser has used its {status['video_session_cap']} video run. "
            "Stills are still available"
        )
    return None


def job_id_of(result: Any) -> str:
    """Pull the job id out of whatever a provider's submit handed back.

    MEASURED: the adapters do not agree on this. The image provider returns the
    id as a plain string; the video provider returns a ``SubmitResult`` object
    carrying ``prediction_id`` and an estimate. Calling ``str()`` on the second
    stores its repr, and polling that gets

        GMICloud poll failed (404): Request not found

    forever, while the render finishes and bills in the background. The failure
    is quiet in the worst way: the job is fine, the page just never learns.

    Read the id off the object rather than trusting either shape, and fall back
    to the string form only when there is nothing else to read.
    """
    for attribute in ("prediction_id", "request_id", "job_id", "id"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(result, dict):
        for attribute in ("prediction_id", "request_id", "job_id", "id"):
            value = result.get(attribute)
            if isinstance(value, str) and value:
                return value
    return str(result)


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


def start_generation(
    brief: str,
    session_id: str | None = None,
    style: str | None = None,
    kind: str | None = None,
) -> dict:
    """Submit the candidates and return immediately with their job ids."""
    from hallmark.providers import (
        IMAGE_MODEL,
        VIDEO_MODEL,
        image_provider,
        video_provider,
    )

    session_id = session_id or str(uuid.uuid4())
    style = style if style in STYLE_PRESETS else DEFAULT_STYLE
    kind = kind if kind in MEDIA_KINDS else DEFAULT_KIND
    preset = STYLE_PRESETS[style]
    spec = MEDIA_KINDS[kind]

    status = quota_status(session_id)
    refusal = _quota_refusal(status, kind)
    if refusal:
        raise QuotaExceeded(refusal)

    is_video = kind == "video"
    hint = preset["video_hint"] if is_video else preset["hint"]
    prompt = f"{brief.strip()}. {hint}"
    model = VIDEO_MODEL if is_video else IMAGE_MODEL
    modality = Modality.VIDEO if is_video else Modality.IMAGE
    provider_name = "gmicloud-video" if is_video else "gmicloud-image"
    provider = (
        video_provider(http_timeout=SUBMIT_TIMEOUT)
        if is_video
        else image_provider(http_timeout=SUBMIT_TIMEOUT)
    )
    count = spec["candidates"]
    batch = int(time.time()) % SEED_CYCLE_SECONDS

    def submit(index: int) -> Candidate:
        # Same prompt, different seed. The seed doubles as a marker: GMI
        # echoes the payload back on the queue listing, so a submit whose
        # response never arrives can still be matched to its job.
        seed = batch * 100 + index
        step = Step(
            step_type=StepType.GENERATE,
            modality=modality,
            provider=provider_name,
            model=model,
            prompt=prompt,
            prompt_visibility=PromptVisibility.PRIVATE,
            seed=seed,
        )
        try:
            submitted = provider.submit(step)
            return Candidate(
                index=index,
                job_id=job_id_of(submitted),
                seed=seed,
                media_type=spec["raw_mime"],
            )
        except Exception as exc:  # noqa: BLE001 - recovered below, not lost
            return Candidate(
                index=index, job_id="", seed=seed, status="unresolved",
                media_type=spec["raw_mime"], reason=str(exc)[:160],
            )

    with ThreadPoolExecutor(max_workers=count) as pool:
        candidates = list(pool.map(submit, range(count)))

    _recover_unresolved(candidates)
    _bump_quota(session_id, kind)

    previous = load_session(session_id) or {}

    session = {
        "session_id": session_id,
        # Numbered so each run keeps its own place in storage. Without this a
        # second run overwrote the first one's files and the page, reading a
        # long-lived cache under an unchanged URL, showed the old picture.
        "run_seq": int(previous.get("run_seq") or 0) + 1,
        "kind": kind,
        "kind_label": spec["label"],
        "brief": brief.strip(),
        "prompt": prompt,
        "style": style,
        "style_label": preset["label"],
        "model": model,
        "created_at": _now(),
        "status": "generating",
        "candidates": [c.__dict__ for c in candidates],
        "selection": None,
        "run_id": None,
        "manifest_uri": None,
        "canonical_hash": None,
        # Everything this browser has finished, oldest first. The marking
        # record is built over all of it; the page shows the current run.
        "runs": previous.get("runs") or [],
    }
    save_session(session)
    return session


def poll_generation(session_id: str) -> dict:
    """Check each outstanding job and record any that finished."""
    from hallmark.providers import image_provider, video_provider

    session = load_session(session_id)
    if session is None:
        raise KeyError(session_id)
    if session["status"] not in ("generating",):
        return session

    spec = _kind_of(session)
    is_video = session.get("kind") == "video"
    provider = video_provider() if is_video else image_provider()
    modality = Modality.VIDEO if is_video else Modality.IMAGE
    provider_name = "gmicloud-video" if is_video else "gmicloud-image"

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
            modality=modality,
            provider=provider_name,
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
            raw["reason"] = filled.error or f"no {spec['modality']} returned"
            return raw

        asset = filled.assets[0]
        raw["status"] = "ready"
        raw["url"] = str(asset.url)
        raw["media_type"] = asset.media_type or spec["raw_mime"]
        return raw

    with ThreadPoolExecutor(max_workers=max(1, len(session["candidates"]))) as pool:
        session["candidates"] = list(pool.map(check, session["candidates"]))

    states = {c["status"] for c in session["candidates"]}
    if states <= {"ready", "failed"}:
        session["status"] = "ready" if "ready" in states else "failed"

    # This is the first point at which a run is known to have produced nothing.
    # Flagged on the session so a browser polling a dead run repeatedly cannot
    # refund itself over and over.
    if session["status"] == "failed" and not session.get("refunded"):
        session["refunded"] = True
        try:
            refund_quota(session["session_id"], session.get("kind") or DEFAULT_KIND)
        except Exception:  # noqa: BLE001 - a refund is a courtesy, not the answer
            pass

    save_session(session)
    return session


RATIONALE_SYSTEM = """You write the note filed against each rejected candidate
in a creative review record.

The reviewer looked at every candidate and picked one. They wrote down why, in
their own words. Your whole job is to turn that reason around: say what each
rejected candidate did not deliver, in the reviewer's own terms.

You have NOT seen any image or clip. Never describe what one looks like.

Rules:
- Every line must be about the REVIEWER'S STATED REASON. If they picked one for
  "a good stance with very vivid features", every note is about stance and
  vividness. Not about file size, not about render time, not about scores.
- Attribute the judgement to the reviewer, because they are the one who looked:
  "the reviewer did not find", "read weaker to the reviewer on", "lost on".
  Never state a visual claim as your own observation.
- Word each line differently. Two candidates rejected on the same grounds still
  get two different sentences.
- Mention a measured fact ONLY when a check actually FAILED, and put it after
  the reviewer's reason, never in front of it.
- If the reviewer gave no reason, say plainly that they picked another on their
  own judgement and left no note. Do not invent a criterion.
- Under 20 words each.

Reply with JSON only: {"1": "line", "2": "line"} keyed by candidate index."""


def _fallback_rationales(session: dict, picked: int, human_reason: str) -> dict[str, str]:
    """Notes written without a model, still anchored to the reviewer's words.

    The chat call can fail, and when it does the rejection notes must not
    silently become something the reviewer never said. These templates say the
    same thing the model is asked to say, in fewer words.
    """
    reason = human_reason.strip().rstrip(".")
    if reason:
        shapes = [
            f"The reviewer was looking for {reason}, and did not find it here.",
            f"Lost on the reviewer's own test: {reason}.",
            f"Held against {reason}, this one came second.",
        ]
    else:
        shapes = [
            "The reviewer picked another on their own judgement and left no note.",
            "Not chosen. The reviewer gave no reason, so none is invented here.",
            "Passed over by the reviewer, with no stated grounds.",
        ]

    notes = {}
    position = 0
    for candidate in session["candidates"]:
        if candidate["index"] == picked:
            continue
        notes[str(candidate["index"])] = shapes[position % len(shapes)]
        position += 1
    return notes


def _rationales(session: dict, picked: int, human_reason: str) -> dict[str, str]:
    """Ask a text model to explain the rejections from the human's reason.

    The model is told plainly that it has not seen the renders, and it is given
    only the reviewer's words and the measured checks. That keeps the output
    honest: no model on this account can see an image, so anything describing
    what a picture looks like would be invented.

    The reviewer's reason leads. Earlier this leaned on the measured facts and
    the notes came back reading like a QA report, quoting render times at
    someone who had just said the pose was better. A rejection note has to
    answer the reason the human actually gave.
    """
    from genblaze_gmicloud import chat

    from hallmark.providers import CHAT_MODEL

    facts = []
    for c in session["candidates"]:
        if c["status"] != "ready":
            facts.append(f'candidate {c["index"]}: failed to generate')
            continue
        failed = [k for k in (c.get("checks") or []) if not k.get("passed")]
        note = (
            "all technical checks passed"
            if not failed
            else "FAILED checks: " + ", ".join(f'{k["name"]} ({k["detail"]})' for k in failed)
        )
        facts.append(
            f'candidate {c["index"]}: '
            f'{"CHOSEN by reviewer" if c["index"] == picked else "not chosen"}, {note}'
        )

    prompt = (
        f'Brief: {session["brief"]}\n'
        f'Medium: {session.get("kind_label", "still image")}\n'
        f'Style: {session.get("style_label", "unspecified")}\n'
        f"Reviewer picked candidate {picked}.\n"
        f'THE REVIEWER\'S REASON, in their words: "{human_reason.strip() or "none given"}"\n\n'
        "Write the rejection note for every other candidate, against that reason.\n\n"
        + "\n".join(facts)
    )

    fallback = _fallback_rationales(session, picked, human_reason)

    try:
        response = chat(model=CHAT_MODEL, prompt=prompt, system=RATIONALE_SYSTEM,
                        temperature=0.4, max_tokens=700)
        text = response.text or ""
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            notes = {str(k): str(v) for k, v in json.loads(text[start : end + 1]).items()}
            # A model that answers for only some candidates leaves the rest
            # with no note at all, which reads as an oversight rather than a
            # decision. Fill the gaps rather than showing a blank.
            return {index: notes.get(index) or default for index, default in fallback.items()}
    except Exception:  # noqa: BLE001 - the demo must not hinge on the planner
        pass

    return fallback


def asset_key(session_id: str, run_seq: int, index: int, chosen: bool, suffix: str) -> str:
    """Where one candidate lives.

    The run number is in the filename rather than in a folder so the streaming
    route can keep matching on the last path segment, and so two runs in one
    browser can never write to the same object.
    """
    role = "chosen" if chosen else "reject"
    return f"{SESSION_PREFIX}/{session_id}/r{run_seq}_{role}_{index}{suffix}"


def select_candidate(
    session_id: str,
    picked: int,
    human_reason: str,
    workdir: Path,
    signer: str = "Visitor",
) -> dict:
    """Store every candidate, sign the chosen one, and record the decision.

    Only the chosen candidate is signed. The rejects are uploaded exactly as
    they came back from the model, unsigned, because nobody approved them.
    Download one and its properties are bare, which is the honest state for a
    file no human ever passed.
    """
    from genblaze_core.models.asset import Asset
    from genblaze_core.models.policy import EmbedPolicy

    from hallmark import approval, compliance, credential, metadata
    from hallmark.approval import Approval
    from hallmark.evaluate import evaluate
    from hallmark.stamp import stamp

    session = load_session(session_id)
    if session is None:
        raise KeyError(session_id)

    spec = _kind_of(session)
    is_video = session.get("kind") == "video"
    modality = spec["modality"]
    # Written back, not just read, so a session started before runs were
    # numbered still records which run its assets belong to rather than
    # filing them under nothing.
    run_seq = int(session.get("run_seq") or 1)
    session["run_seq"] = run_seq

    ready = [c for c in session["candidates"] if c["status"] == "ready"]
    if not any(c["index"] == picked for c in ready):
        raise ValueError(f"Candidate {picked} is not available to pick")

    workdir.mkdir(parents=True, exist_ok=True)
    lanes = max(1, len(ready))

    def fetch(raw: dict) -> dict:
        """Pull the bytes, hash them, and run the technical checks."""
        local = workdir / f"cand_{raw['index']}{spec['raw_suffix']}"
        storage.download(raw["url"], local)
        data = local.read_bytes()

        raw["sha256"] = hashlib.sha256(data).hexdigest()
        raw["size_bytes"] = len(data)
        check = evaluate(local, modality)
        raw["checks"] = check.to_dict()["checks"]
        raw["score"] = check.score
        raw["passed"] = check.passed
        raw["local_path"] = str(local)
        # Kept separately because local_path is repointed at the delivered file
        # for whichever candidate wins, and the credential needs the render it
        # was converted from to record as its parent.
        raw["raw_path"] = str(local)
        return raw

    with ThreadPoolExecutor(max_workers=lanes) as pool:
        fetched = list(pool.map(fetch, ready))

    by_index = {c["index"]: c for c in fetched}

    signer = (signer or "Visitor").strip()[:60] or "Visitor"
    reason = human_reason.strip()[:400]

    # The chosen candidate becomes the delivered asset, carrying the approval
    # in metadata a file browser will show. The metadata is written BEFORE the
    # hash is taken, so the visible signature is inside the bytes the record
    # covers. Edit the signature out and the hash stops matching.
    #
    # A still is re-encoded to JPEG on the way, because Windows displays
    # nothing at all for a PNG. A clip keeps its container and has its property
    # table rewritten in place, since re-encoding it would cost minutes and
    # change every frame.
    chosen_raw = by_index[picked]
    delivery = workdir / f"delivery_{picked}{spec['delivery_suffix']}"
    credit = metadata.Signature(
        approver=signer,
        model=session["model"],
        note=reason or None,
        brief=session["brief"],
        verify_url=VERIFY_BASE,
        medium="video" if is_video else "image",
    )
    if is_video:
        metadata.to_mp4(Path(chosen_raw["local_path"]), delivery, credit)
    else:
        metadata.to_jpeg(Path(chosen_raw["local_path"]), delivery, credit)

    chosen_raw["local_path"] = str(delivery)
    chosen_raw["media_type"] = spec["delivery_mime"]
    chosen_raw["sha256"] = hashlib.sha256(delivery.read_bytes()).hexdigest()
    chosen_raw["size_bytes"] = delivery.stat().st_size

    # One record covering the whole run, so the rejected attempts are inside
    # the provenance rather than thrown away.
    step_modality = Modality.VIDEO if is_video else Modality.IMAGE
    provider_name = "gmicloud-video" if is_video else "gmicloud-image"
    steps = []
    for c in fetched:
        chosen = c["index"] == picked
        steps.append(
            Step(
                step_type=StepType.GENERATE,
                modality=step_modality,
                provider=provider_name,
                model=session["model"],
                prompt=session["prompt"],
                prompt_visibility=PromptVisibility.PRIVATE,
                assets=[
                    Asset(
                        url=c["url"],
                        media_type=c.get("media_type", spec["raw_mime"]),
                        sha256=c["sha256"],
                        size_bytes=c["size_bytes"],
                    )
                ],
                # The style is a preset name, so it is safe to publish. The
                # prompt it expands into is not, and stays withheld.
                params={"style": session.get("style_label", "")},
                metadata={"candidate": c["index"], "chosen": chosen},
            )
        )

    manifest = Manifest.from_run(Run(name=f"demo/{session['brief'][:48]}", steps=steps))
    approval.apply(
        manifest,
        Approval(approver=signer, decision="approved",
                 note=reason or "picked on the page"),
    )
    # Published under a key that carries the run number, for the same reason
    # the assets do: a second run must not overwrite the first run's record.
    manifest_uri = approval.publish_manifest(manifest, f"{session_id}/r{run_seq}")

    notes = _rationales(session, picked, human_reason)

    # Store every candidate, chosen or not. The rejects are the inventory.
    def store(c: dict) -> dict:
        local = Path(c["local_path"])
        chosen = c["index"] == picked
        mime = c.get("media_type", spec["raw_mime"])
        suffix = spec["delivery_suffix"] if chosen else spec["raw_suffix"]
        if chosen:
            stamped = workdir / f"chosen_{c['index']}{suffix}"
            stamp(
                local,
                manifest,
                stamped,
                policy=EmbedPolicy(embed_mode="pointer",
                                   prompt_visibility=PromptVisibility.PRIVATE),
                mime_type=mime,
            )
            local = stamped

            # Last, because the credential's signature has to cover our pointer
            # too. The render it was converted from goes in as the parent, so
            # the provider's signature survives the conversion rather than
            # being thrown away by the encoder.
            credentialed = workdir / f"chosen_{c['index']}_credentialed{suffix}"
            if credential.sign(stamped, credentialed, parent=Path(c["raw_path"]),
                               model=session["model"], approver=signer,
                               note=reason or None):
                local = credentialed

        key = asset_key(session_id, run_seq, c["index"], chosen, suffix)
        storage.upload(local, key, mime)
        # Read off the delivered file while it is still to hand. The sheet
        # re-measures the current run when it is opened, but an earlier run's
        # files would each be a fresh download inside a 60 second function, so
        # what was measured at delivery is what the record falls back to.
        c["marks"] = compliance.marks_for(local, mime)
        c["stored_key"] = key
        c["media_type"] = mime
        c["accepted"] = chosen
        c["reason"] = None if chosen else notes.get(str(c["index"]))
        c.pop("local_path", None)
        c.pop("raw_path", None)
        return c

    with ThreadPoolExecutor(max_workers=lanes) as pool:
        stored = list(pool.map(store, fetched))

    for c in session["candidates"]:
        if c["index"] in by_index:
            c.update(next(s for s in stored if s["index"] == c["index"]))

    session["status"] = "selected"
    session["selection"] = {
        "picked": picked,
        "human_reason": reason,
        "signer": signer,
        "decided_at": _now(),
    }
    session["run_id"] = manifest.run.run_id
    session["manifest_uri"] = manifest_uri
    session["canonical_hash"] = manifest.canonical_hash

    # Fold this run into the session's history, replacing any earlier write of
    # the same run so a repeated selection cannot double up.
    history = [r for r in (session.get("runs") or []) if r.get("run_seq") != run_seq]
    history.append(_run_record(session))
    session["runs"] = sorted(history, key=lambda r: r.get("run_seq") or 0)

    # Every candidate goes on the ledger, the two nobody picked included. The
    # page claims the rejects are kept; until now that was only true of runs
    # started from a terminal, so the published acceptance rate ignored every
    # asset the public generated.
    attempts.record(
        [
            {
                "run_id": manifest.run.run_id,
                "campaign": "demo",
                "modality": modality,
                "model": session["model"],
                "provider": "gmicloud",
                "accepted": bool(c["index"] == picked),
                "score": float(c.get("score") or 0.0),
                "latency_seconds": float(c.get("latency_seconds") or 0.0),
                "reject_reason": None if c["index"] == picked else c.get("reason"),
                "cost_usd": None,
                "sha256": c.get("sha256"),
                "size_bytes": c.get("size_bytes"),
                "media_type": c.get("media_type"),
                "checks": json.dumps(c.get("checks") or []),
            }
            for c in stored
        ]
    )

    save_session(session)
    return session


def _run_record(session: dict) -> dict[str, Any]:
    """One finished run, reduced to what the marking record needs.

    Held separately from the live candidate list because that list is replaced
    the moment the visitor generates again, and the paperwork is supposed to
    cover everything they made rather than only the last thing.
    """
    selection = session.get("selection") or {}
    picked = selection.get("picked")
    return {
        "run_seq": session.get("run_seq"),
        "run_id": session.get("run_id"),
        "kind": session.get("kind", DEFAULT_KIND),
        "kind_label": session.get("kind_label", ""),
        "brief": session.get("brief", ""),
        "style_label": session.get("style_label", ""),
        "model": session.get("model", ""),
        "manifest_uri": session.get("manifest_uri", ""),
        "canonical_hash": session.get("canonical_hash", ""),
        "selection": selection,
        "assets": [
            {
                "index": c.get("index"),
                "accepted": bool(c.get("index") == picked),
                "stored_key": c.get("stored_key"),
                "media_type": c.get("media_type", ""),
                "sha256": c.get("sha256", ""),
                "size_bytes": c.get("size_bytes", 0),
                "score": c.get("score"),
                "reason": c.get("reason"),
                "marks": c.get("marks") or {},
            }
            for c in session.get("candidates") or []
            if c.get("stored_key")
        ],
    }


def session_runs(session: dict) -> list[dict[str, Any]]:
    """Every finished run in a session, oldest first.

    Falls back to reconstructing the current run for sessions written before
    the history existed, so an old link still produces a full record instead of
    an empty one.
    """
    runs = list(session.get("runs") or [])
    if runs:
        return runs
    if session.get("status") == "selected":
        return [_run_record(session)]
    return []


def session_totals(session: dict) -> dict[str, int]:
    """How much this browser has made, for the page to show on step 7."""
    runs = session_runs(session)
    assets = [a for r in runs for a in r.get("assets") or []]
    return {
        "runs": len(runs),
        "assets": len(assets),
        "signed": sum(1 for a in assets if a.get("accepted")),
        "unsigned": sum(1 for a in assets if not a.get("accepted")),
    }
