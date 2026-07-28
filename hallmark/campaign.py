"""Brief in, stamped campaign out.

One brief becomes three assets across three modalities, each stored in B2 and
each carrying a provenance pointer inside the file itself.

Prompts are marked private throughout. The full record, prompts included, goes
to the private bucket; what travels with the asset is only a hash and a
reference. So an agency can prove an asset is AI generated without publishing
the creative work that produced it.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genblaze_core import Modality, Pipeline
from genblaze_core.models.enums import PromptVisibility
from genblaze_core.models.manifest import Manifest
from genblaze_core.models.policy import EmbedPolicy
from genblaze_gmicloud import chat

from hallmark import approval, ledger, storage
from hallmark.approval import Approval
from hallmark.evaluate import Evaluation, evaluate
from hallmark.ledger import Attempt
from hallmark.providers import (
    AUDIO_MODEL,
    CHAT_MODEL,
    IMAGE_MODEL,
    VIDEO_MODEL,
    audio_provider,
    image_provider,
    video_provider,
)
from hallmark.stamp import stamp

PLANNER_SYSTEM = """You write briefs for AI media generation. Given a product brief, \
produce three things:

1. image_prompt: a single vivid still image for a social ad. Describe subject, \
setting, lighting and mood. No text or logos in the image.
2. video_prompt: a short cinematic shot, five seconds, that suits the same campaign. \
Describe camera movement and action.
3. voiceover: one or two spoken sentences for the ad. Plain spoken English, under \
thirty words, no stage directions.

Reply with JSON only, with exactly the keys image_prompt, video_prompt and voiceover."""

EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "video/mp4": ".mp4",
    "audio/mpeg": ".mp3",
}


@dataclass
class Brief:
    product: str
    audience: str = "general"
    tone: str = "warm and confident"
    market: str = "UK"

    def as_prompt(self) -> str:
        return (
            f"Product: {self.product}\n"
            f"Audience: {self.audience}\n"
            f"Tone: {self.tone}\n"
            f"Market: {self.market}"
        )


@dataclass
class Plan:
    image_prompt: str
    video_prompt: str
    voiceover: str

    @classmethod
    def fallback(cls, brief: Brief) -> Plan:
        """Used when the planner is unavailable or returns unusable output.

        A campaign that degrades to literal prompts is far better than one that
        fails outright, and the provenance record still describes exactly what
        was sent to each model.
        """
        return cls(
            image_prompt=(
                f"A striking product photograph of {brief.product}, "
                f"{brief.tone} mood, natural light, shallow depth of field"
            ),
            video_prompt=(
                f"A slow cinematic push in on {brief.product}, "
                f"{brief.tone} atmosphere, soft natural light"
            ),
            voiceover=f"{brief.product}. Made for people who notice the difference.",
        )


@dataclass
class Candidate:
    """One generation attempt, whether it survived selection or not."""

    modality: str
    model: str
    media_type: str
    sha256: str
    stored_url: str
    local_path: Path
    evaluation: Evaluation
    latency_seconds: float
    cost_usd: float | None
    size_bytes: int
    accepted: bool = False
    reject_reason: str | None = None


@dataclass
class StampedAsset:
    modality: str
    model: str
    media_type: str
    sha256: str
    stored_url: str
    stamped_url: str
    embed_mode: str
    local_path: Path
    score: float


@dataclass
class CampaignResult:
    run_id: str
    brief: Brief
    plan: Plan
    manifest_uri: str | None
    canonical_hash: str
    status: str = "pending_approval"
    approval: dict[str, Any] | None = None
    assets: list[StampedAsset] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    ledger_key: str | None = None

    @property
    def rejected(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.accepted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "approval": self.approval,
            "brief": self.brief.__dict__,
            "plan": self.plan.__dict__,
            "manifest_uri": self.manifest_uri,
            "canonical_hash": self.canonical_hash,
            "ledger_key": self.ledger_key,
            "assets": [
                {
                    "modality": a.modality,
                    "model": a.model,
                    "media_type": a.media_type,
                    "sha256": a.sha256,
                    "stored_url": a.stored_url,
                    "stamped_url": a.stamped_url,
                    "embed_mode": a.embed_mode,
                    "score": a.score,
                }
                for a in self.assets
            ],
            "candidates": [
                {
                    "modality": c.modality,
                    "model": c.model,
                    "accepted": c.accepted,
                    "score": c.evaluation.score,
                    "passed": c.evaluation.passed,
                    "reject_reason": c.reject_reason,
                    "latency_seconds": round(c.latency_seconds, 2),
                    "cost_usd": c.cost_usd,
                    "size_bytes": c.size_bytes,
                    "sha256": c.sha256,
                    "checks": c.evaluation.to_dict()["checks"],
                }
                for c in self.candidates
            ],
            "failures": self.failures,
        }


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a chat reply.

    Models wrap JSON in prose or fences often enough that insisting on clean
    output would make the planner needlessly brittle.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def plan_campaign(brief: Brief) -> Plan:
    """Turn a brief into three prompts using an open-weight chat model."""
    try:
        response = chat(
            model=CHAT_MODEL,
            prompt=brief.as_prompt(),
            system=PLANNER_SYSTEM,
            temperature=0.7,
        )
    except Exception:  # noqa: BLE001 - planning must never sink the campaign
        return Plan.fallback(brief)

    data = _extract_json(response.text or "")
    if not data:
        return Plan.fallback(brief)

    default = Plan.fallback(brief)
    return Plan(
        image_prompt=str(data.get("image_prompt") or default.image_prompt),
        video_prompt=str(data.get("video_prompt") or default.video_prompt),
        voiceover=str(data.get("voiceover") or default.voiceover),
    )


def _step_latency(step) -> float:
    if step.started_at and step.completed_at:
        return (step.completed_at - step.started_at).total_seconds()
    return 0.0


def _collect_candidate(step, index: int, plan: Plan, workdir: Path) -> Candidate:
    """Download an attempt and run the technical checks over it."""
    asset = step.assets[0]
    media_type = asset.media_type or "application/octet-stream"
    suffix = EXTENSIONS.get(media_type, ".bin")
    modality = step.modality.value

    raw = workdir / f"{modality}_{index}_raw{suffix}"
    storage.download(str(asset.url), raw)

    script = plan.voiceover if modality == "audio" else None
    evaluation = evaluate(raw, modality, script=script)

    return Candidate(
        modality=modality,
        model=step.model,
        media_type=media_type,
        sha256=asset.sha256 or "",
        stored_url=str(asset.url),
        local_path=raw,
        evaluation=evaluation,
        latency_seconds=_step_latency(step),
        cost_usd=step.cost_usd,
        size_bytes=raw.stat().st_size,
    )


def _select(candidates: list[Candidate]) -> Candidate | None:
    """Pick the best passing attempt per modality, marking the rest rejected."""
    passing = [c for c in candidates if c.evaluation.passed]
    if not passing:
        for candidate in candidates:
            candidate.reject_reason = candidate.evaluation.reason or "failed quality checks"
        return None

    winner = max(passing, key=lambda c: (c.evaluation.score, -c.latency_seconds))
    for candidate in candidates:
        if candidate is winner:
            candidate.accepted = True
        else:
            candidate.reject_reason = (
                candidate.evaluation.reason or "a higher scoring attempt was selected"
            )
    return winner


def _stamp_and_store(
    manifest: Manifest, candidate: Candidate, run_id: str, workdir: Path
) -> StampedAsset:
    suffix = EXTENSIONS.get(candidate.media_type, ".bin")
    stamped = workdir / f"{candidate.modality}_stamped{suffix}"

    mode = stamp(
        candidate.local_path,
        manifest,
        stamped,
        policy=EmbedPolicy(embed_mode="pointer", prompt_visibility=PromptVisibility.PRIVATE),
        mime_type=candidate.media_type,
    )

    key = f"campaigns/{run_id}/{candidate.modality}{suffix}"
    stamped_url = storage.upload(stamped, key, candidate.media_type)

    return StampedAsset(
        modality=candidate.modality,
        model=candidate.model,
        media_type=candidate.media_type,
        sha256=candidate.sha256,
        stored_url=candidate.stored_url,
        stamped_url=stamped_url,
        embed_mode=mode,
        local_path=stamped,
        score=candidate.evaluation.score,
    )


def generate(brief: Brief, workdir: Path | None = None, image_candidates: int = 2) -> CampaignResult:
    """Generate and score a campaign. Nothing is stamped until it is approved.

    More than one image is generated because images are cheap and fast, so
    selection is worth doing. Video is neither, so it runs once.
    """
    plan = plan_campaign(brief)
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hallmark-"))
    workdir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline("hallmark-campaign")
    for _ in range(max(1, image_candidates)):
        pipeline = pipeline.step(
            image_provider(),
            model=IMAGE_MODEL,
            prompt=plan.image_prompt,
            modality=Modality.IMAGE,
            prompt_visibility=PromptVisibility.PRIVATE,
        )
    result = (
        pipeline.step(
            video_provider(),
            model=VIDEO_MODEL,
            prompt=plan.video_prompt,
            modality=Modality.VIDEO,
            prompt_visibility=PromptVisibility.PRIVATE,
        )
        .step(
            audio_provider(),
            model=AUDIO_MODEL,
            prompt=plan.voiceover,
            modality=Modality.AUDIO,
            prompt_visibility=PromptVisibility.PRIVATE,
        )
        .run(sink=storage.sink(), timeout=1200, raise_on_failure=False)
    )

    manifest = result.manifest
    campaign = CampaignResult(
        run_id=result.run.run_id,
        brief=brief,
        plan=plan,
        manifest_uri=manifest.manifest_uri,
        canonical_hash=manifest.canonical_hash,
    )
    campaign._manifest = manifest  # type: ignore[attr-defined]
    campaign._workdir = workdir  # type: ignore[attr-defined]

    by_modality: dict[str, list[Candidate]] = {}
    for index, step in enumerate(result.run.steps):
        if not step.assets or step.status.value != "succeeded":
            campaign.failures.append(
                {
                    "modality": step.modality.value,
                    "model": step.model,
                    "error": step.error or "no asset returned",
                }
            )
            continue
        try:
            candidate = _collect_candidate(step, index, plan, workdir)
        except Exception as exc:  # noqa: BLE001 - one bad asset must not lose the rest
            campaign.failures.append(
                {"modality": step.modality.value, "model": step.model, "error": str(exc)}
            )
            continue
        by_modality.setdefault(candidate.modality, []).append(candidate)

    winners: dict[str, Candidate] = {}
    for modality, candidates in by_modality.items():
        winner = _select(candidates)
        if winner:
            winners[modality] = winner
        campaign.candidates.extend(candidates)

    campaign._winners = winners  # type: ignore[attr-defined]

    campaign.ledger_key = ledger.write(
        [
            Attempt(
                run_id=campaign.run_id,
                campaign=brief.product,
                modality=c.modality,
                model=c.model,
                provider="gmicloud",
                accepted=c.accepted,
                score=c.evaluation.score,
                latency_seconds=c.latency_seconds,
                reject_reason=c.reject_reason,
                cost_usd=c.cost_usd,
                sha256=c.sha256,
                size_bytes=c.size_bytes,
                media_type=c.media_type,
                checks=json.dumps(c.evaluation.to_dict()["checks"]),
            )
            for c in campaign.candidates
        ]
    )

    return campaign


def approve(campaign: CampaignResult, approver: str, note: str | None = None) -> CampaignResult:
    """Sign off a campaign, then stamp and publish the selected assets.

    The approval goes into the record before the hash is recomputed, so the
    published files carry proof of who approved them, not just which model
    produced them.
    """
    manifest: Manifest = campaign._manifest  # type: ignore[attr-defined]
    workdir: Path = campaign._workdir  # type: ignore[attr-defined]
    winners: dict[str, Candidate] = campaign._winners  # type: ignore[attr-defined]

    if not winners:
        campaign.status = "rejected"
        campaign.approval = {"decision": "rejected", "reason": "no attempt passed the checks"}
        return campaign

    decision = Approval(approver=approver, decision="approved", note=note)
    approval.apply(manifest, decision)
    campaign.manifest_uri = approval.publish_manifest(manifest, campaign.run_id)
    campaign.canonical_hash = manifest.canonical_hash
    campaign.approval = approval.read(manifest)

    for candidate in winners.values():
        try:
            campaign.assets.append(
                _stamp_and_store(manifest, candidate, campaign.run_id, workdir)
            )
        except Exception as exc:  # noqa: BLE001 - one bad asset must not lose the rest
            campaign.failures.append(
                {"modality": candidate.modality, "model": candidate.model, "error": str(exc)}
            )

    campaign.status = "approved" if campaign.assets else "failed"
    return campaign


def run_campaign(
    brief: Brief,
    workdir: Path | None = None,
    approver: str | None = None,
    image_candidates: int = 2,
) -> CampaignResult:
    """Generate, and optionally approve in the same call."""
    campaign = generate(brief, workdir=workdir, image_candidates=image_candidates)
    if approver:
        campaign = approve(campaign, approver=approver)
    return campaign
