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

from hallmark import storage
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
class StampedAsset:
    modality: str
    model: str
    media_type: str
    sha256: str
    stored_url: str
    stamped_url: str
    embed_mode: str
    local_path: Path


@dataclass
class CampaignResult:
    run_id: str
    brief: Brief
    plan: Plan
    manifest_uri: str | None
    canonical_hash: str
    assets: list[StampedAsset] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "brief": self.brief.__dict__,
            "plan": self.plan.__dict__,
            "manifest_uri": self.manifest_uri,
            "canonical_hash": self.canonical_hash,
            "assets": [
                {
                    "modality": a.modality,
                    "model": a.model,
                    "media_type": a.media_type,
                    "sha256": a.sha256,
                    "stored_url": a.stored_url,
                    "stamped_url": a.stamped_url,
                    "embed_mode": a.embed_mode,
                }
                for a in self.assets
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


def _stamp_and_store(manifest: Manifest, step, run_id: str, workdir: Path) -> StampedAsset | None:
    asset = step.assets[0]
    media_type = asset.media_type or "application/octet-stream"
    suffix = EXTENSIONS.get(media_type, ".bin")

    raw = workdir / f"{step.modality.value}_raw{suffix}"
    storage.download(str(asset.url), raw)

    stamped = workdir / f"{step.modality.value}_stamped{suffix}"
    mode = stamp(
        raw,
        manifest,
        stamped,
        policy=EmbedPolicy(embed_mode="pointer", prompt_visibility=PromptVisibility.PRIVATE),
        mime_type=media_type,
    )

    key = f"campaigns/{run_id}/{step.modality.value}{suffix}"
    stamped_url = storage.upload(stamped, key, media_type)

    return StampedAsset(
        modality=step.modality.value,
        model=step.model,
        media_type=media_type,
        sha256=asset.sha256 or "",
        stored_url=str(asset.url),
        stamped_url=stamped_url,
        embed_mode=mode,
        local_path=stamped,
    )


def run_campaign(brief: Brief, workdir: Path | None = None) -> CampaignResult:
    """Generate, store and stamp a full campaign."""
    plan = plan_campaign(brief)
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hallmark-"))
    workdir.mkdir(parents=True, exist_ok=True)

    result = (
        Pipeline("hallmark-campaign")
        .step(
            image_provider(),
            model=IMAGE_MODEL,
            prompt=plan.image_prompt,
            modality=Modality.IMAGE,
            prompt_visibility=PromptVisibility.PRIVATE,
        )
        .step(
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
        .run(sink=storage.sink(), timeout=900, raise_on_failure=False)
    )

    manifest = result.manifest
    campaign = CampaignResult(
        run_id=result.run.run_id,
        brief=brief,
        plan=plan,
        manifest_uri=manifest.manifest_uri,
        canonical_hash=manifest.canonical_hash,
    )

    for step in result.run.steps:
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
            stamped = _stamp_and_store(manifest, step, result.run.run_id, workdir)
        except Exception as exc:  # noqa: BLE001 - one bad asset must not lose the rest
            campaign.failures.append(
                {"modality": step.modality.value, "model": step.model, "error": str(exc)}
            )
            continue
        if stamped:
            campaign.assets.append(stamped)

    return campaign
