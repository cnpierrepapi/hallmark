"""Technical quality checks for generated media.

These are deliberately deterministic. No model on this account can see an
image, so an "AI judges the creative" gate would be scoring something it
cannot observe. Instead the machine checks what a machine can actually
establish, and a human approves the creative.

What this catches is real: blank or near-uniform frames from a failed
generation, truncated or silent audio, video with no media data, files that do
not decode at all. Those are the failures that waste a reviewer's time, and
they are exactly what a production pipeline should filter before a person
looks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A greyscale standard deviation below this means the frame is essentially one
# flat colour, which is what a failed image generation usually returns.
MIN_PIXEL_STDDEV = 8.0
MIN_DIMENSION = 256
MIN_AUDIO_SECONDS = 0.8
MIN_VIDEO_SECONDS = 1.0

# Speech runs about 2.5 words per second, so a clip far shorter than the script
# means the audio was cut off.
WORDS_PER_SECOND = 2.5


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: Any = None


@dataclass
class Evaluation:
    modality: str
    passed: bool
    score: float
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def reason(self) -> str | None:
        if self.passed:
            return None
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "value": c.value}
                for c in self.checks
            ],
        }


def _score(checks: list[Check]) -> float:
    if not checks:
        return 0.0
    return round(sum(1 for c in checks if c.passed) / len(checks), 3)


def _finish(modality: str, checks: list[Check]) -> Evaluation:
    return Evaluation(
        modality=modality,
        passed=all(c.passed for c in checks),
        score=_score(checks),
        checks=checks,
    )


def evaluate_image(path: Path) -> Evaluation:
    from PIL import Image, ImageStat

    checks: list[Check] = []
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            stat = ImageStat.Stat(img.convert("L"))
            stddev = stat.stddev[0]
    except Exception as exc:  # noqa: BLE001 - a broken file is a failed check
        return _finish("image", [Check("decodes", False, f"could not decode: {exc}")])

    checks.append(Check("decodes", True, "image decoded", f"{width}x{height}"))
    checks.append(
        Check(
            "resolution",
            width >= MIN_DIMENSION and height >= MIN_DIMENSION,
            f"{width}x{height}, minimum {MIN_DIMENSION}",
            [width, height],
        )
    )
    checks.append(
        Check(
            "not_blank",
            stddev >= MIN_PIXEL_STDDEV,
            f"pixel spread {stddev:.1f}, minimum {MIN_PIXEL_STDDEV}",
            round(stddev, 2),
        )
    )
    return _finish("image", checks)


def _mp4_duration(data: bytes) -> float | None:
    """Read duration from the mvhd box, without a media library."""
    index = data.find(b"mvhd")
    if index == -1:
        return None
    start = index + 4
    if start + 20 > len(data):
        return None
    version = data[start]
    try:
        if version == 1:
            timescale = struct.unpack(">I", data[start + 20 : start + 24])[0]
            duration = struct.unpack(">Q", data[start + 24 : start + 32])[0]
        else:
            timescale = struct.unpack(">I", data[start + 12 : start + 16])[0]
            duration = struct.unpack(">I", data[start + 16 : start + 20])[0]
    except struct.error:
        return None
    return duration / timescale if timescale else None


def evaluate_video(path: Path) -> Evaluation:
    data = path.read_bytes()
    checks: list[Check] = []

    is_mp4 = len(data) > 12 and data[4:8] == b"ftyp"
    checks.append(Check("container", is_mp4, "ftyp box present" if is_mp4 else "not an MP4"))
    if not is_mp4:
        return _finish("video", checks)

    has_media = b"mdat" in data
    checks.append(
        Check("has_media_data", has_media, "mdat present" if has_media else "no mdat box")
    )

    duration = _mp4_duration(data)
    if duration is None:
        checks.append(Check("duration", False, "could not read duration from mvhd"))
    else:
        checks.append(
            Check(
                "duration",
                duration >= MIN_VIDEO_SECONDS,
                f"{duration:.1f}s, minimum {MIN_VIDEO_SECONDS}s",
                round(duration, 2),
            )
        )
    return _finish("video", checks)


def evaluate_audio(path: Path, script: str | None = None) -> Evaluation:
    from mutagen.mp3 import MP3

    checks: list[Check] = []
    try:
        audio = MP3(path)
        duration = float(audio.info.length)
    except Exception as exc:  # noqa: BLE001 - a broken file is a failed check
        return _finish("audio", [Check("decodes", False, f"could not decode: {exc}")])

    checks.append(Check("decodes", True, "audio decoded", round(duration, 2)))
    checks.append(
        Check(
            "duration",
            duration >= MIN_AUDIO_SECONDS,
            f"{duration:.1f}s, minimum {MIN_AUDIO_SECONDS}s",
            round(duration, 2),
        )
    )

    if script:
        words = len(script.split())
        expected = words / WORDS_PER_SECOND
        # Generous floor: only catches audio cut off well short of the script.
        complete = duration >= expected * 0.5
        checks.append(
            Check(
                "covers_script",
                complete,
                f"{duration:.1f}s for {words} words, expected about {expected:.1f}s",
                round(expected, 2),
            )
        )

    return _finish("audio", checks)


def evaluate(path: Path, modality: str, script: str | None = None) -> Evaluation:
    if modality == "image":
        return evaluate_image(path)
    if modality == "video":
        return evaluate_video(path)
    if modality == "audio":
        return evaluate_audio(path, script)
    return _finish(modality, [Check("known_modality", False, f"no checks for {modality}")])
