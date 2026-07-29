"""Provider construction, including a fix for GMI text to speech.

genblaze-gmicloud's TTS family sends the step prompt as ``prompt``, but GMI's
TTS models require ``text``. The submit is rejected with

    400 invalid payload parameters: text (Required parameter is missing)

and passing ``text=`` directly does not help, because ``text`` is absent from
the family's allowlist and gets dropped before the request is built. That makes
MiniMax TTS unreachable through the stock adapter.

Registering an explicit model spec fixes it without forking the SDK: alias the
prompt onto ``text`` and allow the fields GMI actually documents.
"""

from __future__ import annotations

import os

from genblaze_core import ModelRegistry, ModelSpec, Modality
from genblaze_gmicloud import (
    GMICloudAudioProvider,
    GMICloudImageProvider,
    GMICloudVideoProvider,
)

# GMI's POST /requests sometimes holds the connection until the job finishes
# rather than returning a job id. When the client gives up first, the work
# still runs and still bills, so a short timeout orphans paid work.
SUBMIT_TIMEOUT = 300.0

IMAGE_MODEL = "gpt-image-2-generate"
VIDEO_MODEL = "wan2.7-t2v"
AUDIO_MODEL = "minimax-tts-speech-2.6-turbo"
CHAT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

DEFAULT_VOICE = "English_expressive_narrator"

_TTS_ALLOWLIST = frozenset(
    {
        "text",
        "voice_id",
        "speed",
        "vol",
        "pitch",
        "emotion",
        "language_boost",
        "format",
        "audio_sample_rate",
        "bitrate",
        "channel",
    }
)


def tts_registry() -> ModelRegistry:
    """A registry whose TTS spec speaks GMI's actual payload shape."""
    registry = GMICloudAudioProvider.create_registry()
    registry.register(
        ModelSpec(
            model_id=AUDIO_MODEL,
            modality=Modality.AUDIO,
            param_aliases={"prompt": "text", "voice": "voice_id"},
            param_allowlist=_TTS_ALLOWLIST,
            param_defaults={"voice_id": DEFAULT_VOICE, "format": "mp3"},
            extras={"envelope_key": "payload", "is_music": False},
        )
    )
    return registry


def image_provider(http_timeout: float = SUBMIT_TIMEOUT) -> GMICloudImageProvider:
    return GMICloudImageProvider(http_timeout=http_timeout)


def video_provider(http_timeout: float = SUBMIT_TIMEOUT) -> GMICloudVideoProvider:
    """A video provider.

    The timeout is a parameter for the same reason it is on the image one: the
    demo submits from inside a function that dies at 60 seconds, so it gives up
    on a held-open submit early and recovers the job id from the queue listing
    instead. A terminal run has no such deadline and keeps the long default.
    """
    return GMICloudVideoProvider(http_timeout=http_timeout)


def audio_provider() -> GMICloudAudioProvider:
    return GMICloudAudioProvider(http_timeout=SUBMIT_TIMEOUT, models=tts_registry())


def require_api_key() -> str:
    key = os.environ.get("GMI_API_KEY")
    if not key:
        raise RuntimeError("GMI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key
