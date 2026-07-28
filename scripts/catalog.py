"""Print the GMI Cloud model catalog exposed through Genblaze.

Run this before picking models for a pipeline so the choice is made against
what the adapter actually supports, not against a blog post.

    python scripts/catalog.py
"""

from __future__ import annotations

from genblaze_gmicloud import (
    GMICloudAudioProvider,
    GMICloudImageProvider,
    GMICloudVideoProvider,
)

PROVIDERS = (
    ("image", GMICloudImageProvider),
    ("video", GMICloudVideoProvider),
    ("audio", GMICloudAudioProvider),
)


def _as_list(value):
    if value is None:
        return []
    if callable(value):
        value = value()
    return list(value)


def main() -> None:
    for modality, provider_cls in PROVIDERS:
        provider = provider_cls(api_key="offline-catalog-probe")
        registry = provider.models
        print(f"=== {modality} ({provider_cls.__name__})")

        families = _as_list(getattr(registry, "families", None))
        print(f"  families ({len(families)}):")
        for family in families:
            print(f"    {family}")

        try:
            items = _as_list(getattr(registry, "items", None))
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"  items unavailable: {type(exc).__name__}: {exc}")
            items = []

        print(f"  models ({len(items)}):")
        for entry in items:
            slug = entry[0] if isinstance(entry, tuple) else entry
            print(f"    {slug}")
        print()


if __name__ == "__main__":
    main()
