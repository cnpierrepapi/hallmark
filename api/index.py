"""Vercel entry point.

Vercel's Python runtime discovers ASGI applications exported from files under
api/. The application itself lives in hallmark.api so it stays independent of
any one host.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.api import app  # noqa: E402

__all__ = ["app"]
