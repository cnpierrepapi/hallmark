"""Check which advertised genblaze features are actually usable as installed."""

from __future__ import annotations

import importlib
import shutil


def check(label: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - reporting tool
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {label}")


def main() -> None:
    print("CLI entry point:")
    found = shutil.which("genblaze")
    print(f"  {'ok    found at ' + found if found else 'FAIL  no genblaze executable on PATH'}")

    print("\nImports:")
    check("genblaze.ParquetSink", lambda: getattr(importlib.import_module("genblaze"), "ParquetSink"))
    check(
        "genblaze_core.sinks.parquet",
        lambda: importlib.import_module("genblaze_core.sinks.parquet"),
    )
    check("pyarrow", lambda: importlib.import_module("pyarrow"))

    print("\nInstantiation:")

    def build_parquet():
        genblaze = importlib.import_module("genblaze")
        genblaze.ParquetSink("out/ledger.parquet")

    check("ParquetSink(...)", build_parquet)


if __name__ == "__main__":
    main()
