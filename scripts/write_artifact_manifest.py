#!/usr/bin/env python3
"""Write a deterministic SHA-256 inventory for frozen experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        action="append",
        type=Path,
        required=True,
        help="File or directory to inventory; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _files(paths: list[Path], *, excluded: Path) -> list[Path]:
    def is_auditable_file(item: Path) -> bool:
        return (
            item.is_file()
            and item.suffix not in {".pyc", ".pyo"}
            and "__pycache__" not in item.parts
            and item.name != ".DS_Store"
        )

    selected: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            selected.add(path)
        elif path.is_dir():
            selected.update(
                item for item in path.rglob("*") if is_auditable_file(item)
            )
        else:
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
    selected.discard(excluded)
    return sorted(selected, key=_display_path)


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in _files(args.include, excluded=output):
        entries.append(
            {
                "path": _display_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    canonical_entries = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    manifest = {
        "schema_version": "artifact_inventory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "entries_sha256": hashlib.sha256(canonical_entries).hexdigest(),
        "entries": entries,
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": manifest["file_count"],
                "total_size_bytes": manifest["total_size_bytes"],
                "entries_sha256": manifest["entries_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
