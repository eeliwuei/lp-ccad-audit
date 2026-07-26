#!/usr/bin/env python3
"""Generate a deterministic Markdown manifest for an existing Git ref.

The manifest is written to stdout so callers can place it outside the
repository. It hashes the blobs stored in Git, not working-tree files.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], text=text)


@dataclass(frozen=True)
class Entry:
    path: str
    size: int
    sha256: str


def resolve_ref(ref: str) -> tuple[str, str, str]:
    object_id = git("rev-parse", "--verify", ref).strip()
    object_type = git("cat-file", "-t", object_id).strip()
    commit_id = git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    return object_id, object_type, commit_id


def entries_for(ref: str) -> list[Entry]:
    raw = git("ls-tree", "-r", "-z", "-l", ref, text=False)
    entries: list[Entry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, object_type, object_id, raw_size = metadata.split()
        if object_type != b"blob":
            continue
        content = git("cat-file", "blob", object_id.decode(), text=False)
        size = int(raw_size)
        if len(content) != size:
            raise RuntimeError(f"Git reported {size} bytes for {raw_path!r}, read {len(content)}")
        entries.append(
            Entry(
                path=raw_path.decode("utf-8", errors="surrogateescape"),
                size=size,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return entries


def escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def render(ref: str) -> str:
    object_id, object_type, commit_id = resolve_ref(ref)
    entries = entries_for(ref)
    lines = [
        f"# Public release manifest — `{escape_cell(ref)}`",
        "",
        "> Generated from Git objects, not from a working tree. Record the object",
        "> IDs below: a tag name can be force-moved by a privileged user.",
        "",
        "| field | value |",
        "|---|---|",
        f"| ref | `{escape_cell(ref)}` |",
        f"| ref object type | `{object_type}` |",
        f"| ref object | `{object_id}` |",
        f"| peeled commit | `{commit_id}` |",
        f"| tracked blobs | {len(entries)} |",
        f"| total bytes | {sum(entry.size for entry in entries):,} |",
        "",
        "## File table",
        "",
        "| path | bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| `{escape_cell(entry.path)}` | {entry.size:,} | `{entry.sha256}` |"
        for entry in entries
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Existing tag, commit or other Git ref")
    args = parser.parse_args()
    print(render(args.ref), end="")


if __name__ == "__main__":
    main()
