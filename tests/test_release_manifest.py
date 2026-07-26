from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_release_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_release_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_describes_head_without_worktree_state():
    module = load_module()
    manifest = module.render("HEAD")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=ROOT, text=True
    ).splitlines()

    assert f"| peeled commit | `{commit}` |" in manifest
    assert f"| tracked blobs | {len(tracked)} |" in manifest
    assert "| `README.md` |" in manifest
