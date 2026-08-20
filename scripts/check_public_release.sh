#!/usr/bin/env bash
set -euo pipefail

release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$release_root"

candidate_manifest=$(mktemp)
trap 'rm -f "$candidate_manifest"' EXIT
git ls-files --cached --others --exclude-standard -z > "$candidate_manifest"

python - "$candidate_manifest" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1]).read_bytes().split(b"\0")
paths = [Path(raw.decode("utf-8")) for raw in manifest if raw]
if not paths:
    raise SystemExit("release tree is empty")

banned_top_level = {
    ".references",
    "data",
    "dataset",
    "outputs",
    "scratch",
    "build",
    "dist",
    "demo_output",
}
banned_suffixes = {
    ".bvh",
    ".fbx",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
    ".zst",
}

errors: list[str] = []
for path in paths:
    if path.parts and path.parts[0] in banned_top_level:
        errors.append(f"banned top-level path: {path}")
    if path.suffix.lower() in banned_suffixes:
        errors.append(f"banned data/model artifact: {path}")
    if path.is_symlink():
        errors.append(f"symlink is not allowed: {path}")
    if not path.is_file():
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"unexpected binary file: {path}")
        continue

    machine_tokens = (
        "/" + "iridisfs",
        "/" + "scratch" + "/",
    )
    for token in machine_tokens:
        if token in text:
            errors.append(f"machine-local path in {path}: {token}")
    home_pattern = re.compile("/" + "home" + r"/[A-Za-z0-9._-]+")
    host_pattern = re.compile(r"login[A-Za-z0-9._-]*\.cluster\.local")
    secret_pattern = re.compile(
        r"(?i)(?:hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|"
        r"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
    )
    if home_pattern.search(text):
        errors.append(f"home path in {path}")
    if host_pattern.search(text):
        errors.append(f"cluster hostname in {path}")
    if secret_pattern.search(text):
        errors.append(f"credential-like content in {path}")

if errors:
    print("public release check failed:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print(f"public release check passed: {len(paths)} files")
PY

git diff --check
git diff --cached --check
