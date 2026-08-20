#!/usr/bin/env bash
set -euo pipefail

release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$release_root"

python - <<'PY'
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath


def git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(("git", *arguments))


tracked = {
    value.decode("utf-8")
    for value in git_bytes("ls-files", "--cached", "-z").split(b"\0")
    if value
}
untracked = {
    value.decode("utf-8")
    for value in git_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    if value
}
paths = sorted(tracked | untracked)
if not paths:
    raise SystemExit("release tree is empty")

index_modes: dict[str, str] = {}
for value in git_bytes("ls-files", "--stage", "-z").split(b"\0"):
    if not value:
        continue
    header, raw_path = value.split(b"\t", 1)
    index_modes[raw_path.decode("utf-8")] = header.split()[0].decode("ascii")

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
banned_parts = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".ipynb_checkpoints",
}
banned_names = {
    ".env",
    ".netrc",
    "credentials",
    "credentials.json",
    "token",
    "secrets.json",
}
banned_suffixes = {
    ".arrow",
    ".bin",
    ".bvh",
    ".ckpt",
    ".csv",
    ".fbx",
    ".gif",
    ".glb",
    ".h5",
    ".hdf5",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".log",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".webm",
    ".whl",
    ".zip",
    ".zst",
}
machine_roots = "|".join(
    ("home", "Users", "scratch", "iridisfs", "mnt", "tmp", "opt", "data")
)
posix_machine_path = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:" + machine_roots + r")(?:/|$)"
)
windows_machine_path = re.compile(
    r"(?i)\b[A-Za-z]:[\\/](?:Users|home|data|scratch|tmp|mnt)(?:[\\/]|$)"
)
unc_machine_path = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/][A-Za-z0-9._$-]+")
file_uri = re.compile(r"(?i)file" + r"://")
machine_hostname = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)+(?:local|internal|lan|cluster|corp|example)\b"
)
credential_value = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|access[_-]?key|secret)"
    r"\s*[:=]\s*['\"]?[^\s'\"]+"
)
credential_token = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
)
absolute_default = re.compile(
    r"default\s*=\s*(?:ROOT|REPO_ROOT)\s*/|Path\(\s*['\"]/(?:"
    + machine_roots
    + r")(?:/|['\"])"
)

errors: list[str] = []


def inspect_payload(path: str, payload: bytes, *, origin: str) -> None:
    if b"\0" in payload:
        errors.append(f"NUL/binary payload in {origin}: {path}")
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"non-UTF-8 payload in {origin}: {path}")
        return
    checks = (
        (posix_machine_path, "POSIX machine path"),
        (windows_machine_path, "Windows machine path"),
        (unc_machine_path, "UNC machine path"),
        (file_uri, "file URI"),
        (machine_hostname, "machine hostname"),
        (credential_value, "credential assignment"),
        (credential_token, "credential token"),
        (absolute_default, "absolute CLI default"),
    )
    for pattern, label in checks:
        if pattern.search(text):
            errors.append(f"{label} in {origin}: {path}")


for path_text in paths:
    pure = PurePosixPath(path_text)
    path = Path(path_text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        errors.append(f"unsafe repository path: {path_text}")
    if pure.parts and pure.parts[0] in banned_top_level:
        errors.append(f"banned top-level path: {path_text}")
    if any(part in banned_parts for part in pure.parts):
        errors.append(f"cache/generated path: {path_text}")
    if path.name.lower() in banned_names:
        errors.append(f"credential-like filename: {path_text}")
    if path.suffix.lower() in banned_suffixes:
        errors.append(f"banned data/model/runtime artifact: {path_text}")

    if path_text in tracked:
        mode = index_modes.get(path_text)
        if mode not in {"100644", "100755"}:
            errors.append(f"unsafe Git mode {mode}: {path_text}")
        else:
            inspect_payload(
                path_text,
                git_bytes("show", f":{path_text}"),
                origin="Git index",
            )

    if path.exists() or path.is_symlink():
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode):
            errors.append(f"non-regular worktree candidate: {path_text}")
        elif metadata.st_nlink != 1:
            errors.append(f"hard-linked worktree candidate: {path_text}")
        else:
            inspect_payload(path_text, path.read_bytes(), origin="worktree")

if errors:
    print("public release check failed:")
    for error in sorted(set(errors)):
        print(f"- {error}")
    raise SystemExit(1)

print(f"public release check passed: {len(paths)} files")
PY

git diff --check
git diff --cached --check
