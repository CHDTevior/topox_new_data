#!/usr/bin/env python3
"""Download an authorized private KTJD-17 dataset to a relative directory."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.private_release import (  # noqa: E402
    PrivateReleaseError,
    load_published_truebones_release,
    resolve_repository_path,
    validate_private_distribution,
)


DEFAULT_TRUST_RECORD = Path("release/truebones_v1.json")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FALLBACK_PAYLOAD_MARKER = ".payload-"


def _rename_directory_noreplace_errno(source: Path, destination: Path) -> int:
    """Return zero on renameat2 success, otherwise the platform errno."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return 0
    return ctypes.get_errno()


def _directory_identity(descriptor: int) -> tuple[int, int]:
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode):
        raise PrivateReleaseError("publication descriptor is not a directory")
    return current.st_dev, current.st_ino


def _publish_with_relative_alias(source: Path, destination: Path) -> None:
    """Atomically publish a relative alias when no-clobber rename is unavailable."""
    if source.parent != destination.parent:
        raise PrivateReleaseError(
            "fallback publication requires staging beside the destination"
        )
    expected_prefix = f".{destination.name}{_FALLBACK_PAYLOAD_MARKER}"
    if not source.name.startswith(expected_prefix):
        raise PrivateReleaseError("fallback payload name is outside the frozen scheme")
    parent_descriptor = os.open(destination.parent, _DIRECTORY_OPEN_FLAGS)
    try:
        source_descriptor = os.open(
            source.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
        )
        try:
            source_identity = _directory_identity(source_descriptor)
            os.fsync(source_descriptor)
            try:
                os.symlink(
                    source.name,
                    destination.name,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError as exc:
                raise PrivateReleaseError(
                    "download destination appeared during verification"
                ) from exc
            destination_status = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISLNK(destination_status.st_mode)
                or os.readlink(destination.name, dir_fd=parent_descriptor)
                != source.name
            ):
                raise PrivateReleaseError("download destination alias changed")
            rebound = os.stat(
                source.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(rebound.st_mode) or (
                rebound.st_dev,
                rebound.st_ino,
            ) != source_identity:
                raise PrivateReleaseError("download payload directory changed")
            os.fsync(parent_descriptor)
        finally:
            os.close(source_descriptor)
    finally:
        os.close(parent_descriptor)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Install one directory without replacing a concurrent target."""
    error_number = _rename_directory_noreplace_errno(source, destination)
    if error_number == 0:
        return
    if error_number == errno.EEXIST:
        raise PrivateReleaseError("download destination appeared during verification")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        _publish_with_relative_alias(source, destination)
        return
    raise PrivateReleaseError(
        f"atomic no-clobber publication failed: {os.strerror(error_number)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the private KTJD-17 Truebones release using credentials "
            "from the Hugging Face credential store."
        )
    )
    parser.add_argument(
        "--local-dir", type=Path, default=Path("data/ktjd17_truebones")
    )
    args = parser.parse_args()

    staging: Path | None = None
    publication_started = False
    try:
        trust_path = ROOT / DEFAULT_TRUST_RECORD
        trust = load_published_truebones_release(
            trust_path, require_hf_revision=True
        )
        revision = str(trust["hf_revision"])
        destination = resolve_repository_path(
            ROOT,
            args.local_dir,
            argument_name="--local-dir",
            required_top_level="data",
            must_not_exist=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}{_FALLBACK_PAYLOAD_MARKER}",
                dir=destination.parent,
            )
        )
        local_path = Path(
            snapshot_download(
                repo_id=str(trust["repo_id"]),
                repo_type=str(trust["repo_type"]),
                revision=revision,
                local_dir=staging,
                token=True,
                allow_patterns=[
                    "RELEASE.json",
                    f"{trust['generation_id']}/*",
                    f"{trust['generation_id']}/**",
                ],
            )
        ).resolve()
        if local_path != staging.resolve():
            raise PrivateReleaseError(
                "Hugging Face returned an unexpected snapshot directory"
            )
        transport_cache = staging / ".cache"
        if transport_cache.exists() or transport_cache.is_symlink():
            if transport_cache.is_symlink() or not transport_cache.is_dir():
                raise PrivateReleaseError("download transport cache has an unsafe type")
            shutil.rmtree(transport_cache)
        qa = validate_private_distribution(staging, trusted_release=trust)
        if qa.get("status") != "pass":
            raise PrivateReleaseError("downloaded distribution QA did not pass")
        publication_started = True
        _publish_directory_noreplace(staging, destination)
        staging = None
        generation_root = destination / str(trust["generation_id"])
        descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception as exc:  # transport and verification must share one rollback path
        raise SystemExit(f"download verification failed: {exc}") from exc
    finally:
        if (
            staging is not None
            and not publication_started
            and staging.exists()
        ):
            shutil.rmtree(staging)
    relative_destination = destination.relative_to(ROOT).as_posix()
    relative_generation = generation_root.relative_to(ROOT).as_posix()
    print(
        json.dumps(
            {
                "repo_id": trust["repo_id"],
                "revision": revision,
                "local_dir": relative_destination,
                "dataset_root": relative_generation,
                "generation_id": qa["generation_id"],
                "qa_pass_count": qa["pass_count"],
                "status": "downloaded_and_verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
