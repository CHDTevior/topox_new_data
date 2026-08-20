#!/usr/bin/env python3
"""Download an authorized private KTJD-17 dataset to a relative directory."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
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


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically install one directory without replacing a concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PrivateReleaseError(
            "this platform lacks atomic no-clobber directory publication"
        )
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
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise PrivateReleaseError("download destination appeared during verification")
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
            tempfile.mkdtemp(prefix=f".{destination.name}.download-", dir=destination.parent)
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
        if staging is not None and staging.exists():
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
