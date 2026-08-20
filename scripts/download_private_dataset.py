#!/usr/bin/env python3
"""Download an authorized private KTJD-17 dataset to a relative directory."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    load_trusted_release,
    resolve_release_generation,
    resolve_repository_path,
    validate_private_distribution,
)
from src.data.ktjd17.truebones_full_build import verify_full_generation  # noqa: E402


DEFAULT_TRUST_RECORD = Path("release/truebones_v1.json")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the private KTJD-17 Truebones release using credentials "
            "from the Hugging Face credential store."
        )
    )
    parser.add_argument(
        "--trust-record",
        type=Path,
        default=DEFAULT_TRUST_RECORD,
        help="repository-relative public trust record",
    )
    parser.add_argument(
        "--local-dir", type=Path, default=Path("data/ktjd17_truebones")
    )
    parser.add_argument(
        "--revision",
        help="immutable 40-hex Hugging Face commit; defaults to the trust record",
    )
    args = parser.parse_args()

    staging: Path | None = None
    try:
        trust_path = resolve_repository_path(
            ROOT, args.trust_record, argument_name="--trust-record"
        )
        trust = load_trusted_release(trust_path)
        revision = args.revision or trust.get("hf_revision")
        if not isinstance(revision, str) or _IMMUTABLE_REVISION.fullmatch(revision) is None:
            raise PrivateReleaseError(
                "--revision must be an immutable 40-hex commit until the trust record pins one"
            )
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
        generation_root = resolve_release_generation(
            staging, trusted_release=trust
        )
        generation = verify_full_generation(generation_root, require_complete=True)
        qa = validate_private_distribution(staging, trusted_release=trust)
        if qa.get("status") != "pass":
            raise PrivateReleaseError("downloaded distribution QA did not pass")
        os.replace(staging, destination)
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
                "generation_id": generation["generation_id"],
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
