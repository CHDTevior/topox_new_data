#!/usr/bin/env python3
"""Download an authorized private KTJD-17 dataset to a relative directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.private_release import (  # noqa: E402
    PrivateReleaseError,
    resolve_release_generation,
    resolve_repository_path,
)
from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    TruebonesFullBuildError,
    verify_full_generation,
)


DEFAULT_REPO_ID = "Tevior/KTJD17-Truebones-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the private KTJD-17 Truebones release using credentials "
            "from the Hugging Face credential store."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--local-dir", type=Path, default=Path("data/ktjd17_truebones")
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    try:
        destination = resolve_repository_path(
            ROOT, args.local_dir, argument_name="--local-dir"
        )
        local_path = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=destination,
                token=True,
            )
        ).resolve()
        if local_path != destination:
            raise PrivateReleaseError(
                "Hugging Face returned an unexpected snapshot directory"
            )
        generation_root = resolve_release_generation(local_path, require_pointer=True)
        generation = verify_full_generation(generation_root, require_complete=True)
    except (PrivateReleaseError, TruebonesFullBuildError) as exc:
        raise SystemExit(f"download verification failed: {exc}") from exc
    relative_destination = destination.relative_to(ROOT).as_posix()
    relative_generation = generation_root.relative_to(ROOT).as_posix()
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "local_dir": relative_destination,
                "dataset_root": relative_generation,
                "generation_id": generation["generation_id"],
                "status": "downloaded_and_verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
