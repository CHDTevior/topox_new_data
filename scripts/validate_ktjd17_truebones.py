#!/usr/bin/env python3
"""Run standalone distribution QA or explicit source-backed KTJD-17 QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.fixed_qa import validate_prototype  # noqa: E402
from src.data.ktjd17.private_release import (  # noqa: E402
    load_published_truebones_release,
    resolve_repository_path,
    validate_private_distribution,
)
from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    verify_full_generation,
)


def _write_json_atomic(path: Path, value: object) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        descriptor = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset/ktjd17_truebones")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ktjd17_truebones_qa.json"),
    )
    parser.add_argument(
        "--source-backed",
        action="store_true",
        help="also require the proprietary BVH and parent-manifest build workspace",
    )
    args = parser.parse_args()
    dataset_input = resolve_repository_path(
        ROOT,
        args.dataset_root,
        argument_name="--dataset-root",
        preserve_leaf=True,
    )
    output = resolve_repository_path(ROOT, args.output, argument_name="--output")
    if args.source_backed:
        dataset_root = dataset_input
        generation = verify_full_generation(dataset_root, require_complete=True)
        report = validate_prototype(dataset_root)
        report["artifact_kind"] = "full_truebones_source_backed_dataset"
        report["full_build_version"] = generation["full_build_version"]
        report["full_generation_json_sha256"] = hashlib.sha256(
            (dataset_root / "generation.json").read_bytes()
        ).hexdigest()
        conversion_complete = generation.get("conversion_complete") is True
        report["full_conversion_gate"] = {
            "status": "pass" if conversion_complete else "fail",
            "conversion_complete": conversion_complete,
            "full_conversion_authorized": generation.get(
                "full_conversion_authorized"
            )
            is True,
            "required_source_safe_clip_count": 986,
            "observed_clip_count": report["clip_count"],
        }
        if not conversion_complete:
            report["status"] = "fail"
    else:
        trust = load_published_truebones_release(
            ROOT / "release/truebones_v1.json", require_hf_revision=True
        )
        report = validate_private_distribution(
            dataset_input, trusted_release=trust
        )
        conversion_complete = True
    _write_json_atomic(output, report)
    summary = {
        key: report[key]
        for key in (
            "status",
            "generation_id",
            "clip_count",
            "pass_count",
            "fail_count",
            "skeleton_count",
            "J_phys_max",
            "T_max_observed",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report={output.relative_to(ROOT).as_posix()}")
    return 0 if report["status"] == "pass" and conversion_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
