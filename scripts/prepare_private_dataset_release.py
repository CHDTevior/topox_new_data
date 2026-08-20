#!/usr/bin/env python3
"""Prepare a host-sanitized KTJD-17 generation for private distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.private_release import (  # noqa: E402
    PrivateReleaseError,
    prepare_private_distribution,
    resolve_repository_path,
)
from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    TruebonesFullBuildError,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-generation",
        type=Path,
        default=Path("dataset/ktjd17_truebones"),
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("dataset/private_release"),
    )
    parser.add_argument(
        "--postbuild-gate",
        type=Path,
        required=True,
        help="repository-relative signed fixed-QA and visual-review gate",
    )
    parser.add_argument(
        "--fixed-qa-report",
        type=Path,
        required=True,
        help="repository-relative 986-clip fixed-QA report",
    )
    parser.add_argument(
        "--visual-generation",
        type=Path,
        required=True,
        help="repository-relative 66-rig visual generation",
    )
    parser.add_argument(
        "--visual-equivalence-report",
        type=Path,
        required=True,
        help="repository-relative postbuild visual equivalence report",
    )
    parser.add_argument(
        "--review-contact-sheets",
        type=Path,
        required=True,
        help="repository-relative directory containing the 11 reviewed sheets",
    )
    args = parser.parse_args()
    try:
        source = resolve_repository_path(
            ROOT, args.source_generation, argument_name="--source-generation"
        )
        output = resolve_repository_path(
            ROOT, args.output_parent, argument_name="--output-parent"
        )
        gate = resolve_repository_path(
            ROOT, args.postbuild_gate, argument_name="--postbuild-gate"
        )
        fixed_qa = resolve_repository_path(
            ROOT, args.fixed_qa_report, argument_name="--fixed-qa-report"
        )
        visual = resolve_repository_path(
            ROOT, args.visual_generation, argument_name="--visual-generation"
        )
        equivalence = resolve_repository_path(
            ROOT,
            args.visual_equivalence_report,
            argument_name="--visual-equivalence-report",
        )
        contact_sheets = resolve_repository_path(
            ROOT,
            args.review_contact_sheets,
            argument_name="--review-contact-sheets",
        )
        result = prepare_private_distribution(
            source,
            output,
            postbuild_gate=gate,
            fixed_qa_report=fixed_qa,
            visual_generation=visual,
            visual_equivalence_report=equivalence,
            review_contact_sheets=contact_sheets,
        )
    except (PrivateReleaseError, TruebonesFullBuildError) as exc:
        parser.error(str(exc))
    display = dict(result)
    display["generation_root"] = Path(result["generation_root"]).relative_to(ROOT).as_posix()
    display["release_root"] = Path(result["release_root"]).relative_to(ROOT).as_posix()
    display["release_pointer"] = Path(result["release_pointer"]).relative_to(ROOT).as_posix()
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
