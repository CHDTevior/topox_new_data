#!/usr/bin/env python3
"""Build the frozen source-safe Truebones KTJD-17 dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    default_full_build_config,
    run_truebones_full_build,
)


def main() -> int:
    defaults = default_full_build_config(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(
            "dataset/.ktjd17_manifest_generations/"
            "20260819T145535975831Z-ed48b3fd2745"
        ),
    )
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=Path(
            "dataset/.ktjd17_freeze_generations/"
            "20260819T192429040697Z-fe820492caaa"
        ),
    )
    parser.add_argument(
        "--forward-audit-root",
        type=Path,
        default=Path(
            "dataset/.ktjd17_truebones_forward_audit_generations/"
            "20260819T203306371942Z-8541b68c8480"
        ),
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path(
            "dataset/ktjd17_truebones_forward_visual/"
            ".ktjd17_visual_qa_generations/"
            "20260819T203413394509Z-c8a431c08118"
        ),
    )
    parser.add_argument(
        "--visual-gate",
        type=Path,
        default=Path("dataset/ktjd17_truebones_forward_visual_gate.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument("--active-cond", type=Path, default=Path("data/current_btjd/cond.npy"))
    parser.add_argument(
        "--legacy-cond", type=Path, default=Path("data/legacy_truebones_btjd/cond.npy")
    )
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    config = dataclasses.replace(
        defaults,
        manifest_root=args.manifest_root,
        freeze_root=args.freeze_root,
        forward_audit_root=args.forward_audit_root,
        visual_root=args.visual_root,
        visual_gate_path=args.visual_gate,
        output_root=args.output_root,
        active_cond_path=args.active_cond,
        legacy_cond_path=args.legacy_cond,
        update_link=not args.no_update_link,
    )
    result = run_truebones_full_build(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["conversion_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
