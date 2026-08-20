#!/usr/bin/env python3
"""Render the reviewed 66-rig sample from the full KTJD-17 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    default_full_build_config,
    reviewed_representative_clip_ids,
    validate_visual_gate,
    verify_full_generation,
)
from src.data.ktjd17.private_release import (  # noqa: E402
    load_trusted_release,
    resolve_release_generation,
    resolve_repository_path,
)
from src.data.ktjd17.visual_qa import render_prototype_visual_qa  # noqa: E402


def main() -> int:
    defaults = default_full_build_config(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/ktjd17_truebones")
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
        "--output-root",
        type=Path,
        default=Path("outputs/ktjd17_truebones_visual"),
    )
    parser.add_argument(
        "--trust-record", type=Path, default=Path("release/truebones_v1.json")
    )
    parser.add_argument(
        "--source-backed",
        action="store_true",
        help="use the proprietary forward-audit representative selection",
    )
    parser.add_argument("--max-gif-frames", type=int, default=24)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    if args.max_gif_frames < 2:
        parser.error("--max-gif-frames must be at least 2")
    dataset_input = resolve_repository_path(
        ROOT, args.dataset_root, argument_name="--dataset-root"
    )
    output_root = resolve_repository_path(
        ROOT, args.output_root, argument_name="--output-root"
    )
    if args.source_backed:
        dataset_root = dataset_input
    else:
        trust_path = resolve_repository_path(
            ROOT, args.trust_record, argument_name="--trust-record"
        )
        trust = load_trusted_release(trust_path)
        dataset_root = resolve_release_generation(
            dataset_input, trusted_release=trust
        )
    verify_full_generation(dataset_root)
    if args.source_backed:
        forward_audit_root = resolve_repository_path(
            ROOT, args.forward_audit_root, argument_name="--forward-audit-root"
        )
        validate_visual_gate(
            gate_path=defaults.visual_gate_path,
            visual_root=defaults.visual_root,
            forward_audit_root=forward_audit_root,
        )
        clip_ids = reviewed_representative_clip_ids(forward_audit_root)
    else:
        rows = [
            json.loads(line)
            for line in (dataset_root / "manifests/clips.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        representatives: dict[str, str] = {}
        for row in sorted(rows, key=lambda value: str(value["clip_id"])):
            representatives.setdefault(str(row["rig_id"]), str(row["clip_id"]))
        clip_ids = sorted(representatives.values())
    result = render_prototype_visual_qa(
        prototype_root=dataset_root,
        calibration_root=None,
        output_root=output_root,
        clip_ids=clip_ids,
        max_gif_frames=args.max_gif_frames,
        update_link=not args.no_update_link,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
