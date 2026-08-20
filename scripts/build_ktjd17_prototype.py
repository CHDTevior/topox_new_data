#!/usr/bin/env python3
"""Build the immutable six-family KTJD-17 candidate prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.codec import SmootherConfig  # noqa: E402
from src.data.ktjd17.encoder import EncoderConfig  # noqa: E402
from src.data.ktjd17.prototype import (  # noqa: E402
    PrototypeConfig,
    run_prototype_build,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(
            "dataset/.ktjd17_manifest_generations/"
            "20260819T145535975831Z-ed48b3fd2745"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--active-cond",
        type=Path,
        default=Path("data/current_btjd/cond.npy"),
    )
    parser.add_argument(
        "--legacy-truebones-cond",
        type=Path,
        default=Path("data/legacy_truebones_btjd/cond.npy"),
    )
    parser.add_argument("--fps-target", type=float, default=30.0)
    parser.add_argument("--smoother-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--contact-tau-h", type=float, default=0.05)
    parser.add_argument("--contact-tau-v", type=float, default=0.25)
    parser.add_argument("--heading-eps-h", type=float, default=0.05)
    args = parser.parse_args()
    result = run_prototype_build(
        PrototypeConfig(
            manifest_root=args.manifest_root,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            active_cond_path=args.active_cond,
            legacy_truebones_cond_path=args.legacy_truebones_cond,
            encoder=EncoderConfig(
                fps_target=args.fps_target,
                smoother=SmootherConfig(cutoff_hz=args.smoother_cutoff_hz),
                contact_tau_h=args.contact_tau_h,
                contact_tau_v=args.contact_tau_v,
                heading_eps_h=args.heading_eps_h,
                calibration_status="candidate_unfrozen",
            ),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
