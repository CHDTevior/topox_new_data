from __future__ import annotations

import json
import os
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.ktjd17.loader import load_motion_npz
from src.data.ktjd17.truebones_full_build import (
    PARENT_PROTOTYPE_CANDIDATES_SHA256,
    TruebonesFullBuildError,
    _representative_regression,
    _accepted_identity_sha256,
    _file_manifest,
    _sha256_file,
    _snapshot_regular_file,
    _source_scope_identity_sha256,
    _verify_selection_identity,
    _verify_payload_reference_closure,
    default_full_build_config,
    reviewed_representative_clip_ids,
    summarize_strata,
    validate_visual_gate,
    verify_full_generation,
)


ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FULL_CONFIG = default_full_build_config(ROOT)
HAS_PRIVATE_FULL_BUILD_FIXTURES = (
    _DEFAULT_FULL_CONFIG.visual_gate_path.is_file()
    and (_DEFAULT_FULL_CONFIG.manifest_root / "prototype_candidates.json").is_file()
    and (
        _DEFAULT_FULL_CONFIG.forward_audit_root / "manifests/clips.jsonl"
    ).is_file()
)


class FullBuildAuthorityTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_PRIVATE_FULL_BUILD_FIXTURES,
        "requires private immutable forward-audit and visual-gate fixtures",
    )
    def test_live_visual_gate_binds_reviewed_generations(self) -> None:
        config = default_full_build_config(ROOT)
        gate = validate_visual_gate(
            gate_path=config.visual_gate_path,
            visual_root=config.visual_root,
            forward_audit_root=config.forward_audit_root,
        )
        self.assertEqual(gate["verdict"], "pass")
        self.assertTrue(gate["authorization"]["full_source_safe_conversion"])
        self.assertEqual(gate["visual_generation"]["rig_count"], 66)

    @unittest.skipUnless(
        HAS_PRIVATE_FULL_BUILD_FIXTURES,
        "requires the private immutable parent manifest",
    )
    def test_parent_prototype_candidate_pin_is_live(self) -> None:
        config = default_full_build_config(ROOT)
        self.assertEqual(
            _sha256_file(config.manifest_root / "prototype_candidates.json"),
            PARENT_PROTOTYPE_CANDIDATES_SHA256,
        )

    @unittest.skipUnless(
        HAS_PRIVATE_FULL_BUILD_FIXTURES,
        "requires the private immutable forward-audit generation",
    )
    def test_reviewed_representative_scope_is_exactly_the_frozen_66(self) -> None:
        config = default_full_build_config(ROOT)
        clip_ids = reviewed_representative_clip_ids(config.forward_audit_root)
        self.assertEqual(len(clip_ids), 66)
        self.assertEqual(len(set(clip_ids)), 66)

    def test_stable_source_snapshot_checks_size_mtime_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            source.write_bytes(b"HIERARCHY\nMOTION\n")
            stat_result = source.stat()
            snapshot = _snapshot_regular_file(
                source,
                expected_size=stat_result.st_size,
                expected_mtime_ns=stat_result.st_mtime_ns,
            )
            self.assertEqual(snapshot["size_bytes"], stat_result.st_size)
            self.assertEqual(snapshot["sha256"], _sha256_file(source))
            with self.assertRaises(TruebonesFullBuildError):
                _snapshot_regular_file(source, expected_size=stat_result.st_size + 1)
            link = root / "source-link.bvh"
            link.symlink_to(source)
            with self.assertRaises(TruebonesFullBuildError):
                _snapshot_regular_file(link)

    @unittest.skipUnless(
        HAS_PRIVATE_FULL_BUILD_FIXTURES,
        "requires private immutable forward-audit and visual-gate fixtures",
    )
    def test_visual_gate_copy_cannot_self_authorize_after_edit(self) -> None:
        config = default_full_build_config(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            gate_path = Path(temporary) / "gate.json"
            gate = json.loads(config.visual_gate_path.read_text(encoding="utf-8"))
            gate["independent_review"]["model_reasoning_effort"] = "low"
            gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
            with self.assertRaises(TruebonesFullBuildError):
                validate_visual_gate(
                    gate_path=gate_path,
                    visual_root=config.visual_root,
                    forward_audit_root=config.forward_audit_root,
                )


class FullBuildRegressionTests(unittest.TestCase):
    def test_file_manifest_rejects_special_files_and_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.mkfifo(root / "extra.fifo")
            with self.assertRaisesRegex(TruebonesFullBuildError, "special file"):
                _file_manifest(root, forbid_hardlinks=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            os.link(payload, root / "alias.bin")
            with self.assertRaisesRegex(TruebonesFullBuildError, "hard-linked"):
                _file_manifest(root, forbid_hardlinks=True)

    def test_selection_verifier_rejects_self_consistent_foreign_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            (root / "qa").mkdir()
            selected = [
                {"clip_id": "ForeignCorp___Clip_0001", "rig_id": "ForeignCorp", "split": "train"}
            ]
            counts = {
                split: {"selected": int(split == "train")}
                for split in ("train", "val", "held_representative", "held_stress")
            }
            authority = {"selection_kind": "full_source_safe_conversion"}
            core = {
                "selection_authority": authority,
                "selection_counts": counts,
                "selected": selected,
            }
            selection_sha = hashlib.sha256(
                json.dumps(
                    core,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            payload = {
                **core,
                "selection_sha256": selection_sha,
                "selected_count": 1,
            }
            for name in ("full_selection.json", "prototype_selection.json"):
                (root / "manifests" / name).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            (root / "qa/encoder_summary.json").write_text(
                json.dumps(
                    {
                        "selection_authority": authority,
                        "selection_counts": counts,
                        "selection_sha256": selection_sha,
                    }
                ),
                encoding="utf-8",
            )
            generation = {"selection_sha256": selection_sha}
            with self.assertRaisesRegex(TruebonesFullBuildError, "frozen pin"):
                _verify_selection_identity(root, generation, selected)

    def test_identity_digests_change_on_foreign_substitution(self) -> None:
        accepted = [{"clip_id": "A", "rig_id": "Rig", "split": "train"}]
        foreign = [{"clip_id": "B", "rig_id": "Rig", "split": "train"}]
        self.assertNotEqual(
            _accepted_identity_sha256(accepted),
            _accepted_identity_sha256(foreign),
        )
        self.assertNotEqual(
            _source_scope_identity_sha256(accepted, [], []),
            _source_scope_identity_sha256(foreign, [], []),
        )

    def test_payload_reference_closure_rejects_orphan_motion(self) -> None:
        observed = {
            "motions/accepted.npz": {},
            "motions/orphan.npz": {},
            "skeletons/Rig.npz": {},
        }
        with self.assertRaisesRegex(TruebonesFullBuildError, "orphan.npz"):
            _verify_payload_reference_closure(
                observed,
                referenced_motions={"motions/accepted.npz"},
                referenced_skeletons={"skeletons/Rig.npz"},
            )

    def test_verifier_requires_complete_generation_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "incomplete-generation"
            root.mkdir()
            (root / "generation.json").write_text(
                json.dumps(
                    {
                        "generation_id": root.name,
                        "status": "conversion_incomplete",
                        "conversion_complete": False,
                        "full_conversion_authorized": False,
                        "files": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TruebonesFullBuildError, "full conversion is incomplete"
            ):
                verify_full_generation(root)

    @unittest.skipUnless(
        HAS_PRIVATE_FULL_BUILD_FIXTURES,
        "requires the private immutable representative motion fixtures",
    )
    def test_reviewed_representative_requires_exact_arrays(self) -> None:
        config = default_full_build_config(ROOT)
        manifest_path = config.forward_audit_root / "manifests/clips.jsonl"
        first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        path = config.forward_audit_root / first["motion_relpath"]
        payload = load_motion_npz(path, expected_fps_target=30.0)
        encoded = SimpleNamespace(
            motion_float32=payload["motion"].copy(),
            heading_valid=payload["heading_valid"].copy(),
            origin_xz=payload["origin_xz"].copy(),
            clip_id=payload["clip_id"],
            rig_id=payload["rig_id"],
            fps_target=float(payload["fps_target"]),
        )
        _representative_regression(encoded, audit_motion_path=path)
        encoded.motion_float32[0, 0, 0] += np.float32(1.0)
        with self.assertRaises(TruebonesFullBuildError):
            _representative_regression(encoded, audit_motion_path=path)

    def test_stratified_summary_keeps_split_rig_and_bucket(self) -> None:
        records = [
            {
                "status": "pass",
                "split": "train",
                "source_family": "truebones",
                "topology_family": "quadruped",
                "topology_distance_bucket": "train_seen_topology",
                "rig_id": "A",
                "parent_inventory_status": "accept",
                "metrics": {"direct_vs_fk_max_norm": 1e-8},
            },
            {
                "status": "pass",
                "split": "val",
                "source_family": "truebones",
                "topology_family": "winged",
                "topology_distance_bucket": "held_out_topology",
                "rig_id": "B",
                "parent_inventory_status": "review",
                "metrics": {"direct_vs_fk_max_norm": 2e-8},
            },
        ]
        summary = summarize_strata(records)
        self.assertEqual(summary["all"]["count"], 2)
        self.assertEqual(summary["split:train"]["pass"], 1)
        self.assertEqual(summary["rig_id:B"]["count"], 1)
        self.assertEqual(
            summary["all"]["metrics"]["direct_vs_fk_max_norm"]["max"],
            2e-8,
        )


if __name__ == "__main__":
    unittest.main()
