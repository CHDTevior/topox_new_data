from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import scripts.download_private_dataset as downloader
import src.data.ktjd17.private_release as private_release
from src.data.ktjd17.private_release import (
    PRIVATE_RELEASE_VERSION,
    TRUST_RECORD_VERSION,
    PrivateReleaseError,
    _assert_no_absolute_machine_paths,
    _file_manifest,
    _load_postbuild_release_gate,
    _sanitize_value,
    _sanitized_path_label,
    _write_release_pointer,
    load_published_truebones_release,
    load_trusted_release,
    resolve_release_generation,
    resolve_repository_path,
    validate_private_distribution,
)
from src.data.ktjd17.truebones_full_build import (
    EXPECTED_ACCEPTED_IDENTITY_SHA256,
    EXPECTED_SOURCE_GENERATION_ID,
    EXPECTED_SOURCE_GENERATION_JSON_SHA256,
    EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
    TruebonesFullBuildError,
    _file_manifest as _full_file_manifest,
)


REVISION = "1" * 40


def _write_minimal_motion_npz(
    path: Path, *, motion: np.ndarray | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        motion=(
            np.zeros((1, 1, 17), dtype=np.float32)
            if motion is None
            else motion
        ),
        heading_valid=np.asarray([True], dtype=np.bool_),
        clip_id=np.asarray("clip"),
        rig_id=np.asarray("rig"),
        fps_target=np.asarray(30.0, dtype=np.float64),
        origin_xz=np.zeros(2, dtype=np.float64),
    )


def _trust_record(
    generation_id: str,
    generation_sha: str,
    pointer_sha: str,
    *,
    revision: str | None = REVISION,
) -> dict[str, object]:
    return {
        "trust_record_version": TRUST_RECORD_VERSION,
        "release_version": PRIVATE_RELEASE_VERSION,
        "private_required": True,
        "repo_id": "owner/private-dataset",
        "repo_type": "dataset",
        "generation_id": generation_id,
        "generation_json_sha256": generation_sha,
        "release_pointer_sha256": pointer_sha,
        "accepted_identity_sha256": EXPECTED_ACCEPTED_IDENTITY_SHA256,
        "source_scope_identity_sha256": EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
        "hf_revision": revision,
    }


def _make_pointer_snapshot(root: Path) -> tuple[Path, dict[str, object]]:
    generation = root / "generation-v1"
    generation.mkdir()
    payload = b'{"generation_id":"generation-v1"}\n'
    (generation / "generation.json").write_bytes(payload)
    pointer = _write_release_pointer(root, generation)
    trust = _trust_record(
        generation.name,
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(pointer.read_bytes()).hexdigest(),
    )
    return generation, trust


class RepositoryPathTests(unittest.TestCase):
    def test_rejects_parent_absolute_windows_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "escape").symlink_to(outside, target_is_directory=True)
            for value in (
                "../outside",
                "/" + "tmp/outside",
                "C:" + chr(92) + "outside",
                "escape/data",
            ):
                with self.subTest(value=value), self.assertRaises(PrivateReleaseError):
                    resolve_repository_path(root, value, argument_name="--local-dir")

    def test_download_destination_must_be_fresh_data_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/existing").mkdir()
            (root / "data/dangling").symlink_to("owner-chosen")
            for value in (
                ".",
                ".git",
                "scripts",
                "README.md",
                "data",
                "data/existing",
                "data/dangling",
            ):
                with self.subTest(value=value), self.assertRaises(PrivateReleaseError):
                    resolve_repository_path(
                        root,
                        value,
                        argument_name="--local-dir",
                        required_top_level="data",
                        must_not_exist=True,
                    )
            expected = (root / "data/new-snapshot").resolve()
            self.assertEqual(
                resolve_repository_path(
                    root,
                    "data/new-snapshot",
                    argument_name="--local-dir",
                    required_top_level="data",
                    must_not_exist=True,
                ),
                expected,
            )

    def test_preserved_leaf_reaches_the_bounded_alias_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            owner = data / "owner"
            owner.mkdir(parents=True)
            _generation, trust = _make_pointer_snapshot(owner)
            alias = data / "snapshot"
            alias.symlink_to(owner.name)
            preserved = resolve_repository_path(
                root,
                "data/snapshot",
                argument_name="--dataset-root",
                preserve_leaf=True,
            )
            self.assertTrue(preserved.is_symlink())
            with self.assertRaisesRegex(PrivateReleaseError, "frozen scheme"):
                resolve_release_generation(preserved, trusted_release=trust)


class SanitizerTests(unittest.TestCase):
    def test_paths_hosts_and_traversal_are_sanitized(self) -> None:
        poison = {
            "embedded": "source was /" + "home/alice/private/file.bvh",
            "uri": "file" + ":///tmp/private.bin",
            "windows": "C:"
            + chr(92)
            + "Users"
            + chr(92)
            + "alice"
            + chr(92)
            + "private.bvh",
            "unc": chr(92) * 2 + "server" + chr(92) + "share" + chr(92) + "private.bvh",
            "host": "login77" + "." + "cluster" + "." + "local",
            "relative": "../../escape/file.bvh",
        }
        sanitized = _sanitize_value(poison, {})
        self.assertNotEqual(sanitized, poison)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sanitized.json").write_text(json.dumps(sanitized), encoding="utf-8")
            _assert_no_absolute_machine_paths(root)

    def test_truebones_label_never_preserves_parent_traversal(self) -> None:
        safe = _sanitized_path_label(
            "/" + "machine/data/Truebone_Z-OO/Dragon/Fly.bvh"
        )
        malicious = _sanitized_path_label(
            "/" + "machine/data/Truebone_Z-OO/../../escape/file.bvh"
        )
        self.assertEqual(safe, "sources/truebones/Dragon/Fly.bvh")
        self.assertNotIn("..", malicious)
        self.assertTrue(malicious.startswith("sources/redacted/"))

    def test_postscan_rejects_embedded_path_hostname_and_opaque_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "poison.json").write_text(
                json.dumps(
                    {
                        "value": "at /"
                        + "mnt/private on compute7"
                        + "."
                        + "internal"
                        + "."
                        + "example"
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PrivateReleaseError, "host paths"):
                _assert_no_absolute_machine_paths(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "opaque.bin").write_bytes(
                bytes([255]) + b"/" + b"iridisfs/private"
            )
            with self.assertRaisesRegex(PrivateReleaseError, "host paths"):
                _assert_no_absolute_machine_paths(root)

    def test_manifest_rejects_fifo_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root / "extra.fifo")
            with self.assertRaisesRegex(PrivateReleaseError, "special file"):
                _file_manifest(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "payload.bin"
            source.write_bytes(b"payload")
            os.link(source, root / "alias.bin")
            with self.assertRaisesRegex(PrivateReleaseError, "hard-linked"):
                _file_manifest(root)

    def test_manifest_rejects_executable_and_group_writable_files(self) -> None:
        for mode in (0o755, 0o664):
            with self.subTest(mode=oct(mode)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / "payload.json"
                payload.write_text("{}\n", encoding="utf-8")
                payload.chmod(mode)
                with self.assertRaisesRegex(PrivateReleaseError, "unsafe file mode"):
                    _file_manifest(root)

    def test_manifest_rejects_group_writable_directories_in_both_closures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = root / "unsafe"
            unsafe.mkdir()
            unsafe.chmod(0o777)
            with self.assertRaisesRegex(PrivateReleaseError, "unsafe directory mode"):
                _file_manifest(root)
            with self.assertRaisesRegex(
                TruebonesFullBuildError, "unsafe directory mode"
            ):
                _full_file_manifest(root, forbid_hardlinks=True)

    def test_npz_hidden_or_traversing_members_are_rejected(self) -> None:
        for member in (
            "../../escape.npy",
            "/absolute.npy",
            ".hidden.npy",
            "./alias.npy",
            "hidden.bin",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / "motions/motion.npz"
                _write_minimal_motion_npz(payload)
                with zipfile.ZipFile(payload, "a") as archive:
                    archive.writestr(member, b"opaque")
                with self.assertRaisesRegex(PrivateReleaseError, "NPZ member closure"):
                    _assert_no_absolute_machine_paths(root)

    def test_npz_json_strings_are_decoded_before_path_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "motions/motion.npz"
            _write_minimal_motion_npz(
                payload,
                motion=np.asarray(
                    ' {"path":"\\u002fscratch\\u002fprivate"}'
                ).reshape(1, 1, 1),
            )
            with self.assertRaisesRegex(PrivateReleaseError, "host paths"):
                _assert_no_absolute_machine_paths(root)

    def test_npz_structured_dtype_field_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "motions/motion.npz"
            structured = np.zeros(
                (1, 1, 1), dtype=[("/" + "scratch/private/field", "f4")]
            )
            _write_minimal_motion_npz(payload, motion=structured)
            with self.assertRaisesRegex(PrivateReleaseError, "host paths"):
                _assert_no_absolute_machine_paths(root)

    def test_non_utf8_binary_image_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "evidence.jpg"
            Image.new("RGB", (2, 2)).save(
                image_path,
                icc_profile=bytes([255]) + b"/" + b"scratch/private",
            )
            with self.assertRaisesRegex(PrivateReleaseError, "binary image metadata"):
                _assert_no_absolute_machine_paths(root)

    def test_hash_then_archive_precede_generation_semantic_loading(self) -> None:
        events: list[str] = []

        def semantic_stop(*args: object, **kwargs: object) -> None:
            events.append("semantic_loading")
            raise PrivateReleaseError("semantic stop")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            private_release,
            "resolve_release_generation",
            return_value=Path(directory) / "generation",
        ), mock.patch.object(
            private_release,
            "verify_full_generation_file_closure",
            side_effect=lambda *args, **kwargs: events.append("hash_closure"),
        ), mock.patch.object(
            private_release,
            "_preflight_private_release_structure",
            side_effect=lambda *args, **kwargs: events.append("zip_structure"),
        ), mock.patch.object(
            private_release,
            "verify_full_generation",
            side_effect=semantic_stop,
        ) as semantic, mock.patch.object(
            private_release, "_assert_no_absolute_machine_paths"
        ) as content_scan:
            with self.assertRaisesRegex(PrivateReleaseError, "semantic stop"):
                validate_private_distribution(directory, trusted_release={})
            self.assertEqual(
                events, ["hash_closure", "zip_structure", "semantic_loading"]
            )
            semantic.assert_called_once()
            content_scan.assert_not_called()

    def test_evidence_manifest_schema_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.png"
            artifact.write_bytes(b"payload")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": {
                            "artifact.png": {
                                "sha256": hashlib.sha256(b"payload").hexdigest(),
                                "size_bytes": 7,
                            }
                        },
                        "untrusted_extra": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PrivateReleaseError, "schema drifted"):
                private_release._verify_manifest_files(
                    root, manifest, expected_count=1
                )


class ReleasePointerTests(unittest.TestCase):
    def test_resolves_only_externally_pinned_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, trust = _make_pointer_snapshot(root)
            self.assertEqual(
                resolve_release_generation(root, trusted_release=trust),
                generation.resolve(),
            )

    def test_loads_exact_public_trust_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _generation, trust = _make_pointer_snapshot(root)
            path = root / "trust.json"
            path.write_text(json.dumps(trust), encoding="utf-8")
            self.assertEqual(load_trusted_release(path), trust)
            trust["private_required"] = False
            path.write_text(json.dumps(trust), encoding="utf-8")
            with self.assertRaisesRegex(PrivateReleaseError, "values are invalid"):
                load_trusted_release(path)

    def test_published_loader_rejects_a_self_authorized_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged.json"
            path.write_text(
                json.dumps(_trust_record("forged", "0" * 64, "1" * 64)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PrivateReleaseError, "compiled release identity"):
                load_published_truebones_release(path)

    def test_rejects_pointer_escape_and_root_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _generation, trust = _make_pointer_snapshot(root)
            (root / "UNPINNED.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(PrivateReleaseError, "root closure"):
                resolve_release_generation(root, trusted_release=trust)

    def test_rejects_wrong_contract_even_when_external_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _generation, trust = _make_pointer_snapshot(root)
            pointer_path = root / "RELEASE.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["release_version"] = "evil-v99"
            pointer["private_required"] = False
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            trust["release_pointer_sha256"] = hashlib.sha256(
                pointer_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(PrivateReleaseError, "contract fields"):
                resolve_release_generation(root, trusted_release=trust)

    def test_rejects_pointer_generation_symlinks_and_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            generation, trust = _make_pointer_snapshot(root)
            pointer = root / "RELEASE.json"
            external_pointer = Path(outside) / "pointer.json"
            external_pointer.write_bytes(pointer.read_bytes())
            pointer.unlink()
            pointer.symlink_to(external_pointer)
            with self.assertRaisesRegex(PrivateReleaseError, "regular file"):
                resolve_release_generation(root, trusted_release=trust)

            pointer.unlink()
            pointer.write_bytes(external_pointer.read_bytes())
            real_generation = root / "real-generation"
            generation.rename(real_generation)
            generation.symlink_to(real_generation, target_is_directory=True)
            with self.assertRaisesRegex(PrivateReleaseError, "real directory"):
                resolve_release_generation(root, trusted_release=trust)

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            _generation, trust = _make_pointer_snapshot(root)
            pointer = root / "RELEASE.json"
            os.link(pointer, Path(outside) / "pointer-hardlink.json")
            with self.assertRaisesRegex(PrivateReleaseError, "hard-linked"):
                resolve_release_generation(root, trusted_release=trust)

    def test_direct_generation_cannot_bypass_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generation.json").write_text("{}\n", encoding="utf-8")
            trust = _trust_record("generation-v1", "0" * 64, "1" * 64)
            with self.assertRaisesRegex(PrivateReleaseError, "RELEASE.json"):
                resolve_release_generation(root, trusted_release=trust)

    def test_resolves_only_the_downloader_bounded_relative_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            payload = parent / ".snapshot.payload-token"
            payload.mkdir()
            generation, trust = _make_pointer_snapshot(payload)
            alias = parent / "snapshot"
            alias.symlink_to(payload.name)
            self.assertEqual(
                resolve_release_generation(alias, trusted_release=trust),
                generation.resolve(),
            )

        bad_targets = (
            "owner",
            "../.snapshot.payload-token",
            "/" + "tmp/.snapshot.payload-token",
        )
        for target in bad_targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                alias = parent / "snapshot"
                alias.symlink_to(target)
                with self.assertRaisesRegex(PrivateReleaseError, "frozen scheme"):
                    resolve_release_generation(
                        alias,
                        trusted_release=_trust_record(
                            "generation-v1", "0" * 64, "1" * 64
                        ),
                    )

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real_payload = parent / "real-payload"
            real_payload.mkdir()
            payload_alias = parent / ".snapshot.payload-token"
            payload_alias.symlink_to(real_payload.name)
            snapshot_alias = parent / "snapshot"
            snapshot_alias.symlink_to(payload_alias.name)
            with self.assertRaisesRegex(PrivateReleaseError, "real directory"):
                resolve_release_generation(
                    snapshot_alias,
                    trusted_release=_trust_record(
                        "generation-v1", "0" * 64, "1" * 64
                    ),
                )


class PostbuildGateTests(unittest.TestCase):
    def test_public_gate_and_reviewer_are_both_hash_pinned(self) -> None:
        public = Path(__file__).resolve().parents[1] / "release/evidence"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "truebones_postbuild_release_gate.json"
            review_path = root / "truebones_visual_review_gpt56sol.md"
            gate_path.write_bytes(
                (public / "truebones_postbuild_release_gate.json").read_bytes()
            )
            review_path.write_bytes(
                (public / "truebones_visual_review_gpt56sol.md").read_bytes()
            )
            gate = _load_postbuild_release_gate(
                gate_path,
                source_generation_id=EXPECTED_SOURCE_GENERATION_ID,
                source_generation_json_sha256=EXPECTED_SOURCE_GENERATION_JSON_SHA256,
            )
            self.assertEqual(gate["status"], "pass")

            review_path.write_text("# forged pass\n", encoding="utf-8")
            with self.assertRaisesRegex(PrivateReleaseError, "visual review"):
                _load_postbuild_release_gate(
                    gate_path,
                    source_generation_id=EXPECTED_SOURCE_GENERATION_ID,
                    source_generation_json_sha256=EXPECTED_SOURCE_GENERATION_JSON_SHA256,
                )

            review_path.write_bytes(
                (public / "truebones_visual_review_gpt56sol.md").read_bytes()
            )
            gate_path.write_bytes(gate_path.read_bytes() + b" ")
            with self.assertRaisesRegex(PrivateReleaseError, "public pin"):
                _load_postbuild_release_gate(
                    gate_path,
                    source_generation_id=EXPECTED_SOURCE_GENERATION_ID,
                    source_generation_json_sha256=EXPECTED_SOURCE_GENERATION_JSON_SHA256,
                )


class DownloaderTests(unittest.TestCase):
    def _root_and_trust(
        self, directory: str, *, revision: str | None = REVISION
    ) -> tuple[Path, dict[str, object]]:
        root = Path(directory)
        (root / "release").mkdir()
        trust = _trust_record(
            "generation-v1", "0" * 64, "1" * 64, revision=revision
        )
        (root / "release/truebones_v1.json").write_text(
            json.dumps(trust), encoding="utf-8"
        )
        return root, trust

    def test_unpublished_revision_and_removed_overrides_never_reach_hf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _trust = self._root_and_trust(directory, revision=None)
            for extra in (
                [],
                ["--revision", "a" * 40],
                ["--trust-record", "release/alternate.json"],
            ):
                with self.subTest(extra=extra), mock.patch.object(
                    downloader, "ROOT", root
                ), mock.patch.object(
                    downloader, "snapshot_download"
                ) as snapshot, mock.patch.object(
                    sys,
                    "argv",
                    [
                        "download_private_dataset.py",
                        "--local-dir",
                        "data/snapshot",
                        *extra,
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        downloader.main()
                    snapshot.assert_not_called()

    def test_parent_escape_and_preexisting_destination_never_reach_hf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _trust = self._root_and_trust(directory)
            (root / "data/existing").mkdir(parents=True)
            (root / "data/dangling").symlink_to("owner-chosen")
            for local_dir in ("../outside", "data/existing", "data/dangling"):
                with self.subTest(local_dir=local_dir), mock.patch.object(
                    downloader, "ROOT", root
                ), mock.patch.object(
                    downloader,
                    "load_published_truebones_release",
                    return_value=_trust,
                ), mock.patch.object(downloader, "snapshot_download") as snapshot, mock.patch.object(
                    sys,
                    "argv",
                    ["download_private_dataset.py", "--local-dir", local_dir],
                ):
                    with self.assertRaises(SystemExit):
                        downloader.main()
                    snapshot.assert_not_called()

    def test_failed_download_rolls_back_partial_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _trust = self._root_and_trust(directory)

            def partial_download(**kwargs: object) -> str:
                local = Path(str(kwargs["local_dir"]))
                (local / "partial.bin").write_bytes(b"partial")
                raise RuntimeError("transport failed")

            with mock.patch.object(downloader, "ROOT", root), mock.patch.object(
                downloader,
                "load_published_truebones_release",
                return_value=_trust,
            ), mock.patch.object(
                downloader, "snapshot_download", side_effect=partial_download
            ), mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "data/snapshot"],
            ):
                with self.assertRaisesRegex(SystemExit, "transport failed"):
                    downloader.main()
            self.assertFalse((root / "data/snapshot").exists())
            self.assertEqual(list((root / "data").glob(".snapshot.payload-*")), [])

    def test_unsupported_renameat2_uses_an_atomic_relative_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            downloader,
            "_rename_directory_noreplace_errno",
            return_value=errno.EINVAL,
        ):
            root = Path(directory)
            destination = root / "destination"
            source = root / ".destination.payload-token"
            source.mkdir()
            (source / "payload").write_text("ready\n", encoding="utf-8")
            downloader._publish_directory_noreplace(source, destination)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(os.readlink(destination), source.name)
            self.assertTrue(source.is_dir())
            self.assertEqual((destination / "payload").read_text(), "ready\n")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            downloader,
            "_rename_directory_noreplace_errno",
            return_value=errno.EINVAL,
        ):
            root = Path(directory)
            destination = root / "destination"
            source = root / ".destination.payload-token"
            source.mkdir()
            destination.mkdir()
            (destination / "owner").write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(PrivateReleaseError, "destination appeared"):
                downloader._publish_directory_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertEqual((destination / "owner").read_text(), "keep\n")

    def test_fallback_atomic_create_preserves_a_concurrent_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            downloader,
            "_rename_directory_noreplace_errno",
            return_value=errno.EINVAL,
        ):
            root = Path(directory)
            destination = root / "destination"
            source = root / ".destination.payload-token"
            source.mkdir()
            (source / "payload").write_text("ready\n", encoding="utf-8")
            real_symlink = os.symlink

            def create_owner_then_link(*args: object, **kwargs: object) -> None:
                destination.mkdir()
                (destination / "owner").write_text("keep\n", encoding="utf-8")
                real_symlink(*args, **kwargs)

            with mock.patch.object(
                downloader.os, "symlink", side_effect=create_owner_then_link
            ), self.assertRaisesRegex(PrivateReleaseError, "destination appeared"):
                downloader._publish_directory_noreplace(source, destination)
            self.assertEqual((destination / "owner").read_text(), "keep\n")
            self.assertEqual((source / "payload").read_text(), "ready\n")

    def test_fallback_detects_alias_replacement_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            downloader,
            "_rename_directory_noreplace_errno",
            return_value=errno.EINVAL,
        ):
            root = Path(directory)
            destination = root / "destination"
            source = root / ".destination.payload-token"
            owner = root / "owner"
            source.mkdir()
            owner.mkdir()
            (source / "payload").write_text("ready\n", encoding="utf-8")
            (owner / "keep").write_text("keep\n", encoding="utf-8")
            real_readlink = os.readlink
            replaced = False

            def replace_before_readlink(*args: object, **kwargs: object) -> str:
                nonlocal replaced
                if not replaced:
                    destination.unlink()
                    destination.symlink_to(owner.name)
                    replaced = True
                return real_readlink(*args, **kwargs)

            with mock.patch.object(
                downloader.os, "readlink", side_effect=replace_before_readlink
            ), self.assertRaisesRegex(PrivateReleaseError, "alias changed"):
                downloader._publish_directory_noreplace(source, destination)
            self.assertEqual(os.readlink(destination), owner.name)
            self.assertEqual((destination / "keep").read_text(), "keep\n")
            self.assertEqual((source / "payload").read_text(), "ready\n")

    def test_fallback_detects_payload_name_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            downloader,
            "_rename_directory_noreplace_errno",
            return_value=errno.EINVAL,
        ):
            root = Path(directory)
            destination = root / "destination"
            source = root / ".destination.payload-token"
            displaced = root / "displaced"
            source.mkdir()
            (source / "payload").write_text("ready\n", encoding="utf-8")
            real_stat = os.stat
            replaced = False

            def replace_before_payload_rebind(
                *args: object, **kwargs: object
            ) -> os.stat_result:
                nonlocal replaced
                if args and args[0] == source.name and not replaced:
                    source.rename(displaced)
                    source.mkdir()
                    (source / "owner").write_text("keep\n", encoding="utf-8")
                    replaced = True
                return real_stat(*args, **kwargs)

            with mock.patch.object(
                downloader.os, "stat", side_effect=replace_before_payload_rebind
            ), self.assertRaisesRegex(PrivateReleaseError, "payload directory changed"):
                downloader._publish_directory_noreplace(source, destination)
            self.assertEqual((source / "owner").read_text(), "keep\n")
            self.assertEqual((displaced / "payload").read_text(), "ready\n")
            self.assertEqual(os.readlink(destination), source.name)

    def test_success_is_published_only_after_all_verifiers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, trust = self._root_and_trust(directory)

            def completed_download(**kwargs: object) -> str:
                local = Path(str(kwargs["local_dir"]))
                (local / "generation-v1").mkdir()
                return str(local)

            qa = {
                "status": "pass",
                "pass_count": 986,
                "generation_id": "generation-v1",
            }
            with mock.patch.object(downloader, "ROOT", root), mock.patch.object(
                downloader,
                "load_published_truebones_release",
                return_value=trust,
            ), mock.patch.object(
                downloader, "snapshot_download", side_effect=completed_download
            ) as snapshot, mock.patch.object(
                downloader, "validate_private_distribution", return_value=qa
            ), mock.patch.object(
                downloader,
                "_rename_directory_noreplace_errno",
                return_value=errno.EINVAL,
            ), mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "data/snapshot"],
            ):
                self.assertEqual(downloader.main(), 0)
            self.assertTrue((root / "data/snapshot").is_symlink())
            self.assertTrue((root / "data/snapshot/generation-v1").is_dir())
            kwargs = snapshot.call_args.kwargs
            self.assertEqual(kwargs["revision"], trust["hf_revision"])
            self.assertNotIn("README.md", kwargs["allow_patterns"])

    def test_concurrent_destination_preserves_owner_and_verified_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, trust = self._root_and_trust(directory)

            def completed_download_with_race(**kwargs: object) -> str:
                local = Path(str(kwargs["local_dir"]))
                (local / "generation-v1").mkdir()
                destination = root / "data/snapshot"
                destination.mkdir()
                (destination / "owner.txt").write_text("keep\n", encoding="utf-8")
                return str(local)

            qa = {
                "status": "pass",
                "pass_count": 986,
                "generation_id": "generation-v1",
            }
            with mock.patch.object(downloader, "ROOT", root), mock.patch.object(
                downloader,
                "load_published_truebones_release",
                return_value=trust,
            ), mock.patch.object(
                downloader,
                "snapshot_download",
                side_effect=completed_download_with_race,
            ), mock.patch.object(
                downloader, "validate_private_distribution", return_value=qa
            ), mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "data/snapshot"],
            ):
                with self.assertRaisesRegex(SystemExit, "destination appeared"):
                    downloader.main()
            self.assertEqual(
                (root / "data/snapshot/owner.txt").read_text(encoding="utf-8"),
                "keep\n",
            )
            retained = list((root / "data").glob(".snapshot.payload-*"))
            self.assertEqual(len(retained), 1)
            self.assertTrue((retained[0] / "generation-v1").is_dir())


if __name__ == "__main__":
    unittest.main()
