from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.download_private_dataset as downloader
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
    load_trusted_release,
    resolve_release_generation,
    resolve_repository_path,
)
from src.data.ktjd17.truebones_full_build import (
    EXPECTED_ACCEPTED_IDENTITY_SHA256,
    EXPECTED_SOURCE_GENERATION_ID,
    EXPECTED_SOURCE_GENERATION_JSON_SHA256,
    EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
)


REVISION = "1" * 40


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
            for value in (".", ".git", "scripts", "README.md", "data", "data/existing"):
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
            (root / "opaque.bin").write_bytes(b"\xff/" + b"iridisfs/private")
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
    def _root_and_trust(self, directory: str) -> tuple[Path, dict[str, object]]:
        root = Path(directory)
        (root / "release").mkdir()
        trust = _trust_record("generation-v1", "0" * 64, "1" * 64)
        (root / "release/truebones_v1.json").write_text(
            json.dumps(trust), encoding="utf-8"
        )
        return root, trust

    def test_parent_escape_and_preexisting_destination_never_reach_hf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _trust = self._root_and_trust(directory)
            (root / "data/existing").mkdir(parents=True)
            for local_dir in ("../outside", "data/existing"):
                with self.subTest(local_dir=local_dir), mock.patch.object(
                    downloader, "ROOT", root
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
                downloader, "snapshot_download", side_effect=partial_download
            ), mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "data/snapshot"],
            ):
                with self.assertRaisesRegex(SystemExit, "transport failed"):
                    downloader.main()
            self.assertFalse((root / "data/snapshot").exists())
            self.assertEqual(list((root / "data").glob(".snapshot.download-*")), [])

    def test_success_is_published_only_after_all_verifiers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, trust = self._root_and_trust(directory)

            def completed_download(**kwargs: object) -> str:
                local = Path(str(kwargs["local_dir"]))
                (local / "generation-v1").mkdir()
                return str(local)

            generation = {"generation_id": "generation-v1"}
            qa = {"status": "pass", "pass_count": 986}
            with mock.patch.object(downloader, "ROOT", root), mock.patch.object(
                downloader, "snapshot_download", side_effect=completed_download
            ) as snapshot, mock.patch.object(
                downloader,
                "resolve_release_generation",
                side_effect=lambda local, trusted_release: Path(local) / "generation-v1",
            ), mock.patch.object(
                downloader, "verify_full_generation", return_value=generation
            ), mock.patch.object(
                downloader, "validate_private_distribution", return_value=qa
            ), mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "data/snapshot"],
            ):
                self.assertEqual(downloader.main(), 0)
            self.assertTrue((root / "data/snapshot/generation-v1").is_dir())
            kwargs = snapshot.call_args.kwargs
            self.assertEqual(kwargs["revision"], trust["hf_revision"])
            self.assertNotIn("README.md", kwargs["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
