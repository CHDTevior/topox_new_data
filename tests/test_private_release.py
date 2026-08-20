from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.download_private_dataset as downloader
from src.data.ktjd17.private_release import (
    PrivateReleaseError,
    _write_release_pointer,
    _sanitized_path_label,
    resolve_release_generation,
    resolve_repository_path,
)


class RepositoryPathTests(unittest.TestCase):
    def test_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PrivateReleaseError, "stay inside"):
                resolve_repository_path(
                    directory, "../outside", argument_name="--local-dir"
                )

    def test_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(PrivateReleaseError, "stay inside"):
                resolve_repository_path(
                    root, "escape/dataset", argument_name="--local-dir"
                )

    def test_truebones_path_becomes_stable_relative_label(self) -> None:
        value = "/machine/data/Truebone_Z-OO/Dragon/Fly.bvh"
        self.assertEqual(
            _sanitized_path_label(value),
            "sources/truebones/Dragon/Fly.bvh",
        )


class ReleasePointerTests(unittest.TestCase):
    def test_resolves_hash_pinned_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "data/generation-v1"
            generation.mkdir(parents=True)
            payload = b'{"generation_id":"generation-v1"}\n'
            (generation / "generation.json").write_bytes(payload)
            (root / "RELEASE.json").write_text(
                json.dumps(
                    {
                        "generation_subdir": "data/generation-v1",
                        "generation_json_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_release_generation(root), generation.resolve())

    def test_rejects_pointer_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE.json").write_text(
                json.dumps(
                    {
                        "generation_subdir": "../outside",
                        "generation_json_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PrivateReleaseError, "unsafe"):
                resolve_release_generation(root)

    def test_required_pointer_cannot_be_bypassed_by_direct_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generation.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PrivateReleaseError, "RELEASE.json"):
                resolve_release_generation(root, require_pointer=True)

    def test_writer_emits_resolvable_hash_pinned_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation-v1"
            generation.mkdir()
            (generation / "generation.json").write_text(
                '{"generation_id":"generation-v1"}\n', encoding="utf-8"
            )
            pointer = _write_release_pointer(root, generation)
            self.assertEqual(pointer, root / "RELEASE.json")
            self.assertEqual(
                resolve_release_generation(root, require_pointer=True),
                generation.resolve(),
            )


class DownloaderTests(unittest.TestCase):
    def test_parent_escape_never_reaches_hugging_face(self) -> None:
        with mock.patch.object(downloader, "snapshot_download") as snapshot:
            with mock.patch.object(
                sys,
                "argv",
                ["download_private_dataset.py", "--local-dir", "../outside"],
            ):
                with self.assertRaisesRegex(SystemExit, "stay inside"):
                    downloader.main()
            snapshot.assert_not_called()

    def test_incomplete_snapshot_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".test-private-download-", dir=downloader.ROOT
        ) as directory:
            root = Path(directory)
            generation = root / "data/incomplete-generation"
            generation.mkdir(parents=True)
            payload = b'{"generation_id":"incomplete-generation"}\n'
            (generation / "generation.json").write_bytes(payload)
            (root / "RELEASE.json").write_text(
                json.dumps(
                    {
                        "generation_subdir": "data/incomplete-generation",
                        "generation_json_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                downloader, "snapshot_download", return_value=str(root)
            ):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "download_private_dataset.py",
                        "--local-dir",
                        root.relative_to(downloader.ROOT).as_posix(),
                    ],
                ):
                    with self.assertRaisesRegex(SystemExit, "download verification failed"):
                        downloader.main()


if __name__ == "__main__":
    unittest.main()
