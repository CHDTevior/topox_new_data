from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/check_public_release.sh"


class PublicReleaseGuardTests(unittest.TestCase):
    def _repo(self, directory: str) -> Path:
        root = Path(directory)
        (root / "scripts").mkdir()
        shutil.copyfile(GUARD, root / "scripts/check_public_release.sh")
        (root / "README.md").write_text("# clean\n", encoding="utf-8")
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)
        subprocess.run(("git", "add", "."), cwd=root, check=True)
        return root

    def _run(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("bash", "scripts/check_public_release.sh"),
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_clean_text_only_candidate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(self._repo(directory))
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("public release check passed", result.stdout)

    def test_machine_secrets_binary_data_cache_and_runtime_artifacts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._repo(directory)
            (root / "posix.txt").write_text(
                "/" + "mnt/private/project\n", encoding="utf-8"
            )
            (root / "windows.txt").write_text(
                "C:"
                + chr(92)
                + "Users"
                + chr(92)
                + "alice"
                + chr(92)
                + "private\n",
                encoding="utf-8",
            )
            (root / "host.txt").write_text(
                "compute777" + "." + "internal" + "." + "example\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "PASSWORD" + "=not-a-real-secret\n", encoding="utf-8"
            )
            (root / "payload.bin").write_bytes(b"\x7fELF\x00payload")
            (root / "corpus.csv").write_text("clip_id,value\n", encoding="utf-8")
            (root / "runtime.log").write_text("runtime\n", encoding="utf-8")
            (root / ".pytest_cache").mkdir()
            (root / ".pytest_cache/CACHEDIR.TAG").write_text("cache\n", encoding="utf-8")
            subprocess.run(("git", "add", "-f", "."), cwd=root, check=True)
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            for expected in (
                "POSIX machine path",
                "Windows machine path",
                "machine hostname",
                "credential-like filename",
                "NUL/binary payload",
                "banned data/model/runtime artifact",
                "cache/generated path",
            ):
                self.assertIn(expected, result.stdout)

    def test_decoded_json_path_and_unexpected_executable_mode_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._repo(directory)
            (root / "encoded.json").write_bytes(
                b'{"path":"\\u002firidisfs\\u002fprivate"}\n'
            )
            executable = root / "metadata.json"
            executable.write_text("{}\n", encoding="utf-8")
            executable.chmod(0o755)
            subprocess.run(("git", "add", "."), cwd=root, check=True)
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("POSIX machine path in decoded Git index", result.stdout)
            self.assertIn("unexpected executable Git mode", result.stdout)

    def test_decoded_python_bytes_literal_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._repo(directory)
            (root / "encoded.py").write_text(
                "LEAK = b'\\x2firidisfs\\x2fprivate'\n",
                encoding="utf-8",
            )
            subprocess.run(("git", "add", "."), cwd=root, check=True)
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("decoded Python bytes", result.stdout)


if __name__ == "__main__":
    unittest.main()
