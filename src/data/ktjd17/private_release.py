"""Build and resolve host-sanitized private KTJD-17 distribution snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .codec import world_velocity
from .decoder import decode_ktjd17
from .encoder import load_skeleton
from .loader import load_motion_npz
from .truebones_full_build import (
    EXPECTED_ACCEPTED_IDENTITY_SHA256,
    EXPECTED_POSTBUILD_GATE_SHA256,
    EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256,
    EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
    EXPECTED_SCOPE,
    RELEASE_READY_STATUS,
    TruebonesFullBuildError,
    verify_postbuild_release_gate_payload,
    verify_full_generation,
)


PRIVATE_RELEASE_VERSION = "ktjd17-private-distribution-v1"
TRUST_RECORD_VERSION = "ktjd17-public-trust-v1"
RELEASE_POINTER_NAME = "RELEASE.json"
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\\\]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_LABEL_COMPONENT = re.compile(r"^[A-Za-z0-9._+-]+$")
_FILE_URI = re.compile(r"(?i)file" + r"://[^\s\"'`<>]+")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])[^\s\"'`<>]+")
_UNC_PATH = re.compile(r"(?<![\\])\\\\[^\s\"'`<>]+")
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:[A-Za-z0-9._~+-]+/)+[A-Za-z0-9._~%+-]+"
)
_MACHINE_HOST = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)+(?:local|internal|lan|cluster|corp|example)\b"
)
_RELATIVE_TRAVERSAL = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)")


class PrivateReleaseError(RuntimeError):
    """Raised when a private distribution snapshot is unsafe or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise PrivateReleaseError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateReleaseError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise PrivateReleaseError(f"{label} must not be hard-linked")
    return metadata


def _sha256_regular_file(path: Path, *, label: str) -> str:
    _regular_file_stat(path, label=label)
    return _sha256_file(path)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def resolve_repository_path(
    repository_root: str | Path,
    value: str | Path,
    *,
    argument_name: str,
    required_top_level: str | None = None,
    must_not_exist: bool = False,
) -> Path:
    """Resolve a user path and require it to stay inside the repository."""
    root = Path(repository_root).resolve()
    text = str(value)
    normalized = text.replace("\\", "/")
    raw_posix = PurePosixPath(normalized)
    raw = Path(*raw_posix.parts)
    if (
        not text
        or raw_posix.is_absolute()
        or text != normalized
        or any(part in {"", ".", ".."} for part in raw_posix.parts)
    ):
        raise PrivateReleaseError(f"{argument_name} must be repository-relative")
    if required_top_level is not None and (
        not raw_posix.parts or raw_posix.parts[0] != required_top_level
    ):
        raise PrivateReleaseError(
            f"{argument_name} must be below {required_top_level}/"
        )
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PrivateReleaseError(
            f"{argument_name} must stay inside the repository"
        ) from exc
    if required_top_level is not None:
        allowed_root = (root / required_top_level).resolve()
        if candidate == allowed_root or not candidate.is_relative_to(allowed_root):
            raise PrivateReleaseError(
                f"{argument_name} must name a child below {required_top_level}/"
            )
    if must_not_exist and (candidate.exists() or candidate.is_symlink()):
        raise PrivateReleaseError(f"{argument_name} destination already exists")
    return candidate


def _safe_pointer_subdir(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PrivateReleaseError("release pointer generation_subdir is missing")
    subdir = PurePosixPath(value)
    if subdir.is_absolute() or any(part in {"", ".", ".."} for part in subdir.parts):
        raise PrivateReleaseError("release pointer generation_subdir is unsafe")
    return subdir


def _validate_trusted_release_record(
    value: Mapping[str, Any],
    *,
    require_hf_revision: bool,
) -> dict[str, Any]:
    record = dict(value)
    required = {
        "trust_record_version",
        "release_version",
        "private_required",
        "repo_id",
        "repo_type",
        "generation_id",
        "generation_json_sha256",
        "release_pointer_sha256",
        "accepted_identity_sha256",
        "source_scope_identity_sha256",
        "hf_revision",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise PrivateReleaseError("trusted release record schema is not exact")
    if (
        record["trust_record_version"] != TRUST_RECORD_VERSION
        or record["release_version"] != PRIVATE_RELEASE_VERSION
        or record["private_required"] is not True
        or record["repo_type"] != "dataset"
        or not isinstance(record["repo_id"], str)
        or "/" not in record["repo_id"]
        or not isinstance(record["generation_id"], str)
        or not record["generation_id"]
        or record["accepted_identity_sha256"]
        != EXPECTED_ACCEPTED_IDENTITY_SHA256
        or record["source_scope_identity_sha256"]
        != EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256
    ):
        raise PrivateReleaseError("trusted release record values are invalid")
    for field in ("generation_json_sha256", "release_pointer_sha256"):
        if not isinstance(record[field], str) or _SHA256.fullmatch(record[field]) is None:
            raise PrivateReleaseError(f"trusted release field {field} is invalid")
    revision = record["hf_revision"]
    if revision is not None and (
        not isinstance(revision, str) or _HF_REVISION.fullmatch(revision) is None
    ):
        raise PrivateReleaseError("trusted Hugging Face revision is invalid")
    if require_hf_revision and revision is None:
        raise PrivateReleaseError("trusted Hugging Face revision is not published yet")
    return record


def load_trusted_release(
    path: str | Path,
    *,
    require_hf_revision: bool = False,
) -> dict[str, Any]:
    """Load the public, code-reviewed trust root for one private snapshot."""
    trust_path = Path(path)
    _regular_file_stat(trust_path, label="trusted release record")
    try:
        record = json.loads(trust_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateReleaseError(f"cannot read trusted release record: {exc}") from exc
    if not isinstance(record, Mapping):
        raise PrivateReleaseError("trusted release record root must be an object")
    return _validate_trusted_release_record(
        record, require_hf_revision=require_hf_revision
    )


def _require_directory_no_link(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise PrivateReleaseError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise PrivateReleaseError(f"{label} must be a real directory")


def _assert_snapshot_root_closure(snapshot: Path, generation_subdir: str) -> None:
    expected = {RELEASE_POINTER_NAME, generation_subdir}
    observed = {path.name for path in snapshot.iterdir()}
    if observed != expected:
        raise PrivateReleaseError(
            "release snapshot root closure failed: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def resolve_release_generation(
    snapshot_root: str | Path,
    *,
    trusted_release: Mapping[str, Any],
) -> Path:
    """Resolve a snapshot only when an external public trust record pins it."""
    trust = _validate_trusted_release_record(
        trusted_release, require_hf_revision=False
    )
    raw_snapshot = Path(snapshot_root)
    _require_directory_no_link(raw_snapshot, label="release snapshot root")
    snapshot = raw_snapshot.resolve()
    pointer_path = snapshot / RELEASE_POINTER_NAME
    pointer_sha = _sha256_regular_file(pointer_path, label=RELEASE_POINTER_NAME)
    if pointer_sha != trust["release_pointer_sha256"]:
        raise PrivateReleaseError("release pointer does not match the public trust root")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PrivateReleaseError(f"cannot read {RELEASE_POINTER_NAME}: {exc}") from exc
    expected_fields = {
        "release_version",
        "private_required",
        "generation_subdir",
        "generation_json_sha256",
        "accepted_identity_sha256",
        "source_scope_identity_sha256",
    }
    if not isinstance(pointer, dict) or set(pointer) != expected_fields:
        raise PrivateReleaseError("release pointer schema is not exact")
    if (
        pointer.get("release_version") != PRIVATE_RELEASE_VERSION
        or pointer.get("private_required") is not True
        or pointer.get("accepted_identity_sha256")
        != EXPECTED_ACCEPTED_IDENTITY_SHA256
        or pointer.get("source_scope_identity_sha256")
        != EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256
    ):
        raise PrivateReleaseError("release pointer contract fields are invalid")
    subdir = _safe_pointer_subdir(pointer.get("generation_subdir"))
    if subdir.as_posix() != trust["generation_id"]:
        raise PrivateReleaseError("release pointer generation identity drifted")
    generation_unresolved = snapshot / Path(*subdir.parts)
    _require_directory_no_link(generation_unresolved, label="release generation")
    generation = generation_unresolved.resolve()
    if generation.parent != snapshot:
        raise PrivateReleaseError("release pointer must name one direct generation child")
    generation_json = generation / "generation.json"
    observed_sha = _sha256_regular_file(
        generation_json, label="release generation.json"
    )
    expected_sha = pointer.get("generation_json_sha256")
    if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
        raise PrivateReleaseError("release pointer generation hash is invalid")
    if (
        observed_sha != expected_sha
        or observed_sha != trust["generation_json_sha256"]
    ):
        raise PrivateReleaseError("release pointer generation hash mismatch")
    _assert_snapshot_root_closure(snapshot, subdir.as_posix())
    return generation


def _write_release_pointer(release_root: Path, generation_root: Path) -> Path:
    root = release_root.resolve()
    generation = generation_root.resolve()
    try:
        relative = generation.relative_to(root)
    except ValueError as exc:
        raise PrivateReleaseError("release generation is outside its snapshot root") from exc
    if relative == Path(".") or not (generation / "generation.json").is_file():
        raise PrivateReleaseError("release pointer target is not a versioned generation")
    pointer = {
        "release_version": PRIVATE_RELEASE_VERSION,
        "private_required": True,
        "generation_subdir": relative.as_posix(),
        "generation_json_sha256": _sha256_file(generation / "generation.json"),
        "accepted_identity_sha256": EXPECTED_ACCEPTED_IDENTITY_SHA256,
        "source_scope_identity_sha256": EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
    }
    target = root / RELEASE_POINTER_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(pointer, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256_regular_file(target, label=RELEASE_POINTER_NAME) != _sha256_file(target):
        raise PrivateReleaseError("written release pointer could not be read exactly")
    return target


def _looks_absolute_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("\\\\")
        or value.lower().startswith("file" + "://")
        or _WINDOWS_ABSOLUTE.match(value) is not None
    )


def _contains_machine_reference_text(value: str) -> bool:
    return any(
        pattern.search(value) is not None
        for pattern in (
            _FILE_URI,
            _WINDOWS_PATH,
            _UNC_PATH,
            _POSIX_PATH,
            _MACHINE_HOST,
        )
    ) or _looks_absolute_path(value) or _RELATIVE_TRAVERSAL.search(value) is not None


def _safe_path_tail(parts: list[str]) -> list[str] | None:
    if not parts or any(
        part in {"", ".", ".."} or _SAFE_LABEL_COMPONENT.fullmatch(part) is None
        for part in parts
    ):
        return None
    return parts


def _sanitized_path_label(value: str) -> str:
    normalized = re.sub(r"(?i)^file" + r"://", "", value).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if "Truebone_Z-OO" in parts:
        index = parts.index("Truebone_Z-OO") + 1
        tail = _safe_path_tail(parts[index:])
        if tail:
            return PurePosixPath("sources", "truebones", *tail).as_posix()
    raw_filename = parts[-1] if parts else "source"
    filename = re.sub(r"[^A-Za-z0-9._+-]+", "_", raw_filename).strip(".")
    if not filename or filename in {".", ".."}:
        filename = "source"
    if filename == "cond.npy":
        label = (
            "legacy_truebones"
            if "anytop_truebones" in normalized.lower()
            else "current_btjd"
        )
        return PurePosixPath("sources", "conditioning", label, filename).as_posix()
    for marker, label in (
        (".ktjd17_manifest_generations", "parent_manifest"),
        (".ktjd17_freeze_generations", "frozen_schema"),
        (".ktjd17_truebones_forward_audit_generations", "forward_audit"),
    ):
        if marker in parts:
            index = parts.index(marker) + 1
            tail = _safe_path_tail(parts[index:])
            if tail:
                return PurePosixPath("provenance", label, *tail).as_posix()
    opaque = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return PurePosixPath("sources", "redacted", opaque, filename).as_posix()


def _sanitize_string(value: str, replacements: Mapping[str, str]) -> str:
    if value in replacements:
        return replacements[value]
    if _looks_absolute_path(value):
        return _sanitized_path_label(value)
    if _RELATIVE_TRAVERSAL.search(value) is not None:
        opaque = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"redacted-metadata-{opaque}"
    result = value
    for pattern in (_FILE_URI, _WINDOWS_PATH, _UNC_PATH, _POSIX_PATH):
        result = pattern.sub(lambda match: _sanitized_path_label(match.group(0)), result)
    result = _MACHINE_HOST.sub("redacted-host", result)
    if _contains_machine_reference_text(result):
        opaque = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"redacted-metadata-{opaque}"
    return result


def _sanitize_value(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            sanitized_key = _sanitize_string(key, replacements) if isinstance(key, str) else key
            if sanitized_key in result:
                raise PrivateReleaseError("metadata-key collision after sanitization")
            result[sanitized_key] = _sanitize_value(item, replacements)
        return result
    if isinstance(value, list):
        return [_sanitize_value(item, replacements) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, replacements)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    payload = b"".join(_canonical_json(value) + b"\n" for value in values)
    path.write_bytes(payload)


def _rewrite_skeleton(path: Path) -> tuple[str, str]:
    old_sha = _sha256_file(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    changed = False
    for key, array in list(arrays.items()):
        if array.dtype.kind not in {"U", "S"}:
            continue
        values = array.astype(str)
        rewritten: list[str] = []
        array_changed = False
        for item in values.reshape(-1):
            text = str(item)
            replacement: object = text
            if text.startswith("{") or text.startswith("["):
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    decoded = None
                if decoded is not None:
                    sanitized = _sanitize_value(decoded, {})
                    replacement = _canonical_json(sanitized).decode("utf-8")
            else:
                replacement = _sanitize_string(text, {})
            replacement_text = str(replacement)
            rewritten.append(replacement_text)
            array_changed = array_changed or replacement_text != text
        if array_changed:
            width = max(1, max(len(item) for item in rewritten))
            arrays[key] = np.asarray(rewritten, dtype=f"<U{width}").reshape(array.shape)
            changed = True
    if not changed:
        return old_sha, old_sha
    temporary = path.with_name(f".{path.name}.sanitizing")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return old_sha, _sha256_file(path)


def _file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivateReleaseError(f"symlink in private release: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise PrivateReleaseError(f"special file in private release: {path}")
        if metadata.st_nlink != 1:
            raise PrivateReleaseError(f"hard-linked file in private release: {path}")
        if path.name != "generation.json":
            relpath = path.relative_to(root).as_posix()
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": metadata.st_size,
            }
    return result


def _contains_machine_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_machine_reference(key) or _contains_machine_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_machine_reference(item) for item in value)
    return isinstance(value, str) and _contains_machine_reference_text(value)


def _assert_no_absolute_machine_paths(root: Path) -> None:
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            violations.append(f"{path.relative_to(root)}:unsafe-file-type")
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                for key in payload.files:
                    array = np.asarray(payload[key])
                    if array.dtype.kind not in {"U", "S"}:
                        continue
                    for item in array.astype(str).reshape(-1):
                        text = str(item)
                        decoded: object = text
                        if text.startswith("{") or text.startswith("["):
                            try:
                                decoded = json.loads(text)
                            except json.JSONDecodeError:
                                decoded = text
                        if _contains_machine_reference(decoded):
                            violations.append(f"{path.relative_to(root)}:{key}")
                            break
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                violations.append(f"{path.relative_to(root)}:unexpected-binary")
                continue
            if path.suffix in {".json", ".jsonl"}:
                try:
                    values = (
                        [json.loads(line) for line in text.splitlines() if line]
                        if path.suffix == ".jsonl"
                        else json.loads(text)
                    )
                except json.JSONDecodeError as exc:
                    raise PrivateReleaseError(f"invalid JSON during path scan: {path}") from exc
                if _contains_machine_reference(values):
                    violations.append(path.relative_to(root).as_posix())
            elif _contains_machine_reference_text(text):
                violations.append(path.relative_to(root).as_posix())
    if violations:
        raise PrivateReleaseError(
            f"host paths remain in private release: {violations[:10]}"
        )


def _load_postbuild_release_gate(
    path: Path,
    *,
    source_generation_id: str,
    source_generation_json_sha256: str,
) -> dict[str, Any]:
    _regular_file_stat(path, label="postbuild release gate")
    if _sha256_file(path) != EXPECTED_POSTBUILD_GATE_SHA256:
        raise PrivateReleaseError("postbuild release gate does not match the public pin")
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateReleaseError(f"cannot read postbuild release gate: {exc}") from exc
    if not isinstance(gate, Mapping):
        raise PrivateReleaseError("postbuild release gate root must be an object")
    try:
        verify_postbuild_release_gate_payload(
            gate,
            source_generation_id=source_generation_id,
            source_generation_json_sha256=source_generation_json_sha256,
        )
    except TruebonesFullBuildError as exc:
        raise PrivateReleaseError(str(exc)) from exc
    reviewer_path = path.with_name("truebones_visual_review_gpt56sol.md")
    if (
        _sha256_regular_file(reviewer_path, label="independent visual review")
        != EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256
    ):
        raise PrivateReleaseError("independent visual review does not match the public pin")
    sanitized = _sanitize_value(gate, {})
    if _contains_machine_reference(sanitized):
        raise PrivateReleaseError("postbuild release gate contains machine metadata")
    return sanitized


def _verify_distribution_payload_equivalence(source: Path, staging: Path) -> dict[str, int]:
    source_motions = sorted((source / "motions").glob("*.npz"))
    staging_motions = sorted((staging / "motions").glob("*.npz"))
    if [path.name for path in source_motions] != [path.name for path in staging_motions]:
        raise PrivateReleaseError("motion payload scope drifted during sanitization")
    for source_path, staging_path in zip(source_motions, staging_motions, strict=True):
        if _sha256_file(source_path) != _sha256_file(staging_path):
            raise PrivateReleaseError(f"motion payload bytes changed: {source_path.name}")

    source_skeletons = sorted((source / "skeletons").glob("*.npz"))
    staging_skeletons = sorted((staging / "skeletons").glob("*.npz"))
    if [path.name for path in source_skeletons] != [path.name for path in staging_skeletons]:
        raise PrivateReleaseError("skeleton payload scope drifted during sanitization")
    for source_path, staging_path in zip(source_skeletons, staging_skeletons, strict=True):
        with np.load(source_path, allow_pickle=False) as source_payload, np.load(
            staging_path, allow_pickle=False
        ) as staging_payload:
            if set(source_payload.files) != set(staging_payload.files):
                raise PrivateReleaseError(f"skeleton fields changed: {source_path.name}")
            for key in source_payload.files:
                source_array = np.asarray(source_payload[key])
                staging_array = np.asarray(staging_payload[key])
                if source_array.dtype.kind not in {"U", "S"} and not np.array_equal(
                    source_array, staging_array, equal_nan=True
                ):
                    raise PrivateReleaseError(
                        f"skeleton numeric payload changed: {source_path.name}:{key}"
                    )
    return {
        "motion_payload_count": len(source_motions),
        "numeric_skeleton_payload_count": len(source_skeletons),
    }


def prepare_private_distribution(
    source_generation: str | Path,
    output_parent: str | Path,
    *,
    postbuild_gate: str | Path,
) -> dict[str, object]:
    """Copy, sanitize, re-hash, and verify an immutable generation for private hosting."""
    source = Path(source_generation).resolve()
    source_metadata = verify_full_generation(source, require_complete=True)
    destination_parent = Path(output_parent).resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    if any(destination_parent.iterdir()):
        raise PrivateReleaseError(
            f"private release root must be empty: {destination_parent}"
        )
    destination = destination_parent / source.name
    if destination.exists() or destination.is_symlink():
        raise PrivateReleaseError(f"private release already exists: {destination}")
    source_generation_sha = _sha256_file(source / "generation.json")
    release_gate = _load_postbuild_release_gate(
        Path(postbuild_gate),
        source_generation_id=source.name,
        source_generation_json_sha256=source_generation_sha,
    )
    staging_parent = Path(tempfile.mkdtemp(prefix=".ktjd17-private-", dir=destination_parent))
    staging = staging_parent / source.name
    try:
        shutil.copytree(source, staging)
        replacements: dict[str, str] = {}
        skeleton_hashes: dict[str, str] = {}
        for skeleton_path in sorted((staging / "skeletons").glob("*.npz")):
            old_sha, new_sha = _rewrite_skeleton(skeleton_path)
            replacements[old_sha] = new_sha
            skeleton_hashes[skeleton_path.stem] = new_sha

        gate_path = staging / "evidence/postbuild_release_gate.json"
        _write_json(gate_path, release_gate)

        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.name == "generation.json":
                continue
            if path.suffix == ".json":
                original = json.loads(path.read_text(encoding="utf-8"))
                sanitized = _sanitize_value(original, replacements)
                if sanitized != original:
                    _write_json(path, sanitized)
            elif path.suffix == ".jsonl":
                original_rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                sanitized_rows = [
                    _sanitize_value(row, replacements) for row in original_rows
                ]
                if sanitized_rows != original_rows:
                    _write_jsonl(path, sanitized_rows)

        authority_path = staging / "qa/source_file_authority.json"
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        for key in ("source_bvh", "rest_bvh"):
            record = authority[key]
            record["entry_stream_sha256"] = hashlib.sha256(
                _canonical_json(record["entries"])
            ).hexdigest()
        _write_json(authority_path, authority)

        selection_path = staging / "manifests/full_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_core = {
            key: selection[key]
            for key in ("selection_authority", "selection_counts", "selected")
        }
        selection_sha = hashlib.sha256(_canonical_json(selection_core)).hexdigest()
        for relative in (
            "manifests/full_selection.json",
            "manifests/prototype_selection.json",
            "qa/encoder_summary.json",
        ):
            path = staging / relative
            value = json.loads(path.read_text(encoding="utf-8"))
            value["selection_sha256"] = selection_sha
            _write_json(path, value)

        generation_path = staging / "generation.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        generation = _sanitize_value(generation, replacements)
        generation["selection_sha256"] = selection_sha
        generation["source_authority_stream_sha256"] = authority["source_bvh"][
            "entry_stream_sha256"
        ]
        generation["rest_authority_stream_sha256"] = authority["rest_bvh"][
            "entry_stream_sha256"
        ]
        generation["distribution_export"] = {
            "version": PRIVATE_RELEASE_VERSION,
            "host_paths_sanitized": True,
            "source_generation_json_sha256": source_generation_sha,
            "motion_payloads_byte_identical": True,
            "skeleton_payloads_rehashed": len(skeleton_hashes),
            "accepted_identity_sha256": EXPECTED_ACCEPTED_IDENTITY_SHA256,
            "source_scope_identity_sha256": EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
            "postbuild_release_gate_relpath": "evidence/postbuild_release_gate.json",
            "postbuild_release_gate_sha256": _sha256_file(gate_path),
        }
        generation["status"] = RELEASE_READY_STATUS
        generation["postbuild_fixed_qa_complete"] = True
        generation["postbuild_visual_regression_complete"] = True
        generation["ready_for_training"] = True
        summary_path = staging / "qa/encoder_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = RELEASE_READY_STATUS
        summary["visual_qa_status"] = "postbuild_66_of_66_rigs_reviewed_pass"
        summary["ready_for_training"] = True
        _write_json(summary_path, summary)
        equivalence = _verify_distribution_payload_equivalence(source, staging)
        generation["distribution_export"].update(
            {
                "verified_motion_payload_count": equivalence["motion_payload_count"],
                "verified_numeric_skeleton_payload_count": equivalence[
                    "numeric_skeleton_payload_count"
                ],
                "numeric_skeleton_arrays_elementwise_identical": True,
            }
        )
        generation["files"] = _file_manifest(staging)
        _write_json(generation_path, generation)

        _assert_no_absolute_machine_paths(staging)
        verify_full_generation(staging, require_complete=True)
        os.replace(staging, destination)
        pointer_path = _write_release_pointer(destination_parent, destination)
        return {
            "status": "pass",
            "release_version": PRIVATE_RELEASE_VERSION,
            "generation_id": source.name,
            "generation_root": str(destination),
            "release_root": str(destination_parent),
            "release_pointer": str(pointer_path),
            "generation_json_sha256": _sha256_file(destination / "generation.json"),
            "source_generation_json_sha256": source_generation_sha,
            "sanitized_skeleton_count": len(skeleton_hashes),
            "accepted_clip_count": int(source_metadata["scope"]["accepted_clip_count"]),
            "release_pointer_sha256": _sha256_file(pointer_path),
        }
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)


def validate_private_distribution(
    root: str | Path,
    *,
    trusted_release: Mapping[str, Any],
) -> dict[str, object]:
    """Run self-contained integrity and decode QA without proprietary sources."""
    generation_root = resolve_release_generation(
        root, trusted_release=trusted_release
    )
    generation = verify_full_generation(generation_root, require_complete=True)
    manifests = [
        json.loads(line)
        for line in (generation_root / "manifests/clips.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    qa_records = [
        json.loads(line)
        for line in (generation_root / "qa/encoder_qa.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if (
        len(manifests) != EXPECTED_SCOPE["source_safe_clip_count"]
        or len(qa_records) != len(manifests)
    ):
        raise PrivateReleaseError("distribution QA scope is not the frozen accepted corpus")
    if any(record.get("status") != "pass" for record in qa_records):
        raise PrivateReleaseError("distribution contains a non-pass encoder QA record")

    skeleton_cache: dict[str, object] = {}
    aggregate = {
        "direct_vs_fk_max_norm": 0.0,
        "rigid_edge_max_norm": 0.0,
        "velocity_max_norm_fps": 0.0,
        "heading_unit_max_abs": 0.0,
    }
    max_t = 0
    max_j = 0
    for manifest in manifests:
        clip_id = str(manifest["clip_id"])
        rig_id = str(manifest["rig_id"])
        if rig_id not in skeleton_cache:
            skeleton_cache[rig_id] = load_skeleton(
                generation_root / str(manifest["skeleton_relpath"])
            )
        skeleton = skeleton_cache[rig_id]
        payload = load_motion_npz(
            generation_root / str(manifest["motion_relpath"]),
            expected_fps_target=30.0,
        )
        motion = np.asarray(payload["motion"], dtype=np.float64)
        heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
        decoded = decode_ktjd17(
            motion,
            parents=skeleton.parents,
            R_rest_global=skeleton.R_rest_global,
            R_rest_local=skeleton.R_rest_local,
            offset_parent_local=skeleton.offset_parent_local,
            rotation_source_kind=skeleton.rotation_source_kind,
            strict_gt=True,
        )
        if np.any(motion[:, 1:, 13:17] != 0.0):
            raise PrivateReleaseError(f"{clip_id}: non-root channels 13:17 are nonzero")
        if np.any(motion[~heading_valid, 0, 15:17] != 0.0):
            raise PrivateReleaseError(f"{clip_id}: invalid heading sentinel is nonzero")
        if np.any((motion[..., 12] != 0.0) & (motion[..., 12] != 1.0)):
            raise PrivateReleaseError(f"{clip_id}: contact is not binary")

        direct_fk = float(
            np.max(np.linalg.norm(decoded.positions_direct_minus_fk, axis=-1))
            / skeleton.s_rig
        )
        expected_velocity = world_velocity(decoded.positions_direct, fps=30.0)
        velocity = float(
            np.max(np.abs(expected_velocity - motion[..., 9:12]))
            / (skeleton.s_rig * 30.0)
        )
        rest_lengths = np.asarray(
            [
                np.linalg.norm(
                    skeleton.P_rest_global[child]
                    - skeleton.P_rest_global[int(skeleton.parents[child])]
                )
                for child in range(1, motion.shape[1])
            ],
            dtype=np.float64,
        )
        motion_lengths = np.stack(
            [
                np.linalg.norm(
                    decoded.positions_direct[:, child]
                    - decoded.positions_direct[:, int(skeleton.parents[child])],
                    axis=-1,
                )
                for child in range(1, motion.shape[1])
            ],
            axis=-1,
        )
        rigid_edge = float(
            np.max(np.abs(motion_lengths - rest_lengths[None])) / skeleton.s_rig
        )
        heading_norm = np.linalg.norm(motion[heading_valid, 0, 15:17], axis=-1)
        heading_unit = float(
            np.max(np.abs(heading_norm - 1.0), initial=0.0)
        )
        metrics = {
            "direct_vs_fk_max_norm": direct_fk,
            "rigid_edge_max_norm": rigid_edge,
            "velocity_max_norm_fps": velocity,
            "heading_unit_max_abs": heading_unit,
        }
        limits = {
            "direct_vs_fk_max_norm": 1e-4,
            "rigid_edge_max_norm": 1e-4,
            "velocity_max_norm_fps": 1e-5,
            "heading_unit_max_abs": 2e-6,
        }
        for key, value in metrics.items():
            aggregate[key] = max(aggregate[key], value)
            if value > limits[key]:
                raise PrivateReleaseError(
                    f"{clip_id}: {key}={value} exceeds {limits[key]}"
                )
        max_t = max(max_t, motion.shape[0])
        max_j = max(max_j, motion.shape[1])

    return {
        "status": "pass",
        "artifact_kind": "private_distribution_snapshot",
        "generation_id": generation["generation_id"],
        "generation_json_sha256": _sha256_file(
            generation_root / "generation.json"
        ),
        "clip_count": len(manifests),
        "pass_count": len(manifests),
        "fail_count": 0,
        "skeleton_count": len(skeleton_cache),
        "J_phys_max": max_j,
        "T_max_observed": max_t,
        "metrics_max": aggregate,
        "source_backed_checks": "not_run_proprietary_sources_not_required",
    }
