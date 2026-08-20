"""Build and resolve host-sanitized private KTJD-17 distribution snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .codec import world_velocity
from .decoder import decode_ktjd17
from .encoder import load_skeleton
from .loader import load_motion_npz
from .truebones_full_build import (
    EXPECTED_ACCEPTED_IDENTITY_SHA256,
    EXPECTED_FIXED_QA_REPORT_SHA256,
    EXPECTED_POSTBUILD_GATE_SHA256,
    EXPECTED_POSTBUILD_VISUAL_GENERATION_ID,
    EXPECTED_POSTBUILD_VISUAL_INDEX_SHA256,
    EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256,
    EXPECTED_POSTBUILD_VISUAL_REVIEW_THREAD_ID,
    EXPECTED_SOURCE_GENERATION_ID,
    EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
    EXPECTED_SCOPE,
    EXPECTED_VISUAL_EQUIVALENCE_REPORT_SHA256,
    RELEASE_READY_STATUS,
    TruebonesFullBuildError,
    verify_postbuild_release_gate_payload,
    verify_full_generation,
    verify_full_generation_file_closure,
)


PRIVATE_RELEASE_VERSION = "ktjd17-private-distribution-v1"
TRUST_RECORD_VERSION = "ktjd17-public-trust-v1"
RELEASE_POINTER_NAME = "RELEASE.json"
EVIDENCE_BUNDLE_VERSION = "ktjd17-postbuild-evidence-bundle-v1"
PUBLISHED_TRUEBONES_REPO_ID = "Tevior/KTJD17-Truebones-v1"
PUBLISHED_TRUEBONES_GENERATION_SHA256 = (
    "170d3d55bd12ebb22b221c27339bf62ecd99e721825157ba5912607fdf8518ec"
)
PUBLISHED_TRUEBONES_POINTER_SHA256 = (
    "7752119af15e8945f1d10fd3c8743b311cd8b1ca0ac35976f5664edcae07e35d"
)
# This pin is intentionally null until the first private upload returns its
# immutable Hugging Face commit. The downloader requires a non-null value, so
# no network fetch is authorized in the bootstrap commit.
PUBLISHED_TRUEBONES_HF_REVISION: str | None = None
_SOURCE_FIXED_QA_REPORT_SHA256 = (
    "fcf6fb1ce9ede7e1db035dd7c617631e8f729532d35e1ce79e6694029129f1ae"
)
_SOURCE_VISUAL_INDEX_SHA256 = (
    "6754c9d8f909fe11f04c44e3499c69f49c1c524eade9e4634c9a3827755b3992"
)
_SOURCE_VISUAL_EQUIVALENCE_SHA256 = (
    "73e29dd5039c110df41a1b8f1939458ef9570a403589b8024340cf26fa8df4ba"
)
_SOURCE_VISUAL_GENERATION_SHA256 = (
    "cb3d9e2bfeffa76717bcb49564f67824c36f8584e8eae81d7dc374a7fc5413e7"
)
_REVIEW_FILMSTRIP_ATTACHMENTS = (
    "Anaconda___Spin_31-5bbd790a20_filmstrip.png",
    "Bat___AttackBite_65-2d735b5cf1_filmstrip.png",
    "Bear___SwatLeft_74-00d51329cd_filmstrip.png",
    "Dragon___Attack2_295-53ee29f457_filmstrip.png",
    "HermitCrab___Fighting_410-b2a559e35f_filmstrip.png",
    "Monkey___Attack1_579-0a52d358c6_filmstrip.png",
    "Pirrana___Flopping_624-bb67da541e_filmstrip.png",
    "Turtle___Attack3_1060-a8421486ed_filmstrip.png",
)
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
_MOTION_NPZ_MEMBERS = frozenset(
    {
        "clip_id.npy",
        "fps_target.npy",
        "heading_valid.npy",
        "motion.npy",
        "origin_xz.npy",
        "rig_id.npy",
    }
)
_SKELETON_NPZ_MEMBERS = frozenset(
    {
        "P_rest_global.npy",
        "R_rest_global.npy",
        "R_rest_local.npy",
        "artifact_status.npy",
        "conditioning_authority.npy",
        "fixed_rig_rotation_signatures.npy",
        "heading_carrier_joint.npy",
        "heading_payload_provenance.npy",
        "joint_map_metadata.npy",
        "joint_names.npy",
        "offset_parent_local.npy",
        "parents.npy",
        "position_geometry_provenance.npy",
        "reason_codes.npy",
        "representative_clip_id.npy",
        "rig_id.npy",
        "rotation_source_kind.npy",
        "s_rig.npy",
        "skeleton_format_version.npy",
        "source_family.npy",
        "source_rest_path.npy",
        "source_to_canonical_C.npy",
        "source_to_canonical_alpha.npy",
        "source_to_canonical_o.npy",
        "source_to_canonical_provenance.npy",
        "topology_family.npy",
        "u_forward_local.npy",
        "unit_metadata.npy",
    }
)
_GAINS_NPZ_MEMBERS = frozenset(
    {
        "calibration_generation_id.npy",
        "calibration_version.npy",
        "clip_ids.npy",
        "freeze_generation_id.npy",
        "frozen.npy",
        "g_q.npy",
        "g_s.npy",
        "g_v.npy",
        "gains.npy",
        "prototype_generation_id.npy",
        "source_rms.npy",
        "split.npy",
        "valid_scalar_counts.npy",
        "visual_generation_id.npy",
    }
)
_NPZ_MAX_MEMBER_BYTES = 8 * 1024 * 1024
_NPZ_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_NPZ_MAX_COMPRESSION_RATIO = 256.0


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
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o7133 or mode & 0o400 == 0:
        raise PrivateReleaseError(f"{label} has unsafe mode {mode:04o}")
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


def load_published_truebones_release(
    path: str | Path,
    *,
    require_hf_revision: bool = False,
) -> dict[str, Any]:
    """Load the one release identity compiled into this audited code snapshot."""
    record = load_trusted_release(path, require_hf_revision=require_hf_revision)
    expected = {
        "trust_record_version": TRUST_RECORD_VERSION,
        "release_version": PRIVATE_RELEASE_VERSION,
        "private_required": True,
        "repo_id": PUBLISHED_TRUEBONES_REPO_ID,
        "repo_type": "dataset",
        "generation_id": EXPECTED_SOURCE_GENERATION_ID,
        "generation_json_sha256": PUBLISHED_TRUEBONES_GENERATION_SHA256,
        "release_pointer_sha256": PUBLISHED_TRUEBONES_POINTER_SHA256,
        "accepted_identity_sha256": EXPECTED_ACCEPTED_IDENTITY_SHA256,
        "source_scope_identity_sha256": EXPECTED_SOURCE_SCOPE_IDENTITY_SHA256,
        "hf_revision": PUBLISHED_TRUEBONES_HF_REVISION,
    }
    if record != expected:
        raise PrivateReleaseError(
            "public Truebones trust record does not match the compiled release identity"
        )
    return record


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
            stripped = text.lstrip()
            if stripped.startswith(("{", "[", '"')):
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
    for path in (root, *sorted(root.rglob("*"))):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivateReleaseError(f"symlink in private release: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            mode = stat.S_IMODE(metadata.st_mode)
            if mode not in {0o700, 0o750, 0o755}:
                raise PrivateReleaseError(
                    f"unsafe directory mode in private release: {path} ({mode:04o})"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise PrivateReleaseError(f"special file in private release: {path}")
        if metadata.st_nlink != 1:
            raise PrivateReleaseError(f"hard-linked file in private release: {path}")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o7133 or mode & 0o400 == 0:
            raise PrivateReleaseError(
                f"unsafe file mode in private release: {path} ({mode:04o})"
            )
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


def _expected_npz_members(path: Path) -> frozenset[str]:
    if path.parent.name == "motions":
        return _MOTION_NPZ_MEMBERS
    if path.parent.name == "skeletons":
        return _SKELETON_NPZ_MEMBERS
    if path.parent.name == "stats" and path.name == "train_block_gains.npz":
        return _GAINS_NPZ_MEMBERS
    raise PrivateReleaseError(f"NPZ file is outside the frozen release schema: {path}")


def _validate_npz_archive(path: Path) -> None:
    """Validate exact members and bounded ZIP structure before NumPy sees an NPZ."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected_names = _expected_npz_members(path)
            if (
                not members
                or len(names) != len(set(names))
                or set(names) != expected_names
            ):
                raise PrivateReleaseError(f"NPZ member closure is invalid: {path}")
            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed > _NPZ_MAX_TOTAL_BYTES:
                raise PrivateReleaseError(f"NPZ archive is too large: {path}")
            for member in members:
                name = member.filename
                pure = PurePosixPath(name)
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                compression_ratio = member.file_size / max(member.compress_size, 1)
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or "\\" in name
                    or pure.is_absolute()
                    or len(pure.parts) != 1
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or name.startswith(".")
                    or not name.endswith(".npy")
                    or name == ".npy"
                    or file_type not in {0, stat.S_IFREG}
                    or unix_mode & 0o7111
                    or member.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or member.file_size > _NPZ_MAX_MEMBER_BYTES
                    or compression_ratio > _NPZ_MAX_COMPRESSION_RATIO
                ):
                    raise PrivateReleaseError(
                        f"unsafe NPZ archive member {name!r}: {path}"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise PrivateReleaseError(f"invalid NPZ archive: {path}: {exc}") from exc


def _metadata_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {_metadata_text(key): _metadata_text(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_text(item) for item in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PrivateReleaseError("non-UTF-8 binary image metadata") from exc
    return value


def _validate_image_evidence(path: Path) -> None:
    expected_format = {".gif": "GIF", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}[
        path.suffix.lower()
    ]
    try:
        with Image.open(path) as image:
            if image.format != expected_format:
                raise PrivateReleaseError(
                    f"image format/extension mismatch: {path}"
                )
            metadata = _metadata_text(dict(image.info))
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise PrivateReleaseError(f"invalid image evidence: {path}: {exc}") from exc
    if _contains_machine_reference(metadata):
        raise PrivateReleaseError(f"host paths remain in image metadata: {path}")


def _preflight_private_release_structure(root: Path) -> list[Path]:
    violations: list[str] = []
    paths = [root, *sorted(root.rglob("*"))]
    regular_files: list[Path] = []

    for path in paths:
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            mode = stat.S_IMODE(metadata.st_mode)
            if mode not in {0o700, 0o750, 0o755}:
                violations.append(
                    f"{path.relative_to(root)}:unsafe-directory-mode-{mode:04o}"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            violations.append(f"{path.relative_to(root)}:unsafe-file-type")
            continue
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o7133 or mode & 0o400 == 0:
            violations.append(f"{path.relative_to(root)}:unsafe-mode-{mode:04o}")
            continue
        if path.suffix == ".npz":
            _validate_npz_archive(path)
        regular_files.append(path)

    if violations:
        raise PrivateReleaseError(
            f"host paths remain in private release: {violations[:10]}"
        )
    return regular_files


def _assert_no_absolute_machine_paths(root: Path) -> None:
    violations: list[str] = []
    # Complete the filesystem and ZIP-structure pass before loading any array.
    regular_files = _preflight_private_release_structure(root)

    for path in regular_files:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                for key in payload.files:
                    array = np.asarray(payload[key])
                    if (
                        array.dtype.hasobject
                        or array.dtype.fields is not None
                        or array.dtype.metadata is not None
                    ):
                        violations.append(
                            f"{path.relative_to(root)}:{key}:unsafe-NPZ-dtype"
                        )
                        continue
                    if array.dtype.kind not in {"U", "S"}:
                        continue
                    for item in array.reshape(-1):
                        if array.dtype.kind == "S":
                            try:
                                text = bytes(item).decode("utf-8")
                            except UnicodeDecodeError:
                                violations.append(
                                    f"{path.relative_to(root)}:{key}:non-UTF-8"
                                )
                                break
                        else:
                            text = str(item)
                        decoded: object = text
                        stripped = text.lstrip()
                        if stripped.startswith(("{", "[", '"')):
                            try:
                                decoded = json.loads(text)
                            except json.JSONDecodeError:
                                decoded = text
                        if _contains_machine_reference(decoded):
                            violations.append(f"{path.relative_to(root)}:{key}")
                            break
        elif path.suffix.lower() in {".gif", ".jpg", ".jpeg", ".png"}:
            _validate_image_evidence(path)
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


def _copy_regular(source: Path, destination: Path) -> dict[str, int | str]:
    metadata = _regular_file_stat(source, label=f"evidence source {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PrivateReleaseError(f"evidence destination already exists: {destination}")
    shutil.copyfile(source, destination)
    destination.chmod(0o644)
    return {"sha256": _sha256_file(destination), "size_bytes": metadata.st_size}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file_stat(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateReleaseError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PrivateReleaseError(f"{label} root must be an object")
    return value


def _verify_source_visual_evidence(root: Path) -> dict[str, dict[str, int | str]]:
    _require_directory_no_link(root, label="postbuild visual generation")
    generation_path = root / "generation.json"
    if _sha256_regular_file(
        generation_path, label="postbuild visual generation.json"
    ) != _SOURCE_VISUAL_GENERATION_SHA256:
        raise PrivateReleaseError("postbuild visual generation hash drifted")
    generation = _load_json_object(
        generation_path, label="postbuild visual generation.json"
    )
    files = generation.get("files")
    if (
        generation.get("generation_id") != EXPECTED_POSTBUILD_VISUAL_GENERATION_ID
        or not isinstance(files, dict)
    ):
        raise PrivateReleaseError("postbuild visual generation identity is invalid")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "generation.json"
    }
    if observed != set(files):
        raise PrivateReleaseError("postbuild visual generation file closure failed")
    normalized: dict[str, dict[str, int | str]] = {}
    for relpath, expected in files.items():
        pure = PurePosixPath(relpath)
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or not isinstance(expected, dict)
        ):
            raise PrivateReleaseError("postbuild visual manifest contains an unsafe path")
        source = root / Path(*pure.parts)
        metadata = _regular_file_stat(source, label=f"visual evidence {relpath}")
        record = {"sha256": _sha256_file(source), "size_bytes": metadata.st_size}
        if record != expected:
            raise PrivateReleaseError(f"postbuild visual artifact drifted: {relpath}")
        normalized[relpath] = record
    if normalized.get("visual_qa_index.json", {}).get("sha256") != _SOURCE_VISUAL_INDEX_SHA256:
        raise PrivateReleaseError("postbuild source visual index drifted")
    image_paths = {
        relpath
        for relpath in normalized
        if relpath.startswith("clips/")
        and PurePosixPath(relpath).suffix.lower() in {".gif", ".png"}
    }
    if len(image_paths) != 198 or set(normalized) != image_paths | {"visual_qa_index.json"}:
        raise PrivateReleaseError("postbuild visual evidence is not the exact 198-artifact set")
    return normalized


def _write_manifest(path: Path, files: Mapping[str, Mapping[str, int | str]]) -> str:
    _write_json(path, {"files": dict(files)})
    return _sha256_file(path)


def _prepare_postbuild_evidence_bundle(
    staging: Path,
    *,
    release_gate: Mapping[str, Any],
    gate_source: Path,
    fixed_qa_report: Path,
    visual_generation: Path,
    visual_equivalence_report: Path,
    review_contact_sheets: Path,
    enforce_gate_hashes: bool = True,
) -> dict[str, Any]:
    evidence = staging / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    visual_files = _verify_source_visual_evidence(visual_generation)

    fixed_source_sha = _sha256_regular_file(
        fixed_qa_report, label="source fixed-QA report"
    )
    if fixed_source_sha != _SOURCE_FIXED_QA_REPORT_SHA256:
        raise PrivateReleaseError("source fixed-QA report hash drifted")
    fixed = _sanitize_value(
        _load_json_object(fixed_qa_report, label="source fixed-QA report"), {}
    )
    fixed["release_evidence"] = {
        "host_paths_sanitized": True,
        "source_report_sha256": fixed_source_sha,
    }
    fixed_target = evidence / "fixed_qa_report.json"
    _write_json(fixed_target, fixed)
    fixed_sha = _sha256_file(fixed_target)

    index_source = visual_generation / "visual_qa_index.json"
    index = _sanitize_value(
        _load_json_object(index_source, label="source visual-QA index"), {}
    )
    index["status"] = "pass"
    index["full_conversion_authorized"] = True
    for clip in index.get("clips", []):
        if isinstance(clip, dict):
            clip["inspection_status"] = "pass_postbuild_gpt56sol_xhigh"
    index["postbuild_independent_review"] = {
        "attached_image_count": 19,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "xhigh",
        "report_sha256": EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256,
        "thread_id": EXPECTED_POSTBUILD_VISUAL_REVIEW_THREAD_ID,
        "verdict": "pass",
    }
    visual_target_root = evidence / "visual_qa"
    index_target = visual_target_root / "visual_qa_index.json"
    index_target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(index_target, index)
    index_sha = _sha256_file(index_target)

    artifact_manifest: dict[str, dict[str, int | str]] = {}
    for relpath, expected in sorted(visual_files.items()):
        if relpath == "visual_qa_index.json":
            continue
        destination = visual_target_root / Path(*PurePosixPath(relpath).parts)
        observed = _copy_regular(visual_generation / relpath, destination)
        if observed != expected:
            raise PrivateReleaseError(f"copied visual artifact drifted: {relpath}")
        artifact_manifest[relpath] = observed
    artifact_manifest_sha = _write_manifest(
        visual_target_root / "artifact_manifest.json", artifact_manifest
    )

    equivalence_source_sha = _sha256_regular_file(
        visual_equivalence_report, label="source visual-equivalence report"
    )
    if equivalence_source_sha != _SOURCE_VISUAL_EQUIVALENCE_SHA256:
        raise PrivateReleaseError("source visual-equivalence report hash drifted")
    equivalence = _sanitize_value(
        _load_json_object(
            visual_equivalence_report, label="source visual-equivalence report"
        ),
        {},
    )
    equivalence["artifact_kind"] = "ktjd17_postbuild_visual_release_equivalence"
    equivalence["postbuild_visual_generation"] = "evidence/visual_qa"
    equivalence["prebuild_visual_generation"] = "not_distributed_private_source_evidence"
    equivalence["postbuild_visual_index_sha256"] = index_sha
    equivalence["source_report_sha256"] = equivalence_source_sha
    equivalence["postbuild_independent_review"] = index[
        "postbuild_independent_review"
    ]
    equivalence_target = evidence / "visual_equivalence_report.json"
    _write_json(equivalence_target, equivalence)
    equivalence_sha = _sha256_file(equivalence_target)

    review_source = gate_source.with_name("truebones_visual_review_gpt56sol.md")
    review_target = evidence / "truebones_visual_review_gpt56sol.md"
    review_record = _copy_regular(review_source, review_target)
    if review_record["sha256"] != EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256:
        raise PrivateReleaseError("independent visual-review evidence drifted")

    _require_directory_no_link(
        review_contact_sheets, label="visual-review contact sheets"
    )
    expected_sheet_names = {f"filmstrips_{index:02d}.jpg" for index in range(1, 12)}
    observed_sheet_names = {
        path.name for path in review_contact_sheets.iterdir() if path.is_file()
    }
    if observed_sheet_names != expected_sheet_names:
        raise PrivateReleaseError("visual-review contact-sheet closure failed")
    sheet_manifest: dict[str, dict[str, int | str]] = {}
    sheets_target = evidence / "review_contact_sheets"
    for name in sorted(expected_sheet_names):
        sheet_manifest[name] = _copy_regular(
            review_contact_sheets / name, sheets_target / name
        )
    sheet_manifest_sha = _write_manifest(
        sheets_target / "manifest.json", sheet_manifest
    )

    attachments: dict[str, dict[str, int | str]] = {
        f"review_contact_sheets/{name}": record
        for name, record in sheet_manifest.items()
    }
    for name in _REVIEW_FILMSTRIP_ATTACHMENTS:
        relpath = f"clips/{name}"
        if relpath not in artifact_manifest:
            raise PrivateReleaseError(f"review filmstrip is absent: {name}")
        attachments[f"visual_qa/{relpath}"] = artifact_manifest[relpath]
    if len(attachments) != 19:
        raise PrivateReleaseError("independent review attachment closure is not 19 images")
    attachment_manifest_sha = _write_manifest(
        evidence / "review_attachment_manifest.json", attachments
    )

    fixed_gate = release_gate.get("fixed_qa")
    visual_gate = release_gate.get("visual_qa")
    if not isinstance(fixed_gate, Mapping) or not isinstance(visual_gate, Mapping):
        raise PrivateReleaseError("release gate evidence sections are invalid")
    if enforce_gate_hashes and (
        fixed_gate.get("report_sha256") != fixed_sha
        or visual_gate.get("visual_index_sha256") != index_sha
        or visual_gate.get("visual_equivalence_report_sha256") != equivalence_sha
    ):
        raise PrivateReleaseError("release gate does not pin the sanitized evidence bundle")

    bundle = {
        "bundle_version": EVIDENCE_BUNDLE_VERSION,
        "coordinate_contract": visual_gate.get("coordinate_contract"),
        "fixed_qa": {
            "relpath": "fixed_qa_report.json",
            "sha256": fixed_sha,
        },
        "visual_index": {
            "relpath": "visual_qa/visual_qa_index.json",
            "sha256": index_sha,
        },
        "visual_equivalence": {
            "relpath": "visual_equivalence_report.json",
            "sha256": equivalence_sha,
        },
        "visual_artifacts": {
            "count": len(artifact_manifest),
            "manifest_relpath": "visual_qa/artifact_manifest.json",
            "manifest_sha256": artifact_manifest_sha,
        },
        "review_contact_sheets": {
            "count": len(sheet_manifest),
            "manifest_relpath": "review_contact_sheets/manifest.json",
            "manifest_sha256": sheet_manifest_sha,
        },
        "independent_review": {
            "attachment_count": len(attachments),
            "attachment_manifest_relpath": "review_attachment_manifest.json",
            "attachment_manifest_sha256": attachment_manifest_sha,
            "model": "gpt-5.6-sol",
            "reasoning_effort": "xhigh",
            "report_relpath": "truebones_visual_review_gpt56sol.md",
            "report_sha256": review_record["sha256"],
            "thread_id": EXPECTED_POSTBUILD_VISUAL_REVIEW_THREAD_ID,
            "verdict": "pass",
        },
        "status": "pass",
    }
    bundle_path = evidence / "bundle_manifest.json"
    _write_json(bundle_path, bundle)
    bundle["bundle_manifest_sha256"] = _sha256_file(bundle_path)
    return bundle


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


def _verify_manifest_files(
    root: Path,
    manifest_path: Path,
    *,
    expected_count: int,
) -> dict[str, dict[str, int | str]]:
    manifest = _load_json_object(manifest_path, label=f"evidence manifest {manifest_path}")
    if set(manifest) != {"files"}:
        raise PrivateReleaseError(f"evidence manifest schema drifted: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != expected_count:
        raise PrivateReleaseError(f"evidence manifest count drifted: {manifest_path}")
    normalized: dict[str, dict[str, int | str]] = {}
    for relpath, expected in files.items():
        pure = PurePosixPath(relpath)
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or not isinstance(expected, dict)
        ):
            raise PrivateReleaseError(f"unsafe evidence manifest path: {relpath}")
        target = root / Path(*pure.parts)
        metadata = _regular_file_stat(target, label=f"evidence file {relpath}")
        observed = {"sha256": _sha256_file(target), "size_bytes": metadata.st_size}
        if observed != expected:
            raise PrivateReleaseError(f"evidence artifact drifted: {relpath}")
        normalized[relpath] = observed
    return normalized


def _verify_postbuild_evidence_bundle(generation_root: Path) -> dict[str, Any]:
    evidence = generation_root / "evidence"
    bundle_path = evidence / "bundle_manifest.json"
    bundle = _load_json_object(bundle_path, label="postbuild evidence bundle")
    required = {
        "bundle_version",
        "coordinate_contract",
        "fixed_qa",
        "visual_index",
        "visual_equivalence",
        "visual_artifacts",
        "review_contact_sheets",
        "independent_review",
        "status",
    }
    if set(bundle) != required or (
        bundle.get("bundle_version") != EVIDENCE_BUNDLE_VERSION
        or bundle.get("status") != "pass"
    ):
        raise PrivateReleaseError("postbuild evidence bundle schema or status drifted")

    pinned = (
        ("fixed_qa", "fixed_qa_report.json", EXPECTED_FIXED_QA_REPORT_SHA256),
        (
            "visual_index",
            "visual_qa/visual_qa_index.json",
            EXPECTED_POSTBUILD_VISUAL_INDEX_SHA256,
        ),
        (
            "visual_equivalence",
            "visual_equivalence_report.json",
            EXPECTED_VISUAL_EQUIVALENCE_REPORT_SHA256,
        ),
    )
    for section, expected_relpath, expected_sha in pinned:
        record = bundle.get(section)
        if not isinstance(record, dict) or record != {
            "relpath": expected_relpath,
            "sha256": expected_sha,
        }:
            raise PrivateReleaseError(f"postbuild evidence pin drifted: {section}")
        target = evidence / expected_relpath
        if _sha256_regular_file(target, label=section) != expected_sha:
            raise PrivateReleaseError(f"postbuild evidence hash drifted: {section}")

    visual = bundle.get("visual_artifacts")
    sheets = bundle.get("review_contact_sheets")
    review = bundle.get("independent_review")
    if (
        not isinstance(visual, dict)
        or not isinstance(sheets, dict)
        or not isinstance(review, dict)
        or visual.get("count") != 198
        or visual.get("manifest_relpath") != "visual_qa/artifact_manifest.json"
        or sheets.get("count") != 11
        or sheets.get("manifest_relpath") != "review_contact_sheets/manifest.json"
        or review.get("attachment_count") != 19
        or review.get("attachment_manifest_relpath")
        != "review_attachment_manifest.json"
        or review.get("model") != "gpt-5.6-sol"
        or review.get("reasoning_effort") != "xhigh"
        or review.get("report_relpath") != "truebones_visual_review_gpt56sol.md"
        or review.get("report_sha256") != EXPECTED_POSTBUILD_VISUAL_REVIEW_SHA256
        or review.get("thread_id") != EXPECTED_POSTBUILD_VISUAL_REVIEW_THREAD_ID
        or review.get("verdict") != "pass"
    ):
        raise PrivateReleaseError("postbuild visual evidence contract drifted")

    visual_manifest_path = evidence / str(visual["manifest_relpath"])
    sheet_manifest_path = evidence / str(sheets["manifest_relpath"])
    attachment_manifest_path = evidence / str(review["attachment_manifest_relpath"])
    for record, path in (
        (visual, visual_manifest_path),
        (sheets, sheet_manifest_path),
        (review, attachment_manifest_path),
    ):
        if record.get("manifest_sha256", record.get("attachment_manifest_sha256")) != _sha256_regular_file(
            path, label=f"evidence manifest {path.name}"
        ):
            raise PrivateReleaseError(f"postbuild evidence manifest hash drifted: {path}")
    visual_files = _verify_manifest_files(
        evidence / "visual_qa", visual_manifest_path, expected_count=198
    )
    sheet_files = _verify_manifest_files(
        evidence / "review_contact_sheets", sheet_manifest_path, expected_count=11
    )
    attachment_files = _verify_manifest_files(
        evidence, attachment_manifest_path, expected_count=19
    )
    expected_attachments = {
        **{f"review_contact_sheets/{name}": value for name, value in sheet_files.items()},
        **{
            f"visual_qa/clips/{name}": visual_files[f"clips/{name}"]
            for name in _REVIEW_FILMSTRIP_ATTACHMENTS
        },
    }
    if attachment_files != expected_attachments:
        raise PrivateReleaseError("independent visual-review attachments drifted")
    return bundle


def prepare_private_distribution(
    source_generation: str | Path,
    output_parent: str | Path,
    *,
    postbuild_gate: str | Path,
    fixed_qa_report: str | Path,
    visual_generation: str | Path,
    visual_equivalence_report: str | Path,
    review_contact_sheets: str | Path,
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

        evidence_bundle = _prepare_postbuild_evidence_bundle(
            staging,
            release_gate=release_gate,
            gate_source=Path(postbuild_gate),
            fixed_qa_report=Path(fixed_qa_report),
            visual_generation=Path(visual_generation),
            visual_equivalence_report=Path(visual_equivalence_report),
            review_contact_sheets=Path(review_contact_sheets),
        )

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
            "postbuild_evidence_bundle_relpath": "evidence/bundle_manifest.json",
            "postbuild_evidence_bundle_sha256": evidence_bundle[
                "bundle_manifest_sha256"
            ],
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

        verify_full_generation_file_closure(staging, require_complete=True)
        _preflight_private_release_structure(staging)
        verify_full_generation(staging, require_complete=True)
        _assert_no_absolute_machine_paths(staging)
        _verify_postbuild_evidence_bundle(staging)
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
    verify_full_generation_file_closure(generation_root, require_complete=True)
    _preflight_private_release_structure(Path(root))
    generation = verify_full_generation(generation_root, require_complete=True)
    _assert_no_absolute_machine_paths(Path(root))
    _verify_postbuild_evidence_bundle(generation_root)
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
