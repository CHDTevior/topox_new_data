"""Build and resolve host-sanitized private KTJD-17 distribution snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .codec import world_velocity
from .decoder import decode_ktjd17
from .encoder import load_skeleton
from .loader import load_motion_npz
from .truebones_full_build import verify_full_generation


PRIVATE_RELEASE_VERSION = "ktjd17-private-distribution-v1"
RELEASE_POINTER_NAME = "RELEASE.json"
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\\\]")


class PrivateReleaseError(RuntimeError):
    """Raised when a private distribution snapshot is unsafe or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> Path:
    """Resolve a user path and require it to stay inside the repository."""
    root = Path(repository_root).resolve()
    raw = Path(value)
    if raw.is_absolute() or raw != Path(str(raw).replace("\\", "/")):
        raise PrivateReleaseError(f"{argument_name} must be repository-relative")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PrivateReleaseError(
            f"{argument_name} must stay inside the repository"
        ) from exc
    return candidate


def _safe_pointer_subdir(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PrivateReleaseError("release pointer generation_subdir is missing")
    subdir = PurePosixPath(value)
    if subdir.is_absolute() or any(part in {"", ".", ".."} for part in subdir.parts):
        raise PrivateReleaseError("release pointer generation_subdir is unsafe")
    return subdir


def resolve_release_generation(
    snapshot_root: str | Path,
    *,
    require_pointer: bool = False,
) -> Path:
    """Resolve an immutable generation from a downloaded release snapshot."""
    snapshot = Path(snapshot_root).resolve()
    if (snapshot / "generation.json").is_file() and not require_pointer:
        return snapshot
    pointer_path = snapshot / RELEASE_POINTER_NAME
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PrivateReleaseError(f"cannot read {RELEASE_POINTER_NAME}: {exc}") from exc
    if not isinstance(pointer, dict):
        raise PrivateReleaseError("release pointer root must be an object")
    subdir = _safe_pointer_subdir(pointer.get("generation_subdir"))
    generation = (snapshot / Path(*subdir.parts)).resolve()
    try:
        generation.relative_to(snapshot)
    except ValueError as exc:
        raise PrivateReleaseError("release pointer escapes the snapshot") from exc
    generation_json = generation / "generation.json"
    if not generation_json.is_file():
        raise PrivateReleaseError("release pointer target has no generation.json")
    expected_sha = pointer.get("generation_json_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise PrivateReleaseError("release pointer generation hash is invalid")
    observed_sha = _sha256_file(generation_json)
    if observed_sha != expected_sha:
        raise PrivateReleaseError("release pointer generation hash mismatch")
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
    resolved = resolve_release_generation(root, require_pointer=True)
    if resolved != generation:
        raise PrivateReleaseError("written release pointer did not resolve exactly")
    return target


def _looks_absolute_path(value: str) -> bool:
    return value.startswith("/") or _WINDOWS_ABSOLUTE.match(value) is not None


def _sanitized_path_label(value: str) -> str:
    normalized = value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if "Truebone_Z-OO" in parts:
        index = parts.index("Truebone_Z-OO") + 1
        tail = parts[index:]
        if tail:
            return PurePosixPath("sources", "truebones", *tail).as_posix()
    filename = parts[-1] if parts else "source"
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
            tail = parts[index:] or [filename]
            return PurePosixPath("provenance", label, *tail).as_posix()
    opaque = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return PurePosixPath("sources", "redacted", opaque, filename).as_posix()


def _sanitize_value(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        if _looks_absolute_path(value):
            return _sanitized_path_label(value)
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
            elif _looks_absolute_path(text):
                replacement = _sanitized_path_label(text)
            replacement_text = str(replacement)
            rewritten.append(replacement_text)
            changed = changed or replacement_text != text
        if changed:
            width = max(1, max(len(item) for item in rewritten))
            arrays[key] = np.asarray(rewritten, dtype=f"<U{width}").reshape(array.shape)
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
        if path.is_symlink():
            raise PrivateReleaseError(f"symlink in private release: {path}")
        if path.is_file() and path.name != "generation.json":
            relpath = path.relative_to(root).as_posix()
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and _looks_absolute_path(value)


def _assert_no_absolute_machine_paths(root: Path) -> None:
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
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
                        if _contains_absolute_path(decoded):
                            violations.append(f"{path.relative_to(root)}:{key}")
                            break
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
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
                if _contains_absolute_path(values):
                    violations.append(path.relative_to(root).as_posix())
            elif re.search(r'(?<![A-Za-z0-9:])/(?:home|scratch|iridisfs)/', text):
                violations.append(path.relative_to(root).as_posix())
    if violations:
        raise PrivateReleaseError(
            f"host paths remain in private release: {violations[:10]}"
        )


def prepare_private_distribution(
    source_generation: str | Path,
    output_parent: str | Path,
) -> dict[str, object]:
    """Copy, sanitize, re-hash, and verify an immutable generation for private hosting."""
    source = Path(source_generation).resolve()
    verify_full_generation(source, require_complete=True)
    destination_parent = Path(output_parent).resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / source.name
    if destination.exists() or destination.is_symlink():
        raise PrivateReleaseError(f"private release already exists: {destination}")
    source_generation_sha = _sha256_file(source / "generation.json")
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
        }
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
        }
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)


def validate_private_distribution(root: str | Path) -> dict[str, object]:
    """Run self-contained integrity and decode QA without proprietary sources."""
    generation_root = resolve_release_generation(root)
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
    if len(manifests) != 986 or len(qa_records) != len(manifests):
        raise PrivateReleaseError("distribution QA scope is not 986 accepted clips")
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
