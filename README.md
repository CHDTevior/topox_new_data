# TopoX KTJD-17 data tools

This repository contains the conversion, validation, loading, dual-decoding,
and visual-QA tools used to build a lossless KTJD-17 representation from
multi-topology motion sources.

The implementation follows the repository-local
[KTJD-17 design](docs/KTJD17_DESIGN.md), transcribed from the reviewed TopoX
handoff at commit `9181f5cccbad23e941bf94c2874daf36e7f288cf`.
The validated implementation snapshot came from source commit
`7a691cab858a2aebdedb4a4f192aac5d50bdd178`.

## Representation

Raw motion files are unpadded `float32 [T_valid, J_phys, 17]` arrays. Only
physical joints are stored; there is no WORLD node.

| Channels | Meaning |
| --- | --- |
| `0:3` | `q_position = [Px-sx, Py, Pz-sz]` |
| `3:9` | global rest-delta rotation in column-cont6d |
| `9:12` | canonical-world joint velocity in length-unit/s |
| `12` | per-joint ground-support contact |
| `13:15` | smooth-root world XZ, root row only |
| `15:17` | heading `[cos(theta), sin(theta)]`, root row only |

The coordinate convention is right-handed, `Y+` up, `XZ` ground, and
canonical forward `+Z`. In the perspective QA renderer, `Y+` is screen-up and
`+Z` points out of the screen toward the viewer.

Positions are recovered independently at every frame:

```text
Pj = [qj.x + smooth_x, qj.y, qj.z + smooth_z]
```

The independent rotation-FK path is:

```text
R_global[j] = decode(d6[j]) @ R_rest_global[j]
P_fk[c] = P_fk[parent] + R_global[parent] @ offset[c]
```

See [docs/FORMAT.md](docs/FORMAT.md) for the complete storage and mask
contract, and [docs/guide_zh.md](docs/guide_zh.md) for the Chinese guide.

## Install and test

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -q
```

The release tree includes data-independent tests plus fixture-bound regression
tests. A clean checkout runs the data-independent subset and explicitly skips
assertions tied to private immutable generations. The full source environment
passes all 128 tests (with private-fixture tests skipped in a clean checkout).
Tests do
not download or bundle third-party motion data.

## Download the private validated dataset

The validated 986-clip Truebones-derived release is private because the
upstream Truebones terms prohibit redistribution. Authorized users can log in
to Hugging Face and download it into a repository-relative directory:

```bash
hf auth login
python scripts/download_private_dataset.py \
  --local-dir data/ktjd17_truebones
```

Then run the self-contained distribution QA:

```bash
python scripts/validate_ktjd17_truebones.py \
  --dataset-root data/ktjd17_truebones \
  --output outputs/ktjd17_truebones_distribution_qa.json
```

The downloader reads [release/truebones_v1.json](release/truebones_v1.json),
which pins the private repository, immutable remote revision, corpus identity,
release pointer, and `generation.json`. It rejects absolute paths, `..`
escapes, symlink escapes, existing destinations, special files, hard links,
and unexpected snapshot-root entries. Download happens in a fresh staging
directory and is atomically installed only after checking the release pointer,
the pinned `generation.json` digest, every immutable file hash/size, all
manifest-to-NPZ references, split closure, and the 986 accepted clips. The validator accepts
the snapshot root shown above and resolves its versioned generation safely.
Distribution QA decodes all 986 clips and checks the direct/FK agreement,
rigid edges, velocities, headings, contacts, and root-only channels without
requiring proprietary BVH files. Data owners can additionally use
`--source-backed` inside the complete build workspace; that mode intentionally
requires the original source files and parent manifests.

The published Truebones v1 scope is deliberately exact: the upstream catalog
names 70 rigs, 66 have usable authoritative BVH rotations, and 4 are unavailable
(`Ant`, `Crab`, `Deer`, `Jaguar`). The release contains 986 accepted clips;
84 other catalog motions were rejected upstream before conversion. It does not
yet contain the separate Planet Zoo 311-rig or Human 1-rig batches.

The download helper reads the token from the Hugging Face credential store.
Do not put tokens in commands, source files, or configuration committed to Git.

## Read and decode one clip

```python
from pathlib import Path
import json
import numpy as np

from src.data.ktjd17.codec import restore_origin_xz
from src.data.ktjd17.decoder import decode_ktjd17
from src.data.ktjd17.encoder import load_skeleton
from src.data.ktjd17.loader import load_motion_npz
from src.data.ktjd17.private_release import (
    load_trusted_release,
    resolve_release_generation,
)

trust = load_trusted_release(Path("release/truebones_v1.json"))
root = resolve_release_generation(
    Path("data/ktjd17_truebones"), trusted_release=trust
)
rows = [
    json.loads(line)
    for line in (root / "manifests/clips.jsonl").read_text().splitlines()
]
row = rows[0]
payload = load_motion_npz(root / row["motion_relpath"], expected_fps_target=30.0)
skeleton = load_skeleton(root / row["skeleton_relpath"])

decoded = decode_ktjd17(
    payload["motion"].astype(np.float64),
    parents=skeleton.parents,
    R_rest_global=skeleton.R_rest_global,
    R_rest_local=skeleton.R_rest_local,
    offset_parent_local=skeleton.offset_parent_local,
    rotation_source_kind=skeleton.rotation_source_kind,
    strict_gt=True,
)

positions = restore_origin_xz(decoded.positions_direct, payload["origin_xz"])
positions_fk = restore_origin_xz(decoded.positions_fk, payload["origin_xz"])
print(payload["clip_id"], positions.shape)
print("direct/FK max:", np.linalg.norm(positions - positions_fk, axis=-1).max())
```

Never integrate channels `9:12` to recover position. Velocity is a supervision
and diagnostic channel; direct position recovery has no temporal dependency.

## Rebuild pipeline

The full pipeline is deliberately gated. It inventories source provenance,
reproduces source FK, derives canonical skeletons, builds and visually checks
six topology prototypes, freezes train-only calibration, audits every
Truebones rig, and only then permits the full build.

All paths are passed as repository-relative paths. A typical source layout is:

```text
data/
├── current_btjd/
│   ├── cond.npy
│   └── motions/
├── legacy_truebones_btjd/
│   └── cond.npy
├── truebones_raw/
│   └── <rig>/<motion>.bvh
├── holdout_splits_v1/
└── optional_additional_sources/
```

Start with explicit paths rather than relying on project-specific defaults:

```bash
python scripts/write_ktjd17_schema.py --output dataset/schema.json

python scripts/inventory_ktjd17_sources.py \
  --dataset-root data/current_btjd \
  --split-root data/holdout_splits_v1 \
  --truebones-raw-root data/truebones_raw \
  --pz-bvh-root data/optional_additional_sources/planetzoo_bvhs \
  --human272-root data/optional_additional_sources/human272 \
  --output-root dataset/manifests
```

Continue in the order encoded by the scripts:

1. `audit_ktjd17_source_fk.py`
2. `build_ktjd17_canonical_skeletons.py`
3. `build_ktjd17_prototype.py`
4. `validate_ktjd17_prototype.py`
5. `calibrate_ktjd17_prototype.py`
6. `render_ktjd17_visual_qa.py`
7. `freeze_ktjd17_schema.py`
8. `build_ktjd17_truebones_forward_audit.py`
9. `render_ktjd17_truebones_forward_audit.py`
10. `build_ktjd17_truebones.py`
11. `validate_ktjd17_truebones.py`
12. `render_ktjd17_truebones.py`

Each later command must point to the immutable generation emitted by its
predecessor. Do not bypass the prototype, visual, or freeze gates.

Before uploading a complete generation to a private data host, create a
host-sanitized distribution copy. This preserves all motion payload bytes and
source hashes, replaces machine-local provenance paths with stable relative
labels, re-hashes affected skeletons and references, rebuilds the immutable
file closure, verifies it again, and writes the hash-pinned `RELEASE.json`
needed by the downloader:

```bash
python scripts/prepare_private_dataset_release.py \
  --source-generation dataset/ktjd17_truebones \
  --output-parent dataset/private_release \
  --postbuild-gate release/evidence/truebones_postbuild_release_gate.json
```

The postbuild gate must pin a 986/986 fixed-QA pass and a 66/66-rig dynamic
perspective review performed with `gpt-5.6-sol` at `xhigh` reasoning. The
release copy is marked training-ready only after those checks and byte-level
motion / elementwise-numeric skeleton equivalence succeed.

## Data boundary

This GitHub repository contains code and documentation only. It does not grant
rights to Truebones, BTJD, MotionStreamer, PlanetZoo, SMPL, or any other input
data. Obtain each source from its official distributor and follow its terms.
See [docs/DATA_LICENSE.md](docs/DATA_LICENSE.md).

Before a public commit, run:

```bash
bash scripts/check_public_release.sh
```

## License

The code in this repository is released under the MIT License. Third-party
data and dependencies retain their own licenses.
