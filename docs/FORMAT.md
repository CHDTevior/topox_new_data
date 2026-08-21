# KTJD-17 format contract

## Axes and rotations

- right-handed coordinates
- `Y+` up
- `XZ` ground plane
- canonical rest forward `+Z`
- active rotations acting on column vectors
- positive yaw around `+Y`
- column-cont6d order: `R00,R10,R20,R01,R11,R21`

The 6D decoder applies Gram-Schmidt with epsilon `1e-8` and constructs the
third basis vector as `cross(b1, b2)`.

## Motion payload

Each motion NPZ contains:

- `motion`: `float32 [T_valid,J_phys,17]`
- `heading_valid`: `bool [T_valid]`
- `origin_xz`: `float64 [2]`
- `clip_id`, `rig_id`: Unicode scalars
- `fps_target`: scalar, frozen to 30 for the validated release

Raw files are unnormalized and unpadded. The model view is constructed online
after crop translation, optional yaw augmentation, and normalization.

For every non-root physical joint, channels `13:17` are exact zero and excluded
by `channel_valid_mask`. Padded joints and frames are excluded from every loss
and statistic.

## PZ/Human statistics payloads

The private PZ-311 + Human-1 release includes raw population moments over all
accepted clips. `species_stats.npz` stores `[117,17]` biological-species
mean/std/count. `rig_stats.npz` stores padded `[312,J_max,17]` per-physical-rig,
per-joint mean/std/count plus `joint_count` and `valid_mask`. Counts, rather
than zero values, define validity. Non-root channels `13:17`, invalid heading
frames, and padded joints never enter a moment. Standard deviation uses
`ddof=0`; the stored arrays do not silently apply a training-specific floor.

## Skeleton payload

Each skeleton NPZ carries the physical hierarchy, canonical rest positions,
global and local rest rotations, parent-local offsets, animated/fixed rotation
provenance, heading carrier, forward direction, coordinate transform metadata,
and rig scale. No WORLD or control node is permitted.

The root is joint zero, `parents[0] = -1`, and every child obeys
`parents[child] < child`.

## Decode paths

Direct decode restores positions from `q_position` plus the root row's
smooth-XZ. Restoring `origin_xz` returns the clip's canonical absolute XZ.

Rotation-FK decodes global rest deltas against `R_rest_global`, uses direct
root translation, and propagates fixed parent-local offsets. Fixed-DOF joints
are reconstructed from their proven fixed transforms rather than free model
predictions.

The direct/FK difference is an explicit diagnostic. Neither path may silently
replace the other.

## Frozen validated release

- FPS: 30
- `J_max`: 142
- loader `T_max`: selected by the consumer; the current Graph-CodeFlow
  contract uses 300
- train-only gains: `[3.867547101351066, 2.943516881261983, 3.3212471860907744]`
- smooth root: fourth-order 1 Hz Butterworth with the documented short-clip
  linear fallback
- contact thresholds: height `0.05`, velocity `0.25`
- heading epsilon: `0.05`

Validation and held splits must not be used to refit these values.
