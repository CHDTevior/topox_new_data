# KTJD-17 normative design

This file is the repository-local implementation contract for the KTJD-17
multi-topology representation. It mirrors the reviewed TopoX handoff frozen at
commit `9181f5cccbad23e941bf94c2874daf36e7f288cf` so all implementation links and
runtime paths remain repository-relative.

## Tensor and coordinate contract

`motion.shape = [T, J_max, 17]`, where `T` is time and `J_max` contains physical
joints only. A WORLD/control pseudo-joint is forbidden.

The shared coordinate standard is:

- right-handed coordinates;
- `Y+` up and `XZ` the ground plane;
- canonical rest forward is `+Z`;
- in perspective QA, `Y+` is screen-up and `+Z` points out of the screen toward
  the viewer.

Each rig retains its own canonical rest pose. Animals are not forced into a
human T-pose.

## Channels

| Channels | Contract |
| --- | --- |
| `0:3` | `q_position = [Px-sx, Py, Pz-sz]`, relative to smooth-root XZ while preserving world axes |
| `3:9` | global rest-delta column-cont6d for `R_global @ R_rest_global.T` |
| `9:12` | world-space joint velocity in length-unit/second |
| `12` | per-joint ground-support contact |
| `13:15` | smooth-root world XZ, valid only on the physical root row |
| `15:17` | heading `[cos(theta), sin(theta)]`, valid only on the physical root row |

Channels `13:17` are exact zero for non-root joints and are excluded by
`channel_valid_mask` from loss and statistics.

## Two independent reconstruction paths

Position-direct recovery is frame-local and never integrates velocity:

```text
Pj = [qj.x + smooth_x, qj.y, qj.z + smooth_z]
```

Rotation-FK recovery is independent:

```text
R_global[j] = decode(d6[j]) @ R_rest_global[j]
P_fk[c] = P_fk[parent] + R_global[parent] @ offset[c]
```

The direct/FK difference is a required diagnostic. Neither path may silently
replace or manufacture the other.

## Rotation and contact authority

Rotations must come from the real BVH, SMPL, or MotionStreamer rotation source.
An old 13D representation, position IK, or identity-filled animated leaf joint
is not acceptable authority. Fixed-DOF joints may use only transforms proven by
the source hierarchy/rest contract.

Contact is recomputed after canonical FPS, ground, and scale are fixed. It is
not copied from a source-specific legacy channel. The validated Truebones v1
release uses 30 FPS; other source families must pass their own audit before
freezing FPS, smoothing, scale, and contact thresholds.

## Required gates

Before full conversion, one prototype from each topology class must pass both
numeric and dynamic perspective review: human, quadruped, winged, snake,
spider/crab-like, and dragon/dinosaur. Full releases additionally require:

1. source-identity and real-rotation authority closure;
2. per-file immutable hashes and manifest/split closure;
3. full-corpus numeric, direct/FK, rigid-edge, velocity, heading, and mask QA;
4. one dynamic perspective artifact per converted rig, with source,
   position-direct, and rotation-FK shown synchronously;
5. a hostile independent review using the currently mandated reviewer model;
6. private-host sanitization and remote immutable-revision round-trip where
   source licensing forbids public redistribution.
