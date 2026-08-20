---
pretty_name: KTJD-17 Truebones v1
license: other
task_categories:
  - other
---

# KTJD-17 Truebones v1

This is a private, access-controlled research snapshot derived from licensed
Truebones motion assets. It must remain private and must not be redistributed.
The MIT license in the companion code repository does not apply to this data.

## Frozen scope

- 986 accepted clips and 66 physical-joint rigs
- 84 upstream motion rejections and 4 unavailable rigs recorded explicitly
- 30 FPS, right-handed coordinates, `Y+` up, `XZ` ground, canonical `+Z`
  forward; perspective QA renders `+Z` toward the viewer
- raw motion payloads use unpadded `float32 [T_valid, J_phys, 17]`
- maximum observed `T=237`, maximum physical joints `J=142`

The motion rotations come from authoritative BVH rotation channels. Legacy
13D motion channels, position IK, and identity-filled animated leaves are not
used as rotation authority.

## Integrity and visual evidence

`RELEASE.json` selects one immutable generation. Its `generation.json` pins
every data and evidence file by SHA-256 and byte size. The snapshot includes:

- full 986/986 numerical and direct/FK QA;
- 198 synchronized GIF/filmstrip/rest artifacts covering all 66 rigs;
- 11 all-rig contact sheets and the exact 19-image independent-review input
  manifest;
- an independent `gpt-5.6-sol` review at `xhigh` effort with PASS verdict.

The public companion repository pins this private repository's immutable
40-hex revision, release pointer, generation digest, accepted-corpus identity,
and full source-scope identity. Use its downloader rather than a mutable branch
or a manual partial download.

## Authorized use

Authorized users should clone the companion code repository, authenticate with
Hugging Face, and run its repository-relative download command. The downloader
uses a fresh staging directory, verifies the complete snapshot and all 986
clips, then atomically publishes the local data directory. There is no command
line override for the trust root or remote revision.

Do not make this dataset public, mirror it, or share its derived motion,
skeleton, or visual payloads. Obtain the source license separately from the
official Truebones distributor.
