# Data licensing boundary

The software license in this repository does not apply to input or generated
motion data.

The validated animal conversion uses motion originating from Truebones Zoo.
Truebones permits use of its products but explicitly prohibits redistribution
or resale of the Zoo contents in FBX, BVH, or i-Motion form. The public
T2M4LVO caption release likewise omits the motion files because of these
licensing restrictions.

Relevant upstream pages:

- [Truebones terms of service and use restrictions](https://truebones.gumroad.com/l/vlvPq)
- [Truebones Zoo redistribution notice](https://truebones.gumroad.com/p/the-truebones-fbx-bvh-zoo-only-from-truebones-now-with-free-financing)
- [T2M4LVO dataset card](https://huggingface.co/datasets/1Konny/t2m4lvo-truebones-zoo)

Consequently:

- no raw or derived Truebones motion is committed to this public GitHub repo;
- the validated KTJD-17 data is stored only in a private Hugging Face dataset;
- access must remain limited to authorized users who independently satisfy the
  upstream terms;
- recipients must not make the private dataset public or redistribute it;
- public examples must be synthetic or separately licensed.

The PlanetZoo/Human KTJD-17 batch is also kept in a separate private dataset by
project policy. Its public release contains processing code, documentation,
counts, and hashes only; it contains no motion, skeleton, statistics array, or
rendered data artifact. Source users remain responsible for their own
PlanetZoo/MotionStreamer access and terms.

This document records the release boundary; it is not a substitute for legal
advice or the upstream license text.
