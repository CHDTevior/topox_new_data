# PZ-311 + Human-1 转换与使用

这一批次把 PlanetZoo 的 311 套物理 rig 和 MotionStreamer272/HumanML3D 的
1 套 human rig 转成同一个 KTJD-17 合同。源范围为 101,368 个 clip：
PlanetZoo 74,522 个、Human 26,846 个。最终 accepted/rejected 数量以
`generation.json` 和 private release 的 `RELEASE.json` 为准。

PlanetZoo rotation 直接来自 stage-2 BVH 声明的真实 Euler rotation channels；
Human rotation 直接来自 MotionStreamer272 的 `140:272` rotation channels。
转换没有使用旧 BTJD-13 rotation、位置 IK 或 leaf identity 伪造主数据。

## 下载 private 数据

仓库中的 trust record 固定 private Hugging Face dataset 和不可变 revision：

```bash
hf auth login
python scripts/download_private_pz_human312.py \
  --local-dir data/ktjd17_pz_human312
```

下载器校验 `RELEASE.json` 和所有 tar shard 后，输出实际的相对路径：

```text
data/ktjd17_pz_human312/dataset/<generation_id>
data/ktjd17_pz_human312/species_stats
```

数据仓库必须保持 private。GitHub 只包含处理代码、格式说明和不可逆的哈希/计数，
不包含 motion、skeleton、统计数组或可视化数据文件。

## mean/std

一次全量扫描会生成两层 population moments（`ddof=0`）：

- `species_stats.npz`：117 个生物物种的 `[117,17]` mean/std/count；
  PlanetZoo 的 Female/Male/Juvenile 合并到同一个生物物种，Human 单独成组。
- `rig_stats.npz`：312 套物理 rig 的 padded
  `[312,J_max,17]` mean/std/count，另带 `joint_count`、`valid_mask` 和每个 rig
  对应的生物物种。不同性别/年龄骨架不强行建立关节对应。

两个层次都使用所有 accepted train/val/test clip 的原始、未 padding、未
normalization `float32` KTJD-17 值。`0:13` 使用所有物理关节；`13:15` 只统计
root；`15:17` 只统计 `heading_valid` 的 root frame。无效非 root 通道、无效
heading 和 padded joint 的 count 为零，不进入 mean/std。

```python
from pathlib import Path
import numpy as np

stats_root = Path("data/ktjd17_pz_human312/species_stats")

with np.load(stats_root / "species_stats.npz", allow_pickle=False) as z:
    species_ids = z["species_ids"].astype(str)
    species_mean = z["mean"]       # [117, 17]
    species_std = z["std"]         # [117, 17]
    species_count = z["count"]     # [117, 17]

with np.load(stats_root / "rig_stats.npz", allow_pickle=False) as z:
    rig_ids = z["rig_ids"].astype(str)
    rig_mean = z["mean"]           # [312, J_max, 17]
    rig_std = z["std"]             # [312, J_max, 17]
    rig_count = z["count"]         # count > 0 is a valid cell
    joint_count = z["joint_count"]
```

这里保存的 std 是原始经验标准差；训练代码若需要防止除零，应显式选择自己的
floor，不要把 padded/无效通道的零 sentinel 当成统计值。

## 重建与可视化检查

所有示例路径都是仓库相对路径。先分别冻结两类源审计，再构建每 rig 一个动态
prototype：

```bash
python scripts/audit_ktjd17_pz312_sources.py \
  --manifest-root dataset/manifests \
  --pz-bvh-root data/animo4d_anytop/bvhs \
  --active-cond data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy \
  --output-root dataset

python scripts/audit_ktjd17_human312_sources.py --repo-root .

python scripts/build_ktjd17_pz_human312.py \
  --mode prototype \
  --dataset-root dataset \
  --freeze-root dataset/ktjd17_freeze \
  --output-root dataset
```

现有透视渲染器直接读取 prototype 的标准 manifest、motion 和 skeleton，不需要
adapter：

```bash
python scripts/render_ktjd17_pz_human312.py \
  --prototype-root dataset/ktjd17_pz_human312_prototype \
  --output-root dataset/ktjd17_pz_human312_visual

python scripts/validate_ktjd17_pz_human312_visuals.py \
  dataset/ktjd17_pz_human312_visual/ktjd17_visual_qa \
  --report outputs/ktjd17_pz_human312_visual_qa.json

python scripts/build_ktjd17_pz_human312_review_sheets.py \
  dataset/ktjd17_pz_human312_visual/ktjd17_visual_qa \
  --output scratch/ktjd17_pz_human312_review_sheets
```

每个 GIF/filmstrip 固定并排显示 source、position-direct、rotation-FK 三条路径；
相机跨帧/路径固定，`Y+` 向屏幕上方，`+Z` 向屏幕外。人工检查与独立审核通过后，
用 `write_ktjd17_pz_human312_visual_gate.py` 写出 gate，再启动全量转换：

```bash
python scripts/build_ktjd17_pz_human312.py \
  --mode full \
  --workers 24 \
  --source-rehash-workers 1 \
  --visual-gate dataset/ktjd17_pz_human312_visual_gate.json

python scripts/compute_ktjd17_species_stats.py \
  --generation dataset/ktjd17_pz_human312 \
  --output dataset/ktjd17_pz_human312_species_stats \
  --workers 16
```

若存在真实异常 clip，只有显式、逐条记录原因的 anomaly allowlist 才会筛除；未传
allowlist 就要求零 reject。

## private tar 分片

全量生成包含十万多个小 NPZ，因此 private HF 发布使用确定性、未重复压缩的 tar
shard，而不是逐个上传小文件：

```bash
python scripts/package_ktjd17_pz_human312_private.py \
  --generation dataset/ktjd17_pz_human312 \
  --species-stats dataset/ktjd17_pz_human312_species_stats \
  --output dataset/ktjd17_pz_human312_private_release \
  --max-shard-mib 512
```

打包时同步核对 generation 中记录的每个源文件 SHA-256；下载时核对每个 shard
SHA-256、generation/stats identity 和安全相对 member，再逐文件解包。解包不调用
`tarfile.extract`，不接受绝对路径、`..`、重复 member、软/硬链接或特殊文件。
