# KTJD-17 数据处理使用说明

## 坐标和格式

固定约定为右手系、`Y+` 正上方、`XZ` 地面、canonical rest forward 为
`+Z`。透视可视化中 `Y+` 朝屏幕上方，`+Z` 朝屏幕外、指向观察者。

motion 文件是未 padding、未 normalization 的
`float32 [T_valid,J_phys,17]`。通道定义见 [FORMAT.md](FORMAT.md)。非 root
关节的 `13:17` 必须是精确零，并由 `channel_valid_mask` 排除。

## 下载 private 数据

先完成 Hugging Face 登录，然后下载到仓库内的相对路径：

```bash
hf auth login
python scripts/download_private_dataset.py \
  --local-dir data/ktjd17_truebones
```

运行无需原始 BVH 的 distribution QA：

```bash
python scripts/validate_ktjd17_truebones.py \
  --dataset-root data/ktjd17_truebones \
  --output outputs/ktjd17_truebones_distribution_qa.json
```

下载脚本从 [release/truebones_v1.json](../release/truebones_v1.json) 读取
private 仓库、不可变远端 revision、语料身份、release pointer 与 generation
摘要。trust record 路径和 revision 都不能由命令行覆盖；若 revision 尚为
`null`，脚本会在任何网络调用前失败。它会拒绝绝对路径、`..`、内部或未识别的软链接逃逸、
已存在的目标、特殊文件、硬链接、危险文件 mode、NPZ 隐藏/越界成员和
snapshot 根目录额外条目；先下载到临时 staging，并以禁止覆盖的方式发布，
只有在 release pointer、
`generation.json` 摘要、全文件哈希/大小闭包、manifest-to-NPZ 引用、split
闭包和 986 条 accepted clip 均通过后，才会安装并返回成功。验证脚本可以直接接收
上面的 snapshot 根目录，并安全解析其中的版本化 generation。
distribution QA 会逐条解码全部 986 个 clip，并检查 direct/FK、骨长刚性、
速度、heading、contact 与 root-only channels。数据所有者若在完整构建工作区
内还要复跑依赖原始 BVH 的 fixed QA，可显式添加 `--source-backed`；下载后的
独立数据目录不需要、也不包含那些上游文件。

在不支持 `renameat2(RENAME_NOREPLACE)` 的文件系统（包括已测试的 GPFS）上，
下载器会用一次原子的、禁止覆盖的相对软链接创建来发布已经完整验证的同级 payload。
因此请求的 `--local-dir`（例如 `data/ktjd17_truebones`）会显示为软链接，其隐藏同级
目标遵循固定命名 `.ktjd17_truebones.payload-*`。所有文档中的数据路径都可直接通过
这个别名使用；验证器只允许这一种受限的顶层别名，release 内部软链接仍一律拒绝。
payload 不会被复制、半成品暴露或覆盖。若已经进入发布阶段后失败，并发目标会原样
保留，已验证的隐藏 payload 也会保留供检查；重试前人工删除精确的 `.payload-*`
同级目录。进入发布前的传输或验证失败仍会自动清理私有半成品。

数据仓库必须保持 private。Truebones 条款禁止重新分发；不要把下载后的
motion、skeleton 或可逆派生表示上传到 public GitHub/Hugging Face。

Truebones v1 的范围是明确冻结的：上游目录共有 70 个 rig，其中 66 个具备可用
的真实 BVH 旋转源；`Ant`、`Crab`、`Deer`、`Jaguar` 四个不可用。最终包含
986 个 accepted clip，另有 84 个上游阶段 rejected motion。本版本还不包含
单独的 PZ 311 个 rig 与 Human 1 个 rig；它们属于后续独立来源批次。

## 读取和训练视图

最小读取与双路径解码示例见根目录 [README.md](../README.md)。训练时通过
`build_model_view` 在线完成 crop、yaw、train-only gains normalization 和
padding。当前 Graph-CodeFlow 合同是：

- `T_fine_max=300`
- `temporal_stride=4`
- `T_lat_max=75`

不要退回固定 64 帧或 16 个 latent steps。

## 重建顺序

完整重建必须依次通过：源库存、source-FK、canonical skeleton、六类
prototype、数值 QA、动态透视 QA、train-only calibration freeze、全 rig
forward audit、全量转换、全量 fixed QA、后构建动态可视化。

所有输入输出都用仓库相对路径，例如 `data/`、`dataset/` 和 `outputs/`；
不要把服务器绝对路径、主机名或凭证写进配置、说明或提交历史。

发布前运行：

```bash
bash scripts/check_public_release.sh
```

若需要由数据所有者更新 private Hugging Face 数据，先生成不含服务器路径的
发布副本：

```bash
python scripts/prepare_private_dataset_release.py \
  --source-generation dataset/ktjd17_truebones \
  --output-parent dataset/private_release \
  --postbuild-gate release/evidence/truebones_postbuild_release_gate.json \
  --fixed-qa-report dataset/validation_reports/truebones_fixed_qa.json \
  --visual-generation dataset/visual_qa_generation \
  --visual-equivalence-report dataset/visual_equivalence.json \
  --review-contact-sheets scratch/visual_review_sheets
```

该步骤保持 motion payload 字节不变，将机器本地 provenance 路径改为稳定
相对标签，重算受影响的 skeleton/reference 哈希，再次执行完整闭包验证，
并生成下载器强制校验的 hash-pinned `RELEASE.json`。postbuild gate 必须固定
986/986 数值 QA、66/66 rig 动态透视 QA，以及 `gpt-5.6-sol / xhigh` 独立
审查 PASS，才允许标记为 training-ready。private snapshot 还会纳入脱敏后的
完整 fixed-QA、198 个 GIF/filmstrip/rest 视觉产物、11 张 review sheet 和
审查实际使用的 19 张图片清单；这些文件不进入 public Git，但每次 distribution
QA 都会按传递哈希闭包复核。
