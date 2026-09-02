# ADR-0001：TP 上限由 NVSwitch 域决定；instance 数据契约

- 状态：**Amended ×2** —— 两字段建模先因移除「域 > 实例」机型而撤回，后因纳入「域 < 实例」机型（G 系 8 卡无 NVLink）而**恢复**。当前为两字段。
- 日期：2026-08-31，2026-08-31 修订
- 影响：核心计算模型、校验 banner 文案、instance 数据契约

## 背景

当前实现把「一个 p5en 实例 = 8 张卡 = TP 边界」写死在 `HW.gpusPerNode = 8` 里，并由此得出规则：

> TP ≤ 8，否则 all-reduce 走 EFA，93 层 × 2 次跨节点，不可接受。

问题不在这条规则的结论，而在它的**理由**被表述成了「节点」。约束的物理来源是 NVSwitch 连成的 scale-up 域，不是 EC2 实例边界。规则正确但理由错位，会在换硬件时给出错误推广。

## 驱动事实

即便在都是「8 卡 / 域 8」的 P 系机型内部，域也不只是一个大小：

| 实例 | GiB/卡 | NVLink 域 | 域内 P2P | 跨域 | 原生 FP4 |
|---|---|---|---|---|---|
| p5en.48xlarge | 141.0 | 8 | NVSwitch 900 GB/s | 3200 Gbps EFAv3 | ✗ Hopper |
| p6-b200.48xlarge | 179.1 | 8 | NVSwitch 1800 GB/s | 3200 Gbps EFAv4 | ✓ Blackwell |
| p6-b300.48xlarge | 268.6 | 8 | NVSwitch 1800 GB/s | 6400 Gbps EFAv4 | ✓ Blackwell Ultra |

域内带宽差 2×（900 → 1800 GB/s），跨域带宽差 2×（3200 → 6400 Gbps）。

而一旦纳入 G 系 8 卡机型，**域大小本身也不再等于每实例卡数**——见下节。完整候选清单见 ADR-0005 §2，全部 57 个实例的规格见 `docs/instance-specs.md`。显存单位一律 GiB（ADR-0006）。

## 决议

1. **TP 上限 = NVLink 域大小，不是每实例卡数。**
2. **两字段建模：`gpusPerInstance` 与 `nvlinkDomainGpus` 分开。** 二者在范围内确实会不等——见下节。
3. instance 契约字段：`id` / `gpusPerInstance` / `nvlinkDomainGpus` / `gpuMemMiB` / `p2p` / `interDomainGbps` / `efaGen` / `nativeDtypes`。显存以 MiB 为真值，见 ADR-0006。

### 「域 ≠ 实例」在当前范围内的真实反例

纳入 G 系 8 卡机型后（ADR-0005），出现了域**小于**实例的情形：

| 实例 | `gpusPerInstance` | `nvlinkDomainGpus` | `p2p` |
|---|---|---|---|
| p4d / p4de.24xlarge | 8 | **8** | NVSwitch 600 GB/s |
| p5 / p5e / p5en.48xlarge | 8 | **8** | NVSwitch 900 GB/s |
| p6-b200 / p6-b300.48xlarge | 8 | **8** | NVSwitch 1800 GB/s |
| g6e.48xlarge | 8 | **1** | PCIe only |
| g7.48xlarge | 8 | **1** | PCIe only |
| g7e.48xlarge | 8 | **1** | PCIe only |

G 系列一张 NVLink 都没有（g7/g7e 的 spec 页写「Yes via PCIe」，那是 PCIe P2P）。因此它们的 NVLink 域大小是 **1**：`TP > 1` 就已经在走 PCIe，而不是到 8 才出问题。

**这与最初撤回时的理由方向相反**：当时撤回是因为移除了域 > 实例的机型（72 卡域跨 18 实例）；现在恢复是因为纳入了域 < 实例的机型。两个方向都真实存在，单字段无论如何都不够。

## 修订记录

**第一次修订**：原 §1 曾以一类「NVLink 域横跨多个实例、域内远多于 8 卡」的机型作为反例，论证 TP32 在那类硬件上可完整落在 NVLink 内。该论证成立，但机型已移出范围，故两字段建模一度撤回为单字段。

**第二次修订（当前）**：纳入 G 系 8 卡机型后出现了「域 < 实例」的反例，两字段建模恢复。

**贯穿两次修订不变的认识**：`TP ≤ 8` 是 P 系 8 卡机型的性质，不是并行策略的普遍规律。在 G 系上正确的上限是 `TP ≤ 1`。这是把 banner 理由写成「NVLink 域」而非「节点」的全部原因。

## 后果

- `HW.gpusPerNode` 从硬编码常量改为读自 instance 定义；跨域判断为 `TP > nvlinkDomainGpus`，文案称「跨 NVLink 域」。
- **PCIe-only 机型（G 系）在 `TP > 1` 时必须告警**：卡间没有 NVLink，TP 的 all-reduce 走 PCIe，对 93 层 × 2 次的通信量是不可接受的。这是把「G 系 8 卡看着显存够但跑不了高 TP 模型」这个事实教给使用者的机制。
- 域内 P2P 带宽与跨域带宽进入 instance 定义，但**仅作展示**，不参与任何计算（工具只算显存，见 ADR-0003）。PCIe 机型的 P2P 带宽 AWS 未公开声明，字段留空而非填一个猜测值。
- `nativeDtypes` 进模型：Hopper 无原生 FP4/microscaling，Blackwell 有。同一份 MXFP4 权重在两者上占用**完全相同的字节数**，但在 p5en 上只省显存、拿不到算力加速（走 dequant 路径）。处理方式见 ADR-0005 §3。
- **对 Kimi K3 的直接影响**：2 × p6-b300 = 4288 GB，接近 4 × p5en 的 4512 GB。K3 从「4 台 p5en」变成「2 台 p6-b300」，跨域边界从 3 个降到 1 个，EP 从 32 降到 16。这会显著改写此前所有配置对比的结论。

## 未决

若将来引入「一个域横跨多个实例」或「每实例卡数 ≠ 域大小」的机型，`gpusPerInstance` 单字段必须拆分为两个，且节点卡片的视觉分组语言需要重做（要能表达「一个域 = 多个实例」）。届时本 ADR 应被 supersede，而非就地修改。
