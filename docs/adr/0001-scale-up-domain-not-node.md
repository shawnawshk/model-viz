# ADR-0001：TP 上限由 NVLink 域决定；instance 数据契约

- 状态：**Amended ×2**。当前使用两字段建模。
  - 第一次修订：范围内移除了「域 > 实例」的机型，两字段建模撤回为单字段。
  - 第二次修订：范围内加入了「域 < 实例」的机型（G 系 8 卡，无 NVLink），两字段建模**恢复**。
- 日期：2026-08-31，2026-08-31 修订
- 影响：核心计算模型、校验 banner 文案、instance 数据契约

## 背景

当前实现在 `HW.gpusPerNode = 8` 里写死了一个假设：一个 p5en 实例 = 8 张卡 = TP 边界。实现从这个假设得出以下规则：

> TP ≤ 8，否则 all-reduce 走 EFA，93 层 × 2 次跨节点，不可接受。

这条规则的结论正确。错误在规则的**理由**：理由写成了「节点」。

约束的物理来源是 NVLink 域，即 NVSwitch 连成的 scale-up 域。约束的来源不是 EC2 实例的边界。

如果理由写错，换硬件时就会把规则推广到错误的地方。

## 驱动事实

P 系机型都是「8 卡 / 域 8」。但即使在 P 系内部，各个 NVLink 域的带宽也不同：

| 实例 | GiB/卡 | NVLink 域 | 域内 P2P | 跨域 | 原生 FP4 |
|---|---|---|---|---|---|
| p5en.48xlarge | 141.0 | 8 | NVSwitch 900 GB/s | 3200 Gbps EFAv3 | ✗ Hopper |
| p6-b200.48xlarge | 179.1 | 8 | NVSwitch 1800 GB/s | 3200 Gbps EFAv4 | ✓ Blackwell |
| p6-b300.48xlarge | 268.6 | 8 | NVSwitch 1800 GB/s | 6400 Gbps EFAv4 | ✓ Blackwell Ultra |

- 域内带宽相差 2×（900 → 1800 GB/s）。
- 跨域带宽相差 2×（3200 → 6400 Gbps）。

范围内加入 G 系 8 卡机型后，**域大小也不再等于每实例卡数**。下一节给出这些机型。

- 完整候选清单见 ADR-0005 §2。
- 全部 57 个实例的规格见 `docs/instance-specs.md`。
- 显存单位一律用 GiB（ADR-0006）。

## 决议

1. **TP 上限 = NVLink 域大小。TP 上限不是每实例卡数。**
2. **使用两字段建模：`gpusPerInstance` 与 `nvlinkDomainGpus` 是两个字段。** 在当前范围内，这两个值会不相等。下一节给出这些机型。
3. instance 契约的字段：`id` / `gpusPerInstance` / `nvlinkDomainGpus` / `gpuMemMiB` / `p2p` / `interDomainGbps` / `efaGen` / `nativeDtypes`。显存的真值用 MiB，见 ADR-0006。

### 当前范围内「域 ≠ 实例」的机型

ADR-0005 把 G 系 8 卡机型加入范围。这些机型的域**小于**实例：

| 实例 | `gpusPerInstance` | `nvlinkDomainGpus` | `p2p` |
|---|---|---|---|
| p4d / p4de.24xlarge | 8 | **8** | NVSwitch 600 GB/s |
| p5 / p5e / p5en.48xlarge | 8 | **8** | NVSwitch 900 GB/s |
| p6-b200 / p6-b300.48xlarge | 8 | **8** | NVSwitch 1800 GB/s |
| g6e.48xlarge | 8 | **1** | PCIe only |
| g7.48xlarge | 8 | **1** | PCIe only |
| g7e.48xlarge | 8 | **1** | PCIe only |

G 系机型没有 NVLink。g7/g7e 的 spec 页写的是「Yes via PCIe」，意思是 PCIe P2P，不是 NVLink。

所以 G 系机型的 NVLink 域大小是 **1**。`TP > 1` 时，通信就已经走 PCIe。问题从 TP = 2 开始，不是从 TP = 8 开始。

**第二次修订的理由与第一次修订的理由方向相反：**

- 第一次修订：范围内移除了域 > 实例的机型（72 卡的域跨 18 个实例），所以撤回两字段。
- 第二次修订：范围内加入了域 < 实例的机型，所以恢复两字段。

这两个方向都真实存在。所以单字段在两个方向上都不够。

## 修订记录

**第一次修订**：原 §1 用一类机型作反例。这类机型的一个 NVLink 域跨多个实例，域内的卡数远多于 8 张。在这类硬件上，TP32 可以完整放在 NVLink 域内。这个论证正确。但这类机型已移出范围，所以两字段建模暂时撤回为单字段。

**第二次修订（当前）**：范围内加入 G 系 8 卡机型后，出现了「域 < 实例」的反例。两字段建模恢复。

**两次修订都不变的结论**：

- `TP ≤ 8` 是 P 系 8 卡机型的性质。`TP ≤ 8` 不是并行策略的通用规则。
- 在 G 系机型上，正确的上限是 `TP ≤ 1`。
- 所以 banner 的理由写「NVLink 域」，不写「节点」。

## 后果

- `HW.gpusPerNode` 不再是硬编码常量。引擎从 instance 定义读取这个值。
  - 跨域判断：`TP > nvlinkDomainGpus`。
  - 文案写「跨 NVLink 域」。
- **在 PCIe-only 机型（G 系）上，`TP > 1` 时必须告警。** 原因：
  - 这些机型的卡间没有 NVLink，所以 TP 的 all-reduce 走 PCIe。
  - 93 层 × 2 次的通信量走 PCIe，不可接受。
  - G 系 8 卡机型的显存看起来够用，但跑不了需要高 TP 的模型。这条告警把这个事实告诉使用者。
- instance 定义包含域内 P2P 带宽与跨域带宽。这两个值**只用于展示**，不用于计算。原因：工具只算显存，见 ADR-0003。
  - AWS 没有公开 PCIe 机型的 P2P 带宽。这个字段留空，不填猜测值。
- 模型使用 `nativeDtypes`。Hopper 没有原生 FP4/microscaling，Blackwell 有。
  - 同一份 MXFP4 权重在 Hopper 与 Blackwell 上占用的字节数**完全相同**。
  - 在 p5en 上，MXFP4 只省显存，不提供算力加速，因为计算走 dequant 路径。
  - 处理方法见 ADR-0005 §3。
- **对 Kimi K3 的直接影响**：
  - 2 × p6-b300 = 4297.5 GiB。4 × p5en = 4512 GiB。两者接近。
  - K3 的部署从「4 台 p5en」变为「2 台 p6-b300」。
  - 跨域边界从 3 个减少到 1 个。
  - EP 从 32 减少到 16。
  - 所以此前所有配置对比的结论都会大幅改变。

## 未决

将来可能加入一个域跨多个实例的机型。

两字段建模已经可以表示这个关系（`nvlinkDomainGpus > gpusPerInstance`）。但节点卡片的视觉分组必须重新设计，使它能表示「一个域 = 多个实例」。

到那时，用一个新 ADR supersede 本 ADR。不要直接修改本 ADR。
