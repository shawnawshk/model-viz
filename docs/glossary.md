# Glossary

这套词汇是这个工具的领域模型。之前几轮讨论里出现过的每一次误解，根源都是下面某一条没有被区分开。

## 拓扑

### NVLink 域（scale-up 域）
一组通过 NVLink + NVSwitch 直连、彼此间带宽在数百 GB/s 量级的 GPU。**这是决定 TP 上限的边界，与实例大小无关。**

| 实例 | 卡/实例 | **NVLink 域** | 域内 P2P |
|---|---|---|---|
| p4d / p4de.24xlarge | 8 | 8 | NVSwitch 600 GB/s |
| p5 / p5e / p5en.48xlarge | 8 | 8 | NVSwitch 900 GB/s |
| p6-b200 / p6-b300.48xlarge | 8 | 8 | NVSwitch 1800 GB/s |
| g6e / g7 / g7e.48xlarge | 8 | **1** | 无 NVLink，仅 PCIe |

**NVLink 是 P 系列独占。** G 系列一张都没有——g7/g7e 的 spec 页写「Yes via PCIe」，那是 PCIe P2P，不是 NVLink。所以 G 系 8 卡机型的域大小是 **1**：`TP > 1` 就已经在走 PCIe。

同时注意 P 系内部域内带宽也差 2×（600 → 900 → 1800 GB/s）——域不是一个只有大小的概念。

### 实例（instance）
一个 EC2 实例。**与 NVLink 域不是同一个概念**，两者可以在任意方向上不相等：

- P 系 8 卡机型：`1 实例 = 1 个域 = 8 卡`（重合）
- G 系 8 卡机型：`1 实例 = 8 卡`，但 `1 个域 = 1 卡`（**域 < 实例**）
- 域 > 实例的机型（一个域横跨多个实例）确实存在，但已排除出本工具范围

TP 的约束来自 **NVLink 拓扑**，不来自 EC2 实例边界。把理由说成"不能跨节点"，会把 P 系 8 卡机型的性质误当成普遍规律——`TP ≤ 8` 是那批硬件的性质，不是定理；在 G 系上正确的上限是 `TP ≤ 1`。工具的校验文案应称"跨 NVLink 域"。

### scale-out 域
跨 NVSwitch 域的连接，走 EFA + GPUDirect RDMA。带宽比域内低一个数量级以上，是所有跨域集合操作的成本来源。

## 并行维度

### TP（tensor parallelism，张量并行）
把**单个矩阵**横向切开，每张卡算一片，每层结束时用 all-reduce 求和。通信最密（每层 2 次），因此 **TP 度数不应超过 scale-up 域大小**。

### EP（expert parallelism，专家并行）
把 MoE 的**整个专家**原封不动分配到不同卡上，不切矩阵。token 经 all-to-all 发到其专家所在的卡。只覆盖 routed expert，**不覆盖 attention / dense / embedding**。

**EP 的上限是一个 pipeline stage 的卡数，不是总卡数：**

```
world = TP × DP × PP
stage = world ÷ PP = TP × DP        ← 一个 pipeline stage 有多少卡
EP ≤ stage，且 EP 必须整除 routed expert 数
```

理由：PP 是**按层切分**的，每个 stage 只持有自己那几层。这些层的专家只能摊在本 stage 的卡上 —— 摊不到别的 stage 去，因为那些卡装的是别的层。

所以 `PP = 1` 时 EP 才能取到总卡数；`PP = 2` 时上限只有总卡数的一半。例：16 卡上 `TP8/DP2/PP1` → EP 可到 16，但 `TP8/DP1/PP2` → EP 最多 8。

Kimi K3 有 896 = 2⁷ × 7 个 routed expert，而 stage 规模总是 2 的幂，7 除不进去，因此 EP 的实际候选就是「≤ stage 的 2 的幂」。

`EP < stage` 时 expert bank 会被复制 `stage ÷ EP` 份 —— 见「权重复制因子」。

### DP（data parallelism，推理语境）
**与训练的 DP 不是一回事**：没有梯度、没有 all-reduce。含义是"复制权重，切分请求"——每个 DP rank 持有完整的一份 attention 权重，处理不同的请求子集，rank 之间在 attention 阶段零通信。

在 MoE 模型里 DP 是**局部复制**：只复制非 expert 部分，expert bank 由所有 DP 组共享。所以 `DP=4` 不意味着"4 份完整模型"。

### DP attention
上述 DP 只作用于 attention 块的部署方式。引擎里 `dp_size` 指的是这个，不是模型副本数。SGLang 中 `--tp` 是 world size，attention 实际 TP 度数 = `tp_size / dp_size`。

### PP（pipeline parallelism，流水线并行）
按**层**切分。跨域通信只有每 token 几 KB 的 activation send/recv，是通信最省的维度；代价是流水线气泡，需要足够多的 micro-batch 填满。

### micro-batch
调度器凑成一次 forward 的那批请求。PP 需要同时有 ≥PP 个 micro-batch 在飞才能填满流水线。decode 阶段自回归，同一请求的连续 token 无法互相流水，所以这些 micro-batch **必须是互不相干的请求** —— 这构成了 PP 的最低并发门槛。

## 显存构成

### 单位（GiB，不是 GB）
真值是 `describe-instance-types` 的 `MemoryInfo.SizeInMiB`；内部计算用字节（`MiB × 1048576`）；UI 一律显示 **GiB**（`bytes / 2^30`）。

厂商标称的「GB」在不同 GPU 上含义不同、无统一规律：A100/H100/H200/RTX PRO 按 GiB 标，A10G/L4/L40S 与 B300 按十进制标。**因此永远不要拿标称数字直接当某一种单位用。** 这条曾导致把 H200 的 141 GiB 当成 141×10⁹ 字节、容量低估 7.4% 的实际 bug。详见 [[adr-0006]] 与 `instance-specs.md`。

### 权重复制因子
同一份权重在集群里存了几遍。由并行维度决定，是显存账的主项：

- 非 expert 权重：切 `TP × PP` 份，**复制 `DP` 份**
- routed expert：切 `EP × PP` 份，复制 `(stage / EP)` 份，其中 `stage = TP × DP = world ÷ PP`

### KV cache 复制因子
一个 token 的 KV 在集群里存了几遍。**由 attention family 决定，不是全局常数**：

- **MLA**（Kimi K3 的 24 个 full-attn 层）：压缩 latent 被所有 head 共享 → TP 组内每张卡各存一份完整副本 → **复制 TP 份**
- **GQA**：KV 按 kv head 切分 → `n_kv_heads ≥ TP` 时不复制；`n_kv_heads < TP` 时开始复制 `TP / n_kv_heads` 份
- **线性注意力 / SSM 的 recurrent state**：per-head，可按 head 切 → 不复制

DP 和 PP 在任何 family 下都不复制 KV。

### KV cache 的 dtype
**不是模型属性，是引擎启动参数**（`--kv-cache-dtype`，默认 `auto` = 模型 dtype = BF16；FP8 要显式开）。`config.json` 与 checkpoint 里都没有这项信息，工具无论怎么查模型都推不出来。它把每 token 的 KV 字节数直接翻倍/砍半，是本工具敏感度最高的单项，因此必须作为界面输入并显示反事实，不能当常数。详见 [[adr-0007]]。

与「KV cache 复制因子」是两件独立的事：复制因子由 attention family 与并行配置决定，dtype 由部署决定，两者相乘才是每卡的 KV 占用。

### per-request 固定开销
与序列长度无关、只随并发涨的状态。Kimi K3 的 69 个 KDA 层的 recurrent state 属于此类（96 head × 128 × 128 × 69 层）。高并发短上下文场景下它会反超随长度增长的 KV cache——**按"总 token 数"做容量规划会把这种场景估反**。

### util 预算（gpu_memory_utilization）
引擎允许动用的显存 = 物理显存 × util。权重 + overhead + KV + state 必须全部挤进这个预算，预算外的物理显存引擎不会碰。vLLM 默认 0.90。`util = 1.00` 是理论上界，不是可规划值。

## 数据可信度分级

这个工具的每个数字都必须能归到以下之一。混用而不标注是它最大的失效模式。

| 级别 | 含义 | 例 |
|---|---|---|
| `spec` | `describe-instance-types` API 或厂商 spec sheet | p6-b300 每卡 275040 MiB = 268.6 GiB |
| `derived` | 由 `config.json` / HF safetensors 元数据 / `index.json` 的 `total_size` 算出 | routed expert 共 2.7227T 参数、1347.0 GiB；非 expert 57.19B、106.5 GiB |
| `estimated` | 推算，有明确误差来源 | KDA state 的 dtype 假设为 FP32；**KV cache 的 dtype（界面可切，但选哪档仍是假设）**；expert 被重量化为 FP8/BF16 后按参数量推算的字节数 |
| `guessed` | 拍的，无依据 | 每卡 12 GiB 激活 + 通信 buffer（**当前唯一的 guessed 项，且它直接决定最大并发**） |
| `measured` | 目标硬件实测 | （暂无） |
