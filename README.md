# model-viz

**LLM 并行切分与显存分布可视化。**

输入：模型、机型、台数和 `TP/DP/PP/EP`。工具给出以下结果：

- 每张卡的显存构成；
- 显存装不装得下；
- 最大并发；
- decode 每步耗时的 **roofline 下界**，分三条线：HBM 带宽、算力、通信；
- 拐点并发与临界上下文。

roofline 下界与显存结果用同一组输入。

工具不需要构建步骤，没有依赖，也不需要服务器。克隆 repo 后，**双击 `index.html`** 即可使用。把整个目录打包发给其他人，对方也可以直接打开。

## 范围（先读这一节）

**本工具计算显存，并计算 decode 的 roofline 下界。**

- 吞吐与延迟只是**理论下界**，不是预测（ADR-0010）。
  - 计算用厂商 spec 的峰值，没有效率系数。
  - 计算不包含：每步固定开销、attention 本身的 FLOPs、通信延迟。
  - 工具不计算 prefill / TTFT。
  - 在低并发下，实测值比下界慢一个量级以上是正常的。
  - 下界的用途是**确定方向**：
    - 当前配置在哪个 regime；
    - 增加并发或更换机型时，哪条线会变；
    - benchmark 应在哪几个并发点采样。
  - 下界不能代替 benchmark。
- 工具不提供价格与可得性。所以工具**不能**判断「哪个机型更值」。
- 工具是**校验器 + 受限反事实建议器**，不是全局求解器（ADR-0011）。
  - 在每个模型页面上，工具可以枚举当前机型、当前台数内的合法切分，并展示 Pareto 改善。
  - 工具不跨机型或台数搜索最优解。
  - 工具不判断采购价值。
- 头号数字（最大并发）依赖以下软数字：
  - **所有模型：** 每卡 12 GiB 的「激活 + 通信 buffer」是**估计值**。
    - 这个值 ±12 GiB 时，并发变动约 20%。
    - 2026-09-16 有了第一个实测值：GLM-5.3-Flash 在 p5en/TP8 上，引擎自报 6.98 GiB。
    - **这个实测值不能外推到其他模型或其他 TP，误差方向也不确定。** 见 [`docs/measurements.md`](docs/measurements.md) M-001。
  - **K3 与 GLM-5.3-Flash：** KV cache 的 dtype 是**假设**。BF16↔FP8 使并发变动约 100%。
  - **仅 GLM-5.3-Flash：** DSA indexer key cache 的池化方式与 dtype，影响约 ±10%。
  - **仅 DeepSeek-V4.1-Flash：**
    - KV dtype 由架构固定，所以不是假设。
    - 但这个模型有**目前最大的一项软数字**：189 GiB 的 engram 是否整张常驻 HBM。影响约 40%。
    - 短期内**不能**用实测消除这一项，因为还没有生产引擎支持这个结构。
  - **仅 Qwen3.8-27B：** Gated DeltaNet state 的 dtype 有两个官方答案。
    - config 写 float32。
    - vLLM 默认使用 BF16。
    - 在 1K 上下文下，两者相差约 1.5 倍。在 128K 上下文下，相差不到 1%。
- **不要把本工具的结果用作容量规划或采购承诺。**
- 每个页面的顶部按当前模型逐项列出这些软数字。页面也说明：这个清单可能不完整。

## 结构

| 文件 | 作用 |
|---|---|
| `index.html` | 入口：模型索引 + 机型目录 |
| `app.html?model=<id>` | 唯一的引擎。所有公式只有一份实现 |
| `styles.css` | 两个页面共用的主题、导航与响应式布局 |
| `sidebar.js` | 两个页面共用的导航（按 provider 分组模型）与侧栏的收起状态 |
| `assets/providers/` | 本地 provider logo，以及 logo 的来源与许可 |
| `favicon.svg` | 应用图标 |
| `data/instances.js` | 机型规格：显存、NVLink 域、跨域带宽、原生 dtype、HBM 带宽、dense TFLOPS（HBM 带宽与 dense TFLOPS 附 datasheet 出处） |
| `data/models/<id>.js` | 模型定义：层结构、MoE、权重、候选机型 |
| `verify-app.js` | 修改 `app.html` 后，必须执行 `node verify-app.js` |
| `docs/` | 领域词汇、实例规格总目录、实测记录、ADR |

数据文件是 `.js`，不是 `.json`。原因：在 `file://` 下，CORS 会阻止 `fetch()`。classic `<script>` 标签不受这个限制。

加一个模型需要三步：

1. 加一个 `data/models/<id>.js`。
2. 在 `index.html` 里加一行 `<script src>`。
3. 在 `app.html` 里加一行 `<script src>`。

加模型时还要注意两点：

- 侧栏按模型的 `hf` namespace 分组。如果模型来自新的 provider，在 `sidebar.js` 中登记 provider 的名称和 namespace，并把 logo 放入 `assets/providers/`。
- 模型定义中的 `modalities` 字段记录输入/输出模态（`text`、`image`、`video`）及官方来源。详情页用这个字段展示能力标签。

## 设计文档

设计已经确定。所有决策都在 `docs/` 里：

- [`docs/glossary.md`](docs/glossary.md)：领域词汇。包括：NVLink 域 ≠ 实例；推理 DP ≠ 训练 DP；KV 复制因子按 attention family 不同；provenance 分五级。
- [`docs/instance-specs.md`](docs/instance-specs.md)：G5–G7 / P4d–P6 共 57 个实例的规格总目录。真值用 `describe-instance-types` 给出的 MiB。
- [`docs/measurements.md`](docs/measurements.md)：实测记录。目前只有一条（M-001）：2026-09-16 在 p5en 上运行 GLM-5.3-Flash 服务，读取引擎自报的内存账。这条实测用来替换每卡 12 GiB overhead 这个估计常数。**每条记录都写明：结论可以推广到哪里，不能推广到哪里。**
- [`docs/adr/`](docs/adr/)：ADR-0001 至 0012。主要决策：
  - TP 上限由 NVLink 域决定，不由「节点」决定；
  - 单位一律用 GiB；
  - KV cache 的 dtype 是引擎参数，不是模型属性（ADR-0009 记录了例外）；
  - 权重按三桶实测字节计算（第三桶按 TP 切，不按 PP 切）；
  - decode roofline 下界（ADR-0010）；
  - 同机型、同台数内的 Pareto 反事实建议器（ADR-0011）；
  - GQA family 与 dense 模型（ADR-0012）：KV 复制因子是并行配置的函数；`moe` 块是可选的。

## 当前收录

4 个模型，10 个机型（P6/P5/P4 与 G7/G6e）。

| 模型 | 权重 | 结构 | 一台 p5en 的最大并发 |
|---|---|---|---|
| [Kimi K3](data/models/kimi-k3.js) | 1453.7 GiB（MXFP4） | 93 层 = 69 KDA + 24 MLA · 896 experts / top-16 | 一台装不下。最少 4 台 → 68 路 |
| [GLM-5.3-Flash](data/models/glm-5.3-flash.js) | 305.8 GiB（FP8） | 45 层 = 34 KDA + 11 DSA · 288 experts / top-8 | 320 路（TP1×DP8） |
| [DeepSeek-V4.1-Flash](data/models/deepseek-v4.1-flash.js) | 475.2 GiB（原生 FP4/FP8 混合） | 40 层 = 4 CSA2 + 36 SWA · 384 experts / top-6 | 538 路（TP4×DP2；TP1 装不下） |
| [Qwen3.8-27B](data/models/qwen3.8-27b.js) | 51.7 GiB（BF16；官方 FP8 28.7 GiB） | 64 层 = 48 Gated DeltaNet + 16 GQA · **dense，无 MoE** | 100 路（TP4×DP2；TP8 只有 53 路） |

> 上表的四个路数都在以下口径下读取：**128K 上下文 / util 0.90 / BF16 KV / 原生量化**。
>
> **报告任何路数时，都必须同时报告口径。** 例如，同一个 TP1×DP8：
> - 改为 1M 上下文，只剩 40 路；
> - 改为 FP8 KV，增加到 584 路。（DSv4.1 的 KV dtype 由架构固定，所以这一项对 DSv4.1 无效。）
>
> 页面上的预设按钮按当前口径**实时计算**路数，并把口径标在数字旁边（`320 路 @128K/0.90`）。

四个模型的切分权衡**方向各不相同**。这是本工具最值得看的一点：

- **K3 需要 TP。** 非 expert 权重有 106.5 GiB，用 DP 复制它的代价很高。
- **GLM 需要 DP。**
  - 非 expert 权重只有 15.5 GiB，复制的代价几乎为零。
  - TP 会把 KV latent 复制 TP 份。
  - 在同一台机器上、128K 上下文下：TP1×DP8 是 320 路，TP8/DP1 只有 53 路。
  - 这个 6 倍差距随口径变小：在 1K 上下文下只剩 1.11×。在 1K 下，TP2×DP4 的并发超过这两个配置。
- **DSv4.1 的 TP 有下限，下限随单卡显存变化。**
  - 475.2 GiB 权重中有 189.13 GiB 是 engram。engram 只能按 TP 切。
  - 在 p5en 上，TP 必须 ≥ 4（TP4 = 538 路 > TP8 = 488 路）。
  - 在 b300 上，TP = 2 最优。TP = 1 差 5.56 GiB，装不下。
  - 在 GLM 页上，点 TP1×DP8 时并发增加 6 倍。在 DSv4.1 页上，同样的操作直接显示红字「装不下」。
- **Qwen3.8-27B 的最佳 TP 等于 n_kv_heads。**
  - GQA 只有 4 个 kv head。
  - TP ≤ 4 时，KV 按 head 切，不复制。同时 TP 也切分权重。
  - TP = 8 时，TP 超过 4 个 head，KV 复制 2 份，DP 减少一半。并发从 100 路降到 53 路，与 TP1×DP8 的 56 路接近。
  - Qwen3.8-27B 是第一个 dense 模型。它也是第一个把 H100 80GB 与 G 系（g7e）列入候选的模型。
  - g7e 没有 NVLink，域 = 1，所以 TP 上限是 1。每张卡都必须放下全部 51.75 GiB 权重。这是 ADR-0001「显存装得下不代表能用」第一次在页面上起作用。
  - 候选机型原来还有 g6e。2026-09-17 移除了 g6e。原因：
    - g6e 的余量是 11.5 GiB（预算 40.23 − 官方 FP8 权重 28.75）。
    - 每卡 12 GiB overhead 的估计值与实测值之间的差，大于这个余量。
    - 所以红字结论只由这个未校准的常数决定。本工具不能对 g6e 给出可靠结论。见 `docs/measurements.md` M-001。

静态预设主要用于展示结构与反例。每个模型的决策层只在以下范围内展示 Pareto 改善与代价：当前机型、当前台数、用户选定的目标。决策层使用该模型自己的 KV、state 与第三权重桶约束。决策层不声称全局最优（ADR-0011）。
