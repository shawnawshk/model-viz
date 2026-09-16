# model-viz

**LLM 并行切分与显存分布可视化。** 给定模型、机型、台数和 `TP/DP/PP/EP`,算出每张卡的显存构成、装不装得下、能收多少并发;再按同一组输入给出 decode 每步耗时的 **roofline 下界**(HBM 带宽 / 算力 / 通信三条线)、拐点并发与临界上下文。

零构建、零依赖、无需服务器 —— 克隆下来**双击 `index.html`** 就能用,整个目录打包发给别人也一样能开。

## 范围(先读这个)

**这套工具算显存,外加 decode 的 roofline 下界。**

- 吞吐与延迟只给**理论下界**,不是预测(ADR-0010)。按厂商 spec 峰值算,没有效率系数,不含每步固定开销、attention 本身的 FLOPs、通信延迟;不算 prefill / TTFT。低并发下实测比下界慢一个量级以上是正常的。它的用途是**定方向**:当前配置在哪个 regime、加并发 / 换机器动的是哪条线、benchmark 该在哪几个并发点采样 —— 不是替代 benchmark。
- 不回答价格与可得性,因此**不能**用于判断「哪个机型更值」。
- 是**校验器 + 受限反事实建议器**,不是全局求解器:当前收录的模型页面都可在当前机型、当前台数内枚举合法切分并展示 Pareto 改善;不会跨机型 / 台数搜索最优解,也不做采购价值判断(ADR-0011)。
- 头号数字(最大并发)建立在若干软数字上 —— 每卡 12 GiB 的「激活 + 通信 buffer」是**猜测**(±12 GiB 使并发变动约 20%),KV cache 的 dtype 是**假设**(BF16↔FP8 使并发变动约 100%);GLM-5.3-Flash 还多一项 DSA indexer key cache 的池化与 dtype(±10%);DeepSeek-V4.1-Flash 的 KV dtype 由架构固定、不再是假设,但它多出**至今最大的一项** —— 189 GiB 的 engram 是否整张常驻 HBM(约 40%,且短期内**无法**用实测消除,还没有生产引擎支持这个结构);Qwen3.8-27B 的 Gated DeltaNet state dtype 有两个官方答案(config 写 float32,vLLM 默认落到 BF16),1K 上下文下差约 1.5 倍、128K 下不到 1%。**不可作为容量规划或采购承诺。** 每个页面顶部都会按当前模型逐项列出,并明说清单不保证已穷举。

## 结构

| 文件 | 作用 |
|---|---|
| `index.html` | 入口:模型索引 + 机型目录 |
| `app.html?model=<id>` | 唯一引擎,所有公式只有一份实现 |
| `data/instances.js` | 机型规格:显存、NVLink 域、跨域带宽、原生 dtype、HBM 带宽、dense TFLOPS(后两项附 datasheet 出处) |
| `data/models/<id>.js` | 模型定义:层结构、MoE、权重、候选机型 |
| `verify-app.js` | `node verify-app.js` —— 改完 `app.html` 必须跑 |
| `docs/` | 领域词汇、实例规格总目录、ADR |

数据文件是 `.js` 而不是 `.json`:`file://` 下 `fetch()` 会被 CORS 拦掉,classic `<script>` 标签不受限制。

加一个模型 = 加一个 `data/models/<id>.js` + 在 `index.html` 和 `app.html` 里各加一行 `<script src>`。

## 设计文档

设计已收敛,决策都在 `docs/` 里:

- [`docs/glossary.md`](docs/glossary.md) — 领域词汇。NVLink 域 ≠ 实例、推理 DP ≠ 训练 DP、KV 复制因子按 attention family 而异、provenance 五级
- [`docs/instance-specs.md`](docs/instance-specs.md) — G5–G7 / P4d–P6 共 57 个实例的规格总目录,以 `describe-instance-types` 的 MiB 为真值
- [`docs/adr/`](docs/adr/) — ADR-0001 至 0012:TP 上限由 NVLink 域决定(不是「节点」)、单位一律 GiB、KV cache 的 dtype 是引擎参数而非模型属性(ADR-0009 承认有例外)、权重按三桶实测字节记账(第三桶按 TP 切、不按 PP 切)、decode roofline 下界(ADR-0010)、同机型同台数的 Pareto 反事实建议器(ADR-0011)、GQA family 与 dense 模型(ADR-0012:KV 复制因子是并行配置的函数;`moe` 块可选)

## 当前收录

4 个模型、10 个机型(P6/P5/P4 与 G7/G6e)。

| 模型 | 权重 | 结构 | 一台 p5en 能收多少并发 |
|---|---|---|---|
| [Kimi K3](data/models/kimi-k3.js) | 1453.7 GiB(MXFP4) | 93 层 = 69 KDA + 24 MLA · 896 experts / top-16 | 装不下,4 台起 → 68 路 |
| [GLM-5.3-Flash](data/models/glm-5.3-flash.js) | 305.8 GiB(FP8) | 45 层 = 34 KDA + 11 DSA · 288 experts / top-8 | 320 路(TP1×DP8)|
| [DeepSeek-V4.1-Flash](data/models/deepseek-v4.1-flash.js) | 475.2 GiB(原生 FP4/FP8 混合) | 40 层 = 4 CSA2 + 36 SWA · 384 experts / top-6 | 538 路(TP4×DP2;TP1 装不下)|
| [Qwen3.8-27B](data/models/qwen3.8-27b.js) | 51.7 GiB(BF16;官方 FP8 28.7 GiB) | 64 层 = 48 Gated DeltaNet + 16 GQA · **dense,无 MoE** | 100 路(TP4×DP2;TP8 只有 53)|

> 上面这四个路数都是 **128K 上下文 / util 0.90 / BF16 KV / 原生量化** 下的读数。**任何路数都必须连口径一起报** —— 同一个 TP1×DP8,换成 1M 上下文只剩 40 路,换成 FP8 KV 则翻到 584 路(DSv4.1 的 KV dtype 由架构固定,那一档对它无效)。页面上预设按钮的路数是按你当前拖到的口径**现算**的,并直接标在数字旁边(`320 路 @128K/0.90`)。

四个模型的切分权衡**方向各不相同**,这是这套工具最值得看的一点:

- **K3 要 TP** —— 非 expert 权重有 106.5 GiB,DP 复制它很贵。
- **GLM 要 DP** —— 非 expert 只有 15.5 GiB,复制几乎免费,而 TP 会把 KV latent 复制 TP 份。同一台机器上 128K 时 TP1×DP8 是 320 路、TP8/DP1 只有 53 路。但这个 6 倍差距随口径缩水:1K 上下文下只剩 1.11×,那时 TP2×DP4 还会反超两者。
- **DSv4.1 的 TP 有下限,而下限随单卡显存移动** —— 475.2 GiB 里有 189.13 GiB 的 engram 只能按 TP 切。p5en 上 TP 必须 ≥4(TP4 = 538 路 > TP8 = 488 路),b300 上 TP=2 最优而 TP=1 差 5.56 GiB 装不下。GLM 页上「点 TP1×DP8 看并发跳 6 倍」的演示,在这一页是直接红字装不下。
- **Qwen3.8-27B 的 TP 甜点在 n_kv_heads 上** —— GQA 只有 4 个 kv head,TP ≤ 4 时 KV 按 head 切、不复制,TP 同时还切权重;TP=8 越过 4 个 head,KV 复制 2 份、DP 少一半,并发从 100 掉回 53,与 TP1×DP8 的 56 打平。它也是第一个 dense 模型,和第一个把 H100 80GB 与 G 系(g7e / g6e)列进候选的模型:g6e 上 BF16 装不下,TP=2 显存够却要跨 PCIe —— ADR-0001「显存装得下不代表能用」第一次实际起作用。

静态预设主要用于展示结构与反例；各模型的决策层只在当前机型、当前台数和用户选定目标内展示 Pareto 改善与代价,并沿用该模型自己的 KV、state 与第三权重桶约束,不声称全局最优(ADR-0011)。
