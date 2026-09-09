# model-viz

**LLM 并行切分与显存分布可视化。** 给定模型、机型、台数和 `TP/DP/PP/EP`,算出每张卡的显存构成、装不装得下、能收多少并发。

零构建、零依赖、无需服务器 —— 克隆下来**双击 `index.html`** 就能用,整个目录打包发给别人也一样能开。

## 范围(先读这个)

**这套工具只算显存。**

- 不回答吞吐、延迟、TTFT/ITL —— 「装得下」不等于「跑得快」。
- 不回答价格与可得性,因此**不能**用于判断「哪个机型更值」。
- 是**校验器不是求解器**:你给配置,它判定;它不会替你搜索最优解。
- 头号数字(最大并发)建立在若干软数字上 —— 每卡 12 GiB 的「激活 + 通信 buffer」是**猜测**(±12 GiB 使并发变动约 20%),KV cache 的 dtype 是**假设**(BF16↔FP8 使并发变动约 100%);GLM-5.3-Flash 还多一项 DSA indexer key cache 的池化与 dtype(±10%)。**不可作为容量规划或采购承诺。** 每个页面顶部都会按当前模型逐项列出,并明说清单不保证已穷举。

## 结构

| 文件 | 作用 |
|---|---|
| `index.html` | 入口:模型索引 + 机型目录 |
| `app.html?model=<id>` | 唯一引擎,所有公式只有一份实现 |
| `data/instances.js` | 机型规格:显存、NVLink 域、跨域带宽、原生 dtype |
| `data/models/<id>.js` | 模型定义:层结构、MoE、权重、候选机型 |
| `verify-app.js` | `node verify-app.js` —— 改完 `app.html` 必须跑 |
| `docs/` | 领域词汇、实例规格总目录、ADR |

数据文件是 `.js` 而不是 `.json`:`file://` 下 `fetch()` 会被 CORS 拦掉,classic `<script>` 标签不受限制。

加一个模型 = 加一个 `data/models/<id>.js` + 在 `index.html` 和 `app.html` 里各加一行 `<script src>`。

## 设计文档

设计已收敛,决策都在 `docs/` 里:

- [`docs/glossary.md`](docs/glossary.md) — 领域词汇。NVLink 域 ≠ 实例、推理 DP ≠ 训练 DP、KV 复制因子按 attention family 而异、provenance 五级
- [`docs/instance-specs.md`](docs/instance-specs.md) — G5–G7 / P4d–P6 共 57 个实例的规格总目录,以 `describe-instance-types` 的 MiB 为真值
- [`docs/adr/`](docs/adr/) — ADR-0001 至 0008:TP 上限由 NVLink 域决定(不是「节点」)、只算显存、单位一律 GiB、KV cache 的 dtype 是引擎参数而非模型属性、权重按两桶实测字节记账

## 当前收录

2 个模型、10 个机型(P6/P5/P4 与 G7/G6e)。

| 模型 | 权重 | 结构 | 一台 p5en 能收多少并发 |
|---|---|---|---|
| [Kimi K3](data/models/kimi-k3.js) | 1453.7 GiB(MXFP4) | 93 层 = 69 KDA + 24 MLA · 896 experts / top-16 | 装不下,4 台起 → 68 路 |
| [GLM-5.3-Flash](data/models/glm-5.3-flash.js) | 305.8 GiB(FP8) | 45 层 = 34 KDA + 11 DSA · 288 experts / top-8 | 320 路(TP1×DP8)|

> 上面这两个路数都是 **128K 上下文 / util 0.90 / BF16 KV / 原生量化** 下的读数。**任何路数都必须连口径一起报** —— 同一个 TP1×DP8,换成 1M 上下文只剩 40 路,换成 FP8 KV 则翻到 584 路。页面上预设按钮的路数是按你当前拖到的口径**现算**的,并直接标在数字旁边(`320 路 @128K/0.90`)。

两个模型的切分权衡**方向相反**:K3 的非 expert 权重有 106.5 GiB,DP 复制它很贵;GLM 只有 15.5 GiB,复制几乎免费,而 TP 会把 KV latent 复制 TP 份 —— 同一台机器上 128K 时 TP1×DP8 是 320 路、TP8/DP1 只有 53 路。但这个 6 倍差距随口径缩水:1K 上下文下只剩 1.11×,而且那时 TP2×DP4 会反超两者。所以工具**不标「推荐」**,排序交给现算的数字(见 [ADR-0008 §9](docs/adr/0008-two-bucket-measured-weights-and-dsa.md))。
