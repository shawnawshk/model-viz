# model-viz

**LLM 并行切分与显存分布可视化。** 给定模型、机型、台数和 `TP/DP/PP/EP`,算出每张卡的显存构成、装不装得下、能收多少并发。

零构建、零依赖、无需服务器 —— 克隆下来**双击 `index.html`** 就能用,整个目录打包发给别人也一样能开。

## 范围(先读这个)

**这套工具只算显存。**

- 不回答吞吐、延迟、TTFT/ITL —— 「装得下」不等于「跑得快」。
- 不回答价格与可得性,因此**不能**用于判断「哪个机型更值」。
- 是**校验器不是求解器**:你给配置,它判定;它不会替你搜索最优解。
- 头号数字(最大并发)建立在两个软数字上 —— 每卡 12 GiB 的「激活 + 通信 buffer」是**猜测**(±12 GiB 使并发变动约 20%),KV cache 的 dtype 是**假设**(BF16↔FP8 使并发变动约 100%)。**不可作为容量规划或采购承诺。**

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
- [`docs/adr/`](docs/adr/) — ADR-0001 至 0007:TP 上限由 NVLink 域决定(不是「节点」)、只算显存、单位一律 GiB、KV cache 的 dtype 是引擎参数而非模型属性

## 当前收录

1 个模型(Kimi K3)、10 个机型(P6/P5/P4 与 G7/G6e)。
