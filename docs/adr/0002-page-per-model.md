# ADR-0002：模型维度用「单引擎 + 数据驱动」，而非每模型一个页面

- 状态：**Accepted** —— 用户已确认取「读法 A：导航分页，单引擎」
- 日期：2026-08-31

## 用户的表述

> 除了 instance，如果我要再想增加一个维度，支持其他模型呢？我期待是可以以模型为一个大的维度，分页面。

## 分歧点

「分页面」有两种读法，代价差一个数量级：

**读法 A（导航语义）**：用户想要「一个模型一个入口」的浏览体验。
**读法 B（代码语义）**：每个模型一个自包含的 HTML 文件。

我认为用户想要的是 A，但字面表述是 B。若按 B 实现：

1. **内存模型被复制 N 份。** 当前 `compute()` 里的每一条公式（权重复制因子、KV 复制因子、util 预算、容量反解）都会在每个文件里存在一份。修一个 bug 要改 N 个文件，且必然漂移。
2. **跨模型比较做不了。** 而这恰恰是有价值的问题：「同样 4 台 p6-b300，K3 和 DeepSeek-V4 哪个能跑更高并发」在 B 下无法回答。
3. **instance 维度会与 model 维度相乘。** N 个模型 × M 个 instance，若两者都靠文件复制，组合爆炸。

## 提议

```
model-viz/
├── index.html          # 索引页：模型卡片列表（满足读法 A 的导航需求）
├── app.html            # 唯一的引擎 + UI
├── data/
│   ├── instances/      # p5en.json, p6-b200.json, p6-b300.json
│   └── models/         # kimi-k3.json, deepseek-v4.json, ...
└── docs/
```

- 选择通过 URL 传递：`app.html?model=kimi-k3&instance=p6-b300`
- `index.html` 生成模型卡片链到 `app.html?model=<id>`，用户看到的仍是「一个模型一页」
- 所有公式只有一份实现

## 关键难点：attention family 不是常数，是语义

这是模型维度的真正成本，不在「换几个数字」上。当前实现硬编码了两个 attention family：

| family | KV 结构 | TP 交互 |
|---|---|---|
| MLA | 压缩 latent，所有 head 共享 | **复制 TP 份** |
| 线性注意力（KDA） | per-head recurrent state，与序列长度无关 | 按 head 切，**不复制** |

要支持其他模型，至少还需要：

| family | KV 结构 | TP 交互 |
|---|---|---|
| GQA | `2 × n_kv_heads × head_dim` / token / 层 | `n_kv_heads ≥ TP` 不复制；否则复制 `TP / n_kv_heads` 份 |
| MHA | `2 × n_heads × head_dim` | 按 head 切，不复制 |
| 滑窗 / 混合 | 每层按窗口上限截断 | 同上，但每层容量有上限 |
| SSM / Mamba | 固定大小 state | per-channel 可切 |

**GQA 那一行是关键反例**：它的 TP 复制因子不是常数，而是 `max(1, TP / n_kv_heads)` —— 一个依赖并行配置的函数。也就是说「KV 复制因子」不能建模成模型属性，必须是 `f(attention_family, model_config, parallel_config)`。

同理，模型维度还会改变**控件集本身**：dense 模型没有 EP 轴，MoE 才有。UI 必须能按模型隐藏无意义的维度。

## 建议的数据契约（草案）

```jsonc
{
  "id": "kimi-k3",
  "totalParams": 2.8e12,
  "layers": [
    { "count": 24, "family": "mla",    "kvLoraRank": 512, "qkRopeDim": 64, "kvDtype": 1 },
    { "count": 69, "family": "linear", "heads": 96, "headDim": 128, "stateDtype": 4 }
  ],
  "moe": { "layers": 92, "experts": 896, "topk": 16, "sharedExperts": 2,
           "latent": 3584, "inter": 3072, "bytesPerParam": 0.53125 },
  "nonExpertParams": { "value": 77.26e9, "confidence": "estimated",
                       "note": "2.8T 减 expert 部分的残差，误差可能几十 B" }
}
```

注意 `layers` 是**分段列表**而非单一 family —— 混合注意力（K3 是 69 KDA + 24 MLA）是常态而非特例，数据契约从第一天就必须支持。

## 决议

取**读法 A**。`index.html` 做模型索引，`app.html?model=` 是唯一引擎，公式只有一份实现。

## 实施记录（2026-09-01）与两处偏离

已实施。实际结构：

```
index.html                  模型索引（卡片 + 已收录机型表 + 边界声明 + docs 链接）
app.html?model=kimi-k3      唯一引擎
data/instances.js           10 个机型的目录
data/models/kimi-k3.js      模型定义
```

**偏离 1：数据文件用 `.js` 而非 `.json`。**
原方案设想 `fetch()` 读 JSON。但本工具是从 `file://` 直接打开的（用户一直 `open index.html`，且页面可能要发给外部读者），而 `file://` 下 `fetch()` 会被 CORS 拦掉。可选方案是要求跑本地 HTTP server 或加构建步骤，两者都增加摩擦。
最终用 classic `<script>` 标签加载 `.js` 数据文件（`REG.instances = {...}` / `REG.models["id"] = {...}`）——script 标签在 `file://` 下不受 CORS 限制，**零构建、零服务器、双击即用**。代价是数据文件不是纯 JSON，且新增模型要同时加一个 `<script>` 标签。

**偏离 2：机型不按 `instances/<id>.json` 一个一个拆。**
10 个机型共用同一 shape、且是扁平目录性质，拆成 10 个文件属于为不存在的规模做结构。合并为单个 `data/instances.js`。
**模型仍是一文件一个**（`data/models/<id>.js`）——模型才是会长的那个维度，也是本 ADR 关心的轴。

## attention family 抽象的落地形态

引擎里是一张语义表，加一个 family 就是加一条，`compute()` 不动：

```js
const FAMILY = {
  mla:    { perTokenBytes: g => (g.kvLoraRank + g.qkRopeDim) * g.kvDtypeBytes,
            perReqBytes: () => 0, stateShardableByTp: false },
  linear: { perTokenBytes: () => 0,
            perReqBytes: g => g.heads * g.headDim * g.headDim * g.stateDtypeBytes,
            stateShardableByTp: true },
};
```

模型的 `layers` 是分段列表，引擎遍历求和得到 `kvBytesPerToken` 与 `stateBytesPerReq`。K3 的 69 KDA + 24 MLA 由此自然表达。

**仍未验证**：`gqa` 那一条（复制因子 `max(1, TP/n_kv_heads)` 是并行配置的函数，不是常数）在只有一个模型时无法验证，因此**没有预先写进表里**。加第二个模型时再加。

## 未决

- 是否需要跨模型对比视图，或只需跨 instance 对比？（不阻塞，可后加）
- attention family 抽象的落地时机：见 ADR-0003，model 轴排在 instance 轴之后，契约先抽出但只填 Kimi K3 一个模型。
