# ADR-0013：按 TP 切的查表放在哪是引擎参数，提为输入，默认取 vLLM 的 host offload

- 状态：**Accepted → 已在 `app.html` + `data/models/deepseek-v4.1-flash.js` + `data/models/qwen3.8-flash-next.js` 实施**（2026-10-30）
- 触发：收录 Qwen3.8-Flash-Next 时发现，vLLM 已经实现了 engram / PLE，而且默认把表放在 host 内存
- 影响：
  - **取代** [ADR-0009](0009-third-weight-bucket-and-csa2.md) §2（「engram 整张算进 HBM，不做成界面开关」）。ADR-0009 §1 的第三桶与 §8 的「不按 PP 切」不变。
  - 新增界面输入「查表放置」，只在带第三桶的模型上显示。
  - data 契约：`expertFormats` 的条目可以带 `tpShardedBytes`（这一档 checkpoint 里第三桶的字节数）与 `measured`（这一档是官方 checkpoint 的实测）；模型可以带 `lookupHbmCaveat`（常驻那一档的局限）。
  - DeepSeek-V4.1-Flash 页面的头号数字改变（见「后果」）。
- 相关：[ADR-0007](0007-kv-dtype-is-an-input.md)（引擎参数提为输入的先例）、[ADR-0004](0004-provenance-first-class.md)（反事实同屏）、[ADR-0009](0009-third-weight-bucket-and-csa2.md)（第三桶）、[ADR-0011](0011-bounded-counterfactual-advisor.md)（决策层）

## 触发原因：ADR-0009 §2 的前提不再成立

ADR-0009 §2 不做开关的理由是：「engram 的放置方式，在任何生产引擎里都还不存在。给一个不存在的引擎行为做配置项，就是把猜测伪装成选项。」

2026-10-30 读 vLLM main 的源码与文档，这个前提已经不成立：

| 事实 | 出处 |
|---|---|
| vLLM 有 engram 的功能页，2026-09-21 加入 | `docs/features/engram.md`（commit `fba33e39cb`） |
| 支持两个架构：`DeepseekV41ForCausalLM`（`engram_layer_ids`）与 `Qwen4ExpFor*`（`ple_layer_ids`，「same mechanism, called PLE」） | `vllm/config/engram.py` 的 `_NGRAM_LAYER_FIELDS` |
| 模型带这类层，就自动建 `EngramConfig()`，不用传参 | `vllm/config/vllm.py` 的 `_resolve_and_verify_engram_config` |
| `cpu_offload` 默认 `True`：表放在 pinned host memory，经 UVA 查，在前几层计算时从侧 stream 预取 | `EngramConfig.cpu_offload: bool = True`；文档原文「The freed GPU memory goes to the KV cache」 |
| 常驻时，表按 ETP 组做 vocab-parallel 切分。ETP 组默认就是 TP 组 | `parallel_state.py`：`engram_tensor_parallel_size == tensor_model_parallel_size` 时 `_ETP = _TP` |
| `embedding_across_dp: true` 时，hash head 摊到 TP × DP 全部卡上，只存一份（仅 DSv4.1） | `EngramConfig.get_parallel_size` |
| 同节点的 DP replica 默认经 `/dev/shm` 共享一份 host 表（仅 DSv4.1；文档写明 Qwen4Exp 的 PLE 只认 `cpu_offload`） | `EngramConfig.dp_shared_memory`；`docs/features/engram.md` |
| H100 上 Qwen3.8-Flash-Next 必须 offload；H200 及以上可选；4×H100 TP4 不 offload 会在启动时 OOM | vLLM recipe `Qwen/Qwen3.8-Flash-Next` |

所以「放在哪」和 `--kv-cache-dtype` 是同一类东西：**引擎里已经存在的启动参数，有一个默认值**。ADR-0007 的处理方式直接适用。

## 决议

### 1. 提为界面输入，第一档 = vLLM 默认

```js
const LOOKUP_PLACES = [
  { id: "host", label: "host 内存(vLLM 默认 offload)" },
  { id: "hbm",  label: "常驻 HBM(按 TP 切)" },
];
```

| 档 | 单卡 HBM 占用 | 对应参数 |
|---|---|---|
| `host`（默认） | 0 | 默认，或 `--engram-config.cpu_offload true` |
| `hbm` | `第三桶字节 ÷ TP`，不 ÷PP（ADR-0009 §8） | `--engram-config.cpu_offload false` |

这个输入是引擎级的，不是模型属性，所以写在 `app.html` 里，不写在数据文件里。控件只在 `weights.tpShardedBytes > 0` 的模型上显示。

### 2. 反事实与结论同屏（ADR-0004 / ADR-0007 的结构）

`compute()` 多算一个 `maxConcLookupAlt`：同一配置换另一种放置后的最大并发。它出现在四处：

- banner：默认档是 `info`，非默认档是 `warning`。两档都写出另一档的路数。
- 边界声明块的假设清单：排第一。这一项能把结论翻得最狠（DSv4.1 的 TP1×DP8 在两档之间是 0 路 ⇄ 4,928 路）。
- 假设区：放置方式的完整说明。
- 决策层：常驻时，「增加容量」目标的首选动作是「把查表放回 host」。

### 3. host 那一档不算的东西

- **host 内存。** 本工具只算 GPU 显存。banner 写明 host 内存要放得下这张表，但不给数。
- **查表读取。** 每 token 只查几十行（KB 量级）。roofline 在两档下都不计这一项。host 那一档走 PCIe，经 UVA 预取，也不计。

### 4. 常驻那一档的路数是上限

Qwen3.8-Flash-Next 的 recipe 记录：4×H100 上 TP4 不 offload，启动时 OOM。按「÷TP 常驻」算，这时每卡扣完 12 GiB overhead 还剩 16.8 GiB。所以常驻的真实开销比 ÷TP 更大（可能是加载峰值）。本页常驻那一档的路数因此只是上限。这一条写在模型的 `lookupHbmCaveat` 里，显示在常驻那条 banner 中。

### 5. provenance：第三桶从 `guessed` 回到 `derived`

ADR-0009 §2 把这一项标成 `guessed`：字节数是实测，但「整张常驻」是猜测。现在放置是输入，默认值来自 vLLM 源码，所以这一项标 `derived`。

### 6. 第三桶的字节数可以随 checkpoint 变

Qwen3.8-Flash-Next 的官方 FP8 checkpoint 把 n-gram 表也量化了（BF16 95.37 GiB → FP8 47.68 GiB）。所以 `expertFormats` 的 FP8 那一档要连第三桶一起换：

```js
{ id: "fp8", label: "...", bytesPerParam: 1.0001220703125,
  tpShardedBytes: 51_200_245_762, measured: true }
```

`compute()` 取 `expFmt.tpShardedBytes ?? weights.tpShardedBytes`。DSv4.1 的 FP8 那一档来自 `convert.py`，不动 engram，所以不带这个字段。

`measured: true` 表示这一档是官方另发的 checkpoint，字节数逐张量实测，不是按参数量推算。banner、边界声明块、明细表三处的文案随它走。这个标记同样用于 GLM-5.3 的官方 BF16 checkpoint。

## 不建模

- `embedding_across_dp`（DSv4.1）：介于两档之间，要加每步一次 DP 间集合通信。写在 DSv4.1 的 `lookupHbmCaveat` 里。
- DEP（TP1 × DP）必须 offload（Qwen3.8-Flash-Next）：写在 notes 里，页面不出红字。
- host 内存容量与 PCIe 预取带宽。

## 后果与验证

DeepSeek-V4.1-Flash 的数字（128K / util 0.90 / 原生 FP4）：

| 切分（p5en ×1） | 原来（常驻） | 现在（默认 host） |
|---|---|---|
| TP8/DP1（官方 MP=8） | 488 | 697 |
| TP4/DP2 | 538 | 1,370 |
| TP2/DP4 | 0（装不下） | 2,648 |
| TP1×DP8 | 0（装不下） | 4,928 |

结论方向翻转：放 host 时 TP 越小越好（KV 只有 890 B/token，复制它几乎不疼），与 GLM-5.3-Flash 同向；常驻时 TP 有下限，最优 TP 是 4。两种结论都留在页面上，切换输入就能看到。

`verify-app.js`：

- 原来 7 条 DSv4.1 用例全部显式设成 `lookupPlace: "hbm"`，期望值**逐字未改**。常驻那条代码路径（÷TP 不 ÷PP、TP 下限、装不下的红字、建议器里的 engram 代价）仍有用例钉住。
- 新增 3 条 DSv4.1 用例，用默认口径。期望值来自独立手算。
- 新增断言：控件只在带第三桶的模型上显示；两档各有自己的 banner；常驻时决策层的首选动作是放回 host。
- K3 / GLM-5.3-Flash / Qwen3.8-27B 的断言逐字不变。

## 仍未解决

- 常驻那一档的真实开销（§4）。要在 H100 上起一次服务、读启动时的显存峰值才能定。
- host 那一档对 decode 的影响。vLLM 说预取能藏在前几层的计算后面，本工具没有验证。
