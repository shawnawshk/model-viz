# ADR-0012：GQA family（KV 复制因子是并行配置的函数），dense 模型（没有 expert 桶）

- 状态：**Accepted → 已在 `app.html` + `index.html` + 第四个 data 文件实施**（2026-09-16）
- 触发：收录第四个模型 Qwen3.8-27B（`Qwen/Qwen3.8-27B`）
- 影响：`FAMILY` 表加 `kvShards(g, tp)` 钩子与 `gqa` family；data 契约的 `moe` 块改为**可选**（缺省 = dense）；`nonExpertFormats` 允许 `dtype` 字段；新增 `nonExpertFixedDesc / nonExpertFixedDtype / weights.nonMatmulParams`；候选机型第一次包含 G 系
- 相关：[ADR-0001](0001-scale-up-domain-not-node.md)（NVLink 域 = TP 上限，本 ADR 让它第一次实际起作用）、[ADR-0007](0007-kv-dtype-is-an-input.md)（KV dtype 是输入）、[ADR-0008](0008-two-bucket-measured-weights-and-dsa.md)（GQA 待办的来源）、[ADR-0009](0009-third-weight-bucket-and-csa2.md)（§4 与「仍未解决」里把 GQA 与 `csa2` 划清界线）、[ADR-0010](0010-decode-roofline-lower-bound.md)（roofline 的 KV 读取也要按同一规则切）

## 触发原因：第四个模型没有 MoE，而它的 KV 复制因子随 TP 变

Qwen3.8-27B 的一手件对账（18 个 shard 逐张量读 header，1,199 个张量，字节合计 `55,562,855,904` 与 `index.json` 的 `total_size` **精确相等**；参数 `27,781,427,952` 与 HF API 的 `safetensors.parameters.BF16` **精确相等**；全部 BF16，零 F32）：

| 项 | 值 | 与前三个模型的差别 |
|---|---|---|
| 结构 | 64 层 = 48 Gated DeltaNet（线性）+ 16 Gated Attention（**GQA**，24 q / 4 kv head，head_dim 256） | 前三个的 full-attn 层全是 MLA 系（latent 共享） |
| MoE | **没有**，27.8B 全部激活 | 前三个全是 MoE，引擎处处假设有 `moe` 块 |
| 权重 | 51.75 GiB BF16；官方另有 FP8 checkpoint 28.75 GiB | 一张卡就装得下 |
| 每 token KV | **65,536 B**（BF16）—— 四个模型里最大，是 K3 的 2.4 倍、DSv4.1 的 74 倍 | 但**复制因子不是 TP** |
| 每请求 state | 144.0 MiB（FP32），与上下文无关 | 与 GLM / K3 同类，但 dtype 有两个官方答案（§4） |

引擎里有两条隐含假设在这个模型上同时不成立：

1. **每个模型都有 `moe` 块。** `M.nExperts / topk / moeLayers / expertWeightLayers / expertBytesPerParam` 无条件计算，EP 下拉、expert 权重下拉、expert 色块、专家徽章、all-to-all 行、roofline 的命中率与「命中专家的矩阵乘」全部无条件渲染。没有 `moe` 就是 `0 ÷ 0`，NaN 会一路流到页面上。
2. **每 token 的 KV 在 TP 组内每卡一份完整副本。** `kvPerReq = ctx × kvBytesPerToken ÷ PP`，没有 ÷TP —— 这对 MLA / DSA / CSA2 是对的（latent 被所有 head 共享），对 GQA 是错的：KV 按 kv head 切，TP ≤ n_kv_heads 时每卡只存 `n_kv/TP` 个 head，TP 组合起来恰好一份；TP 超过 n_kv_heads 后每卡至少 1 个 head，才开始复制。**复制因子是 `max(1, TP ÷ n_kv_heads)`，是并行配置的函数**。ADR-0008 / 0009 都把这一条留成待办，理由相同：`perTokenElems(g)` 的签名里没有 TP。

按旧口径算，Qwen 在 TP=4 时的单卡 KV 会被高估 4 倍，最大并发低估 4 倍；TP=8 时高估 4 倍（真实复制 2 份，旧口径按 8 份）。这不是几个百分点的误差，是结论翻转的量级。

## 决议

### 1. `FAMILY` 加 `kvShards(g, tp)` 钩子，新增 `gqa` family

```js
gqa: { perTokenElems: g => 2 * g.kvHeads * g.headDim,      // K + V,所有 kv head 合计 = 「一份」
       kvShards:      (g, tp) => Math.min(tp, g.kvHeads),  // 单卡份额 = 一份 ÷ kvShards
       decodeReadPerReq: (g, ctx, kvB) => ctx * 2 * g.kvHeads * g.headDim * kvB,   // 调用方再 ÷ kvShards
       kvDesc / kvTpDesc / decodeReadDesc: ...,
       stateShardableByTp: false }
```

| family | `kvShards(g, tp)` | TP 组内复制份数 = `tp ÷ kvShards` |
|---|---|---|
| mla / dsa / csa2（latent 共享） | 无钩子 = **1** | TP |
| **gqa** | `min(TP, n_kv_heads)` | `max(1, TP ÷ n_kv_heads)` |
| linear / swa | 不持有随上下文涨的 KV | — |

引擎侧新增一个 helper，**显存侧与 roofline 侧共用同一条规则**：

```
kvBytesPerTokenOnGpu(kvB, tp) = Σ_g count × (perTokenElems(g) × kvB + perTokenFixedBytes(g)) ÷ kvShards(g, tp)
```

用在四处：`compute()` 的 `kvPerReq` 与反事实 `perReqAlt`、决策层的 `capacityAt()`、`roofTerms()` 的 KV 读取、`renderRoof()` 的 KV 行。`M.kvElemsPerToken × dtype + kvFixedBytesPerToken` 仍然是「一份」的字节数，继续给 `B/token/份` 标签与集群存量用；`compute()` 多返回 `kvBytesPerTokenGpu` 与 `kvCopies`。

**K3 / GLM / DSv4.1 的输出一个字节都不变**：它们的 family 没有 `kvShards`，helper 退化为原式。`verify-app.js` 里 14 条老断言逐字通过。

页面文案随 `kvCopies` 走：hero 里「KV 每 token 被复制 TP 份」在 GQA 上改成「复制 `max(1, TP/n_kv)` 份」或「按 kv head 切开，TP 组内不复制」；决策层 `parallelCard` 的「跨卡副本从 ×a 降到 ×b」也按 `kvCopies` 算（GLM 那条 `×8 → ×1` 的断言不变）。两条新 banner：TP > n_kv_heads 时 `warn`「KV 每 token 复制 N 份 …… 这个模型 TP 的甜点在 n_kv_heads 上」；1 < TP ≤ n_kv_heads 时 `good`「KV 不复制」。

### 2. `moe` 块改为可选，缺省即 dense

```js
const MOE = MD.moe || { layers: 0, experts: 0, topk: 0, sharedExperts: 0, latent: 0, inter: 0 };
M.dense = !MD.moe;
```

dense 时引擎做的事，逐项列出（这些都是「无条件渲染」的旧代码，每一处都会漏出 `0 ÷ 0`、`专家 #0–-1` 或「expert bank 被复制了 8 份，白占 0 GiB」这类东西）：

| 处 | dense 分支 |
|---|---|
| EP 候选 | `epOptions(stage) = [1]`，EP 控件整个隐藏（新加 `#epCtl`） |
| expert 权重下拉 | 隐藏（新加 `#expFmtCtl`）；`expertFormats` 不允许存在（断言） |
| 「非 expert 权重」标签 | 改叫「权重」（`NE_LABEL`，新加 `#precName`），series / tile / banner / roofline 行同步 |
| expert series | 整条不进 `SERIES`：图例、表格、色块都不留一个 0 行 |
| 断言 | 不再要求 `expertParams ÷ 结构` 是整数，改为要求 `expertBytes = expertParams = 0` |
| banner | 「expert bank 被复制」「EP 跨域」不出现；DP>1 那条改写成「dense 模型，全部参数复制 N 份 …… DP 的账只有一项」 |
| GPU 徽章 | 不显示 EP / 专家编号 |
| 通信表 / roofline 通信行 | 「MoE all-to-all」改写为「dense 模型，没有 MoE 层」 |
| roofline | `coverage = 0`、`expRead = 0`；「命中的 routed expert」「命中专家的矩阵乘」两行与「均匀路由期望」那条口径都不出现 |
| 假设区 | 「routed expert 逻辑参数量」那条不出现 |

`index.html` 的模型卡同步：dense 不显示「其中 expert」行，「非 expert」改叫「权重（dense，无 MoE）」，`quantFormat = "bf16"` 显示为「无（BF16）」。

### 3. 「非 expert → FP8」那一档第一次是实测；两个字段的切分线重新定义

K3 / GLM 的 FP8 档是**假设重量化**（每参数按 1 B 推算）。Qwen 有官方 `Qwen/Qwen3.8-27B-FP8`，1,606 个张量同样逐个读了：

| 部分 | 参数 | 存储 | 字节 |
|---|---|---|---|
| 量化的 407 个张量（64 层 MLP、16 层 attention 投影、48 层 GDN 的 qkv/z/out、MTP 的同类） | 24,699,207,680 | F8_E4M3 + **BF16** `weight_scale_inv`（block 128×128） | 24,702,222,720 |
| 保留 BF16 的 792 个（embed、lm_head、vision 塔、全部 norm、GDN 的 A_log / dt_bias / conv1d / in_proj_a / in_proj_b） | 3,082,220,272 | BF16 | 6,164,440,544 |
| 合计 | 27,781,427,952（与 BF16 checkpoint 同一总数，HF API 亦如此） | | **30,866,663,264 = 28.75 GiB** |

量化部分每参数 `1 + 2/16384 = 1.0001220703125` B，精确 —— scale 是 BF16 而不是 GLM 的 F32（GLM 是 `1 + 4/16384`），这个指纹能区分两家的量化脚本。

于是 `weights.nonExpertBf16Params / nonExpertFixedBytes` 在本模型上的切分线**不是**「BF16 vs F32 / 原生 FP8」（本 checkpoint 没有 F32），而是「官方 FP8 量化了哪些 / 保留了哪些」：`nonExpertBf16Params = 24,699,207,680`、`nonExpertFixedBytes = 6,164,440,544`。闭合断言 `× 2 + fixed = totalBytes` 照常成立；FP8 档 `bf16Bytes = 1.0001220703125` 使 `nonExpertBf16Params × bf16Bytes + fixed` **精确等于** FP8 checkpoint 的逐张量求和。

两个字段的**语义**没变（「开关能动的部分 / 开关不动的部分」），变的是这条线在本模型上落在哪里。data 文件注释里必须写明，否则下一个人会以为 `fixedBytes` 一定是 F32。

配套三个可选字段：

- `nonExpertFormats[].dtype`：roofline 算力行原先用 `bf16Bytes === 1` 判 FP8，实测的 `1.0001220703125` 判不出来，允许显式声明
- `nonExpertFixedDtype`：固定部分缺省按 FP8 算力行（K3 / GLM / DSv4.1 的原生 FP8 模块），Qwen 保留的那 3.08B 是 BF16，要按 BF16 行
- `nonExpertFixedDesc`：series 的 pvNote 与 roofline 行里那个括号，缺省「F32 norms + 原生 FP8 模块」

`nonExpertFormats` 的 FP8 档 label 写「官方 FP8 checkpoint，实测」，边界声明块里「权重字节数是实测值」这句在两档下都成立。**不是「一半」**：6.16 GiB 官方也没量化，FP8 只省 44%，页面上要能看出来。

### 4. GDN state 的 dtype 有两个官方答案，本页取 config 的

`config.json` 写 `mamba_ssm_dtype: "float32"`，HF 参考实现按此分配。但 vLLM 的 `--mamba-ssm-cache-dtype` 缺省 `auto`，对 Gated DeltaNet 这条路（`MambaStateDtypeCalculator._mamba_state_dtype`）`auto → conv state dtype → 模型 dtype = BF16`；而 KDA 那条路（`kda_state_dtype`，K3 / GLM）`auto` 反而是 float32 —— 同一个引擎、同一个参数、两种线性注意力的缺省相反。

决议：**按 FP32 计**（`stateDtypeBytes: 4`）。理由：(a) 一手件是 config；(b) 与 K3 / GLM 同口径，四个模型的并发才能并排比；(c) 保守方向。反事实写进 `notes`：vLLM 不传参就是 72 MiB/请求。

敏感度**不是**「不到 2%」——那句原来写死在假设区里，对 GLM 的 128K 是对的，对任何模型的 1K 都不对。本 ADR 把它改成现算：`1 ÷ (1 − state/2 ÷ perReq) − 1`，Qwen 1K/TP8 下是 36%，TP ≤ 4 下 53%（state 与 KV 的 ÷TP 规则不同，比值随 TP 变）；128K 下不到 1%。同一条里 state 与 KV 相等的临界上下文公式也从 MLA 的 `X ÷ TP` 改成 `X ÷ TP × min(TP, n_kv_heads)`（MLA 系 `min(TP, ∞)` 退化为原式）—— Qwen TP8 下是 1,152 tokens，旧式会算出 288。

顺带一条实测印证（vLLM recipe，1× RTX 5090 @32K）：KV 切 FP8 只把池子从 76,458 涨到 91,022 tokens，1.19× 而不是 2× —— 因为 state 不受 `--kv-cache-dtype` 影响却与 KV 共用同一个池。这与本页「短上下文下 state 主导」的推论方向一致。

### 5. 候选机型第一次包含 G 系与 H100 80GB；ADR-0001 第一次实际起作用

前三个模型的候选都是同一组 4 个 P 系（便于并排）。Qwen 51.75 GiB 一张卡就装得下，真正会部署它的是 H100 80GB 与 G 系，所以在 4 个 P 系之外加了 3 个（**业务判断，2026-09-16 由本次会话拟定，尚未逐个与用户确认，可删**）：

| 机型 | 128K / 0.90 / BF16 KV | 说明 |
|---|---|---|
| p5（H100 80GB，NVSwitch） | TP4×DP2 **46** 路；TP1×DP8 只有 8 路 | 权重 + overhead 占预算 35% |
| g7e（RTX PRO 6000 96GB，**无 NVLink**） | TP1×DP8 **16** 路 | 域 = 1，TP 上限 1，每卡背满 51.75 GiB |
| g6e（L40S 44.7GB，**无 NVLink**） | BF16 **装不下**（51.75 + 12 > 40.2）；官方 FP8 在 util 0.90 下仍差 0.5 GiB，0.95 才装下，1K 下 64 路 | TP=2 显存够，但要跨 PCIe |

ADR-0001 写的「显存装得下不代表能用」在前三个模型上从未被触发（它们根本不会进 G 系）。g6e 是第一个：TP=2 能把 51.75 GiB 切成两半塞进去，但域 = 1。「装不下」banner 因此在 `p2p.type === "pcie" && TP === 1` 时多说一句「卡间没有 NVLink，TP 上限是 1，权重切不开 —— 出路只有更小的权重格式或更大显存的卡」。

g6e 也是整页里唯一由 12 GiB overhead 猜测直接决定生死的机型（FP8 权重 28.75 + 12 = 40.75 对 40.23 的预算）。留着它就是为了把这件事摆出来。

### 6. TP 与 head 数的整除检查（只对声明了 head 数的层组）

24 个 q head、4 个 kv head、GDN 的 16 个 k head / 48 个 v head → 合法 TP 只有 `1 / 2 / 4 / 8`。TP=16（两台）被 `24 % 16 ≠ 0` 挡住，vLLM / SGLang 会拒绝启动。

新增 `headSplitIssues(tp)`，不整除时出 `critical` banner，并列出当前卡数下合法的 TP。**同一条约束也进 ADR-0011 的枚举器**：`parallelAlternatives()` 过滤掉不整除的 TP，否则 3 台（stage 24）上会推荐 TP6/DP4、5 台上 TP5 会进候选 —— 告警说起不来，卡片却让你「应用到页面」。这一条是 PR review 抓出来的：第一版只把检查接进了 banner。顺带修了 `verify-app.js` 探针里的一个旧问题：`insightHtml` 原来在 `render()` 之前读，拿到的是上一次渲染（首轮是页面默认状态）的内容，延迟目标那一栏的断言此前并没有对着用例状态查。**只对声明了 `qHeads / kvHeads`（gqa）或 `kHeads`（linear）的层组检查** —— K3 / GLM 的 linear 层组没有 `kHeads`，这条对它们永远为空，输出不变（否则 GLM 在 3 台 / TP=3 下会多出一条正确但此前没有的告警，那属于另一个改动）。

### 7. 预设：第四种方向 —— TP 的甜点在 n_kv_heads 上

前三个模型的故事：K3 要 TP（非 expert 106.5 GiB，DP 复制它很贵）；GLM 要 DP（TP 复制 latent）；DSv4.1 的 TP 有下限（engram 只能按 TP 切）。Qwen 是第四种：**TP ≤ 4 时 KV 按 head 切、不复制，TP 同时还切权重，所以 TP 越大越好；TP=8 越过 4 个 kv head，KV 复制 2 份、DP 少一半，并发掉回 TP1 的水平。**

手算（BF16 KV、原生 BF16 权重、每卡 12 GiB overhead）：

| 口径 | TP8/DP1 | **TP4×DP2** | TP2×DP4 | TP1×DP8 | p5 TP4×DP2 | g7e TP1×DP8 |
|---|---|---|---|---|---|---|
| 1K / 0.90 | 3,265 | **4,014** | 3,504 | 2,480 | 1,852 | 888 |
| 128K / 0.90 | 53 | **100** | 84 | 56 | 46 | 16 |
| 1M / 0.90 | 6 | **12** | 8 | 0 | 4 | 0 |

TP4×DP2 在三个口径下都是 p5en 上最高的一档，b300 / b200 / p5 上也是（b300：168 / 200 / **212** / 110）。但按 ADR-0008 §9 的规矩，名字里只写机制（括号里标 KV 被复制几份），不写路数、不写「最优」。

`defaultParallel` 取 TP8/DP1 —— 8 卡机器上的反射动作。页面一打开就有「KV 复制 2 份」那条告警，点旁边的 TP4×DP2 看并发接近翻倍。与 GLM 页同一套路，方向不同。

### 8. 顺带修的四处旧代码

都是被本模型第一次触发、或本模型让它明显错了的：

- `SERIES[1].short = "routed expert 权重"` 按下标赋值，expert 条目可能不存在后改为按 `key` 查找。
- roofline 口径里「参数量含 embedding 与 vision 塔，高估不到 3%」写死；Qwen 的 embed（1.27B，查表）+ vision（0.46B，decode 不跑）是 6.2%。data 加可选 `weights.nonMatmulParams`，有则现算比例，无则保留原句。**只标注，不从 FLOPs 里扣**：扣了就与另三个模型口径不一致，ADR-0010 选的是「含 embedding，高估」，四个模型保持同一方向。
- 假设区 state 那条的「不到 2%」与临界上下文公式，见 §4。
- roofline KV 行的「latent 是所有 head 共享的，TP 内每卡读一份完整副本」改为 family 提供 `kvTpDesc`，缺省沿用原句。

**没有动的**：MTP 层的 KV 未计（开投机解码时 draft 层也是一层 GQA，每 token 多 1/16），口径与 GLM 的 MTP 处理一致（权重算、KV 不算），差额写 `notes`。

## 后果与验证

`verify-app.js` 从 14 个用例扩到 **24 个**，新增 10 条全是本模型，期望值由独立手算脚本给出（输入抄自 config / safetensors / datasheet，按 ADR-0008 / 0009 / 0010 的式子），再与引擎对账 —— **显存侧全部精确一致，roofline 三位有效数字一致，临界上下文在 ±1% 内**：

| 用例 | 钉住的是什么 | 路数 |
|---|---|---|
| `1×p5en TP8/DP1` @128K | 65,536 B/token；KV ÷4 不是 ÷8（复制 2 份）；整组预设 `[53, 100, 84, 56, 46, 16]`；roofline 16.0 ms | 53 |
| `1×p5en TP4/DP2` | TP = n_kv_heads 的甜点；容量建议从 TP8 出发要能找到它、且副本文案是 `×2 → ×1` | 100 |
| `1×p5en TP1/DP8` · 1 路 | 每卡背满权重；`expectLoad`（1 路只落一个 rank，峰值 71.9 / 最低 63.7 GiB） | 56 |
| **PP2** `TP4/DP1/PP2` | PP 分支渲染；KV 与 state 都 ÷PP | 106 |
| **1K** `TP8/DP1` | GDN state 主导（单卡 16 MiB KV + 18 MiB state）；预设整组 `[3265, 4014, 3504, 2480, 1852, 888]` | 3,265 |
| `TP8/DP1` · **官方 FP8 权重** | 实测 28.75 GiB 那一档；roofline 算力行按 FP8 + BF16 两段 | 55 |
| `1×g7e TP1/DP8` | 域 = 1 上 TP=1 不出跨域告警；HBM 1597 GB/s → 40.4 ms | 16 |
| `1×g6e TP1/DP8` BF16 | **装不下**，banner 点出「TP 上限 1，权重切不开」 | 0 |
| `1×g6e` FP8 · 1K · util 0.95 | 只差 0.5 GiB 的那条线：0.95 装下、0.90 装不下 | 64 |
| `3×p5en TP4/DP6` | 建议器不得推荐 TP3/6/12/24（§6）；延迟 / 容量两个目标下的候选 TP 都必须在 {1, 2, 4, 8} 内 | 300 |

dense / GQA 专属断言：图例无 `routed expert`、tiles 无 `expert`、无 expert bank 告警、徽章无专家编号、假设区无 expert 参数量条、roofline 无 expert 两行且通信行写明「没有 MoE 层」、EP 恒为 1、同机型建议里没有 EP≠1 的候选、TP>4 有复制告警 / 1<TP≤4 有不复制 banner、hero 在 TP8 写「复制 2 份」、控件改名与隐藏。

**K3 / GLM / DSv4.1 的 14 条断言逐字不变通过。**

## 仍未解决

- **attention 本身的 FLOPs 在 dense 小模型上更要紧。** 128K 下 `4 × ctx × 24 head × 256 × 16 层 ≈ 51.5 GFLOP/token`，与权重矩阵乘的 55.6 GFLOP **同量级**；MoE 模型的权重项大得多，漏掉这一项相对不显。ADR-0010 的待办在这里从「偏松」变成「差一倍」。regime 不受影响（GDN state 读写的斜率仍压过算力），但算力行的绝对值在长上下文下不可信。
- **GDN state 的 dtype 两个官方答案**（§4），以及未计的短卷积 state（≈2.8 MiB/请求）。消除办法同 ADR-0004 / 0007：在目标机型上起一次服务读引擎自报的 KV block 数。
- **MTP 的 KV** 未计（+1/16）。
- **12 GiB overhead 在小模型上分量更大**（TP8 时是权重的 1.9 倍），g6e 的生死由它决定。vLLM recipe 还记了一件与本页口径相反的事：CUDA graph capture 的分配在 util 预算**之外**，本页把 12 GiB 整个放在预算之内 —— 反解 overhead 时要先对齐这一点。
- **第一组引擎自报数出现了，但用不上。** vLLM recipe（2× RTX 5090 TP2、FP8 权重 14.28 GiB/卡、FP8 KV、262K）自报 KV 池 377,456 tokens。它不在 AWS 机型上，且 vLLM 混合池（attention KV 与 GDN state 共享、按页对齐）的分配规则本页不建模，所以不能直接反解 overhead。`measured` 一级仍空。
- **候选机型列表未经用户确认**（§5）。
- **第三方 NVFP4 checkpoint**（nvidia / Inferact / unsloth，约 24.6 GiB）未收录：不是 Qwen 官方件，且三份的混合精度策略各不相同（vLLM recipe 自己就区分了两种）。若要收，应作为 `nonExpertFormats` 的第三档、逐张量实测。
- **`kvShards` 只覆盖「每卡至少 1 个 head」这种复制。** 若某引擎在 TP > n_kv_heads 时不复制而是做 sequence-parallel 之类的切法，复制因子会不同 —— 目前 vLLM / SGLang 都是复制，先按此。
