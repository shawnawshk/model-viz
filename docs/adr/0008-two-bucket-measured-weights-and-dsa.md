# ADR-0008：权重改为「两桶实测字节」，并引入 DSA family

- 状态：**Accepted → 已在 `app.html` + 两个 data 文件实施**（2026-09-09）
- 触发：收录第二个模型 GLM-5.3-Flash（`zai-org/GLM-5.3-Flash`）
- 影响：data 契约的 `weights` 块（**不兼容改动**）、新增 `dsa` family、新增 `notes` / `defaultParallel` / `hidden` 字段
- 相关：[ADR-0002](0002-page-per-model.md)（单引擎 + data 契约）、[ADR-0004](0004-provenance-first-class.md)（provenance）、[ADR-0007](0007-kv-dtype-is-an-input.md)（同类的「引擎参数 ≠ 模型属性」）

## 触发原因：第一个模型的口径撑不住第二个模型

原来的 `weights` 契约是：

```js
weights: { bf16Params, f32Params, totalBytes }
// 引擎:nonExpertBytes = bf16Params×2 + f32Params×4
//       expertBytes    = totalBytes − nonExpertBytes      ← 残差
```

这在 K3 上成立，且成立得很干净：K3 的 `quantization_config` 把**所有**非 expert 模块都排除在量化之外，所以「非 expert = BF16 + F32」是全等式，残差恰好就是纯 routed expert。

GLM-5.3-Flash 不是这样。它的非 expert 部分**本身就是混合 dtype**：

| 非 expert 里的模块 | dtype | 参数量 |
|---|---|---|
| 11 个 DSA 层的 `q_a / q_b / kv_a / o_proj`、shared expert、前 3 层 dense MLP | **FP8 e4m3** | 2,743,074,816 |
| 34 个 KDA 层的全部投影、`kv_b_proj`、indexer、mHC 的 `hc_*`、embedding、lm_head | BF16 | 6,926,096,640 |
| norms / bias / `A_log` / `dt_bias` | F32 | 295,518 |

用旧口径算，那 27.4 亿个已是 FP8 的非 expert 参数会被残差**误并进 expert 桶**，得出 `expertBytesPerParam = 1.033` —— 一个不对应任何真实量化格式的数字。而这两桶的切法完全不同（expert 按 EP 切，非 expert 按 TP 切、在 DP 上复制），归错桶就是算错显存。

## 决议

### 1. `weights` 改为两桶实测字节 + 两个参数量

```js
weights: {
  totalBytes,            // 全部张量字节(= index.json 的 metadata.total_size)
  expertBytes,           // routed expert 桶
  expertParams,          // routed expert 逻辑参数量
  nonExpertParams,       // 其余参数量
  nonExpertBf16Params,   // 其中以 BF16 存储、能被「非 expert 权重」开关改成 FP8 的部分
  nonExpertFixedBytes,   // 其余非 expert 字节(F32 + 原生 FP8),开关不动它
}
```

不再有残差推导。K3 的数字**一个都没变**（`verify-app.js` 里 69 / 136 路的断言原样通过）—— 那些值原本就写在它的注释里，这次只是从注释提升为字段。

引擎在加载时**硬断言**两条闭合关系，写错一位数字立刻白屏而不是静默出错：

- `nonExpertBf16Params × 2 + nonExpertFixedBytes === totalBytes − expertBytes`
- `expertParams ÷ (experts × 3 × latent × inter)` 必须是整数（= 带 expert 权重的层数）

### 2. 参数量排除 `*_scale_inv`，字节数包含它

这样「每参数字节数」才是**存储格式的指纹**，可与理论值对账：

| 模型 | 实测 B/参数 | 理论值 | 吻合 |
|---|---|---|---|
| Kimi K3 | `0.53125` | MXFP4：4 bit + 每 32 元素一个 8 bit scale = 0.5 + 1/32 | ✓ 精确 |
| GLM-5.3-Flash | `1.000244140625` | FP8 e4m3：1 B + 每 128×128 块一个 F32 scale = 1 + 4/16384 | ✓ 精确 |

这个口径与 HF `/api/models/<id>` 的 `safetensors.parameters` 完全一致（GLM 两边都是 `321,323,031,390`），所以它同时是一条免费的交叉校验。

### 3. GLM 的权重取值方式：逐张量读 header，不用残差

对 62 个 shard 各发两个 Range 请求（前 8 字节拿 header 长度，再拿 header JSON），把 76,108 个张量的 `dtype / shape / data_offsets` 求和。结果与 `index.json` 的 `metadata.total_size` **精确相等**（`328,326,771,576`）。

**取 `index.json` 必须走 `resolve` 端点**，`raw` 只返回 git-lfs 指针 —— 与 K3 时的坑相同。

### 4. MTP 层算进显存

GLM 的 `num_nextn_predict_layers = 1`，checkpoint 里第 45 层带一整套完整的 288 个专家（6.75 GiB）。**算进 expert 桶**，与 K3「整个 checkpoint 全算」的口径一致。

后果是 `expertParams` 对应 **43** 层而 `moe.layers` 是 **42**，两者故意不同：

- `expertParams` 决定显存 → 43 层（MTP 那层的专家也要占卡）
- `moe.layers` 决定 all-to-all 次数 → 42 层（MTP 是额外一次前向，不在主干 45 层里）

引擎因此派生 `M.expertWeightLayers`，假设区显示的是这个派生值，不是 `moe.layers`。不开投机解码时可省 6.75 GiB，写在该模型的 `notes` 里。

### 5. 新增 `dsa` family：稀疏注意力不省显存

DSA（DeepSeek Sparse Attention）= 同样的 MLA latent + lightning indexer 自己的一份 key cache。两件事必须在页面上说清：

**（a）稀疏只省算力，不省显存。** `index_topk = 2048` 决定的是每个 query 去**看**多少 token，不是**存**多少。全部 token 的 latent 依然要留在 cache 里，显存与 full MLA 一模一样。省下的是 attention FLOPS 与 KV 读带宽 —— 而这两样按 [ADR-0003](0003-scope-memory-feasibility-only.md) 都不在本工具范围内。这一条写进了该模型的 `notes`，因为「稀疏应该更省显存」是读者最可能带进来的误解。

**（b）indexer 的 key cache 走自己的 dtype。** 它不受 `--kv-cache-dtype` 影响，所以不能塞进「元素数 × dtype 字节」的换算里。`FAMILY` 表新增可选的 `perTokenFixedBytes(g)`，直接返回字节：

```js
kvBytesPerToken = M.kvElemsPerToken × kvDt.bytes + M.kvFixedBytesPerToken
```

反事实（换另一档 KV dtype）也走同一个式子，否则两个方向不对称。

### 6. indexer key cache 的取值：计入并标 `估算`，而不是省略

`wk` 输出 128 维，`index_kpool = 4` 且 `index_kpool_compress = true`。本页按「4 个 token 池化成 1 个 key」+ FP8 计：**32 元素/token/层 → 11 层合计 352 B/token**，约 MLA latent 的 **+3%**。若引擎实际不做池化压缩（128 元素/token/层）则是 **1,408 B/token**，约 **+13%**。

两个读数都写在 `notes` 里。**选择「计入」而不是「省略」**：省略会让并发被高估 3%–13%，且页面上不会有任何色块提示这一项存在 —— 那正是 [ADR-0007](0007-kv-dtype-is-an-input.md) 记录过的失效模式。计入则至少有一个带 `估算` 标记的可见项，读者知道该去质疑什么。

### 7. 顶部边界声明块不再声明假设的**数量**

原文案写的是「但有**两个** dtype 是假设」。GLM 多出 indexer 这第三项后，这句封闭断言会把它盖掉 —— 与 ADR-0007 里那句「**仅** state dtype 是假设」是同一个错。

改为动态列举，并显式写明**清单不保证已穷举**。GLM 页面因此列 3 项，K3 页面列 2 项，加第三个模型时不需要再改引擎。

### 8. 引擎里剩下的 K3 字面量全部抽走

第二个模型暴露出引擎里还有一批写死的 K3 数字：通信面板标题的「93 层」、PP banner 的「24 个 MLA 层」、KV 表格的「24 MLA 层 × (512+64)」、假设区的「92 层 × 896 experts × 3 × 3584 × 3072」，以及三整条 K3 专属的溯源段落。

- 能从 `MD` 派生的（层数、family 名与计数、expert 结构）→ 派生，新增 `KVLAYERS` / `kvFamilyName` / `kvLayerDesc`
- 属于某个模型自己的一手件溯源 → 新增顶层 `notes: [html]`，渲染进假设区
- 起始并行切分 → 新增 `defaultParallel`（K3 要 4 台起，GLM 一台就够，原先 `n:4, tp:8, dp:4` 写死在引擎里）
- `hidden_size` → 新增 `hidden`。引擎的 PP send/recv 那一行本来就在读 `M.hidden`，但从没有人喂过它，`PP > 1` 时那格显示 `NaN KB`。K3 的探针从不设 PP，所以两个月没被发现

### 9. 预设按钮上的路数必须现算,且名字里不写绝对数

第一版把路数写进了预设名字(`推荐 TP1×DP8/EP8(325 路)`）。这是错的:那个 325 是按 `128K / util 0.90 / BF16 KV` 算的，而**点预设时 util、上下文、两个 dtype、两个权重格式一律不重置**（有意为之 —— 你把上下文拖到 1M 就是想在 1M 下比较几个切分，预设不该把它拽回去）。于是拖过滑块之后，按钮上标着 325 而页面显示 40，两个数都在屏幕上，互相打脸。

改法：

- 路数从数据文件里删掉，改由 `markPreset()` 调 `presetCompute(p)` **按当前口径现算**，写在按钮第三行，并直接标注口径：`320 路 @128K/0.90`。装不下时显示 `装不下 @…` 并转红。
- `markPreset()` 挂到 util / 上下文滑块的 `input` 上（原先只有下拉框会触发它）。于是拖滑块时六个预设的路数整组联动 —— **预设列表顺带变成了一张对比表**，这是意外收获。
- hover title 给全口径：卡数、切分、util、上下文、三个 dtype/格式、最大并发 + 换另一档 KV dtype 的反事实、单卡权重+overhead / 预算。
- `PRESETS` 过滤新增 `TP×DP×PP === 卡数` 校验。既然数字是现算的，写错的预设会算出一个看不出错的数，所以要在加载时喊出来。

**顺带撤掉了「推荐 / 折中」这类价值标签。** 按 [ADR-0005](0005-validator-not-solver-and-instance-list.md)，这是校验器不是求解器，标「推荐」本就越界；更实际的问题是那个排序**本身随口径变**。原生 FP8 下四个角实测（六个预设依次）：

| 口径 | TP8/DP1 | TP1×DP8 | TP2×DP4 | PP2 | 两台 | b300 一台 |
|---|---|---|---|---|---|---|
| 1K / 0.70 | 1,751 | 1,936 | **2,200** | 2,372 | 5,904 | 6,904 |
| 1K / 1.00 | 3,279 | 4,288 | **4,384** | 4,724 | 10,592 | 11,384 |
| 128K / 0.90 | 53 | **320** | 188 | 364 | 832 | 912 |
| 1M / 1.00 | 7 | **48** | 28 | 56 | 128 | 136 |

**1K 上下文下 TP2×DP4 反超 TP1×DP8** —— 短上下文时 KV 复制的代价小，而 TP 能切非 expert 权重与 recurrent state，反倒占优。TP1×DP8 的领先幅度从 6.1× 缩到 1.11×。再把 expert 切成 BF16（580 GiB）会直接翻转：TP1×DP8 装不下，TP8/DP1 还剩 1 路。

所以 GLM 的预设名字只描述**结构**，括号里标该切分下 KV 被复制几份（`TP1×DP8(KV ×1)`）—— 这是差异的机制，恒真；排序交给现算的数字。相对比较（`b300 一台 ⇄ p5en 两台`）可以留在名字里，因为它稳健得多（四个角上都成立，幅度 1.07–1.17×）。

K3 的预设名字**没动**：它的「推荐 TP8×DP4」在 128K/0.90 下是 68 路，而「方案 2 TP8/PP4」是 81 路 —— 看似矛盾，但那个「推荐」包含了流水线气泡这类本页不算的因素，PP banner 里已经写明。这是决策时点的既有判断，按惯例不回改。

## 后果与验证

- `verify-app.js` 从「单模型 + 两档 KV dtype」扩为**用例表**：每个模型至少一个，且**必须有一个 `PP > 1` 的用例**；`NaN / undefined` 检查从 `tiles`/`tbl` 扩到 `commtbl`/`assumplist`/`banners`；并断言 `index.html` 与 `app.html` 挂的 data 文件列表一致。
- 新增 `expectPresets`：钉住预设按钮现算出来的**整组**路数。GLM 特意留了同一组切分的**两个口径**（`128K/0.90` → `[53,320,188,364,832,912]`，`1M/1.00` → `[7,48,28,56,128,136]`），用来证明这组数真的跟着 util / 上下文走 —— 别把它们合并掉。K3 那组 `[20,81,68,0,76,108]` 里的 `68` 与 `108` 与 ADR-0007 和数据文件注释里独立记下的数一致；`0` 是「反例 纯 DP32」本来就装不下（单卡 160.6 GiB，连 141 GiB 物理显存都超，与 util 无关）。
- 8 个用例的期望值都是先手算、再与引擎对账定下的。额外的 b300 1M/0.91 用例钉住 DP 整数装箱：连续平均上限是 125.9 路，但 8 个 DP rank 每个只能完整容纳 15 路，所以实际是 120 路。
- `index.html` 的模型卡片改用 `totalBytes − expertBytes`，不再自己重算一遍非 expert。

## 仍未解决

- **`overhead` 还是常数**，每卡 12 GiB 仍不可调。第二个模型没有改变这一点，见 CLAUDE.local.md 的待做项。
- **indexer key cache 的两个读数都不是一手依据。** 消除办法与 ADR-0007 的 KV dtype 相同：在目标机型上起一次服务，读引擎自报的 block 数反解。
- **GQA family 仍未落地。** 届时 KV 复制因子是 `max(1, TP/n_kv_heads)`，是并行配置的函数，`FAMILY` 表的 `perTokenElems(g)` 签名不够，需要传入当前切分 —— 与本 ADR 新增的 `perTokenFixedBytes` 是两件事，不要混。
