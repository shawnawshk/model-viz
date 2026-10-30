# ADR-0014：QSA family（复制的那部分 KV 不按 head 切），DSA 支持 IndexCache；收录 GLM-5.3 与 Qwen3.8-Flash-Next

- 状态：**Accepted → 已在 `app.html` + `index.html` + 两个新 data 文件实施**（2026-10-30）
- 触发：用户要求加三个模型：DeepSeek-V4.1-Flash-1017、GLM-5.3、Qwen3.8-Flash-Next
- 影响：
  - `FAMILY` 表新增 `qsa`；新增钩子 `decodeReadFixedPerReq`；`kvShards` 只切 `perTokenElems` 那部分。
  - `dsa` 层组可带 `indexerScaleBytes`；`indexerElems: 0` 表示 IndexCache 的 shared 层。
  - data 契约新增 `indexerAsm`、`ppUnsupported`；`expertFormats` 的 `tpShardedBytes` / `measured` 见 [ADR-0013](0013-lookup-placement-is-an-input.md)。
  - 没有 recurrent state 也没有滑窗环的模型，不再显示 state 那一档。
- 相关：[ADR-0008](0008-two-bucket-measured-weights-and-dsa.md)（DSA）、[ADR-0010](0010-decode-roofline-lower-bound.md)（decode 读取钩子）、[ADR-0012](0012-gqa-family-and-dense-models.md)（`kvShards`）、[ADR-0013](0013-lookup-placement-is-an-input.md)（查表放置）、[ADR-0005](0005-validator-not-solver-and-instance-list.md) §2（候选机型由用户确认）

## 1. DeepSeek-V4.1-Flash-1017：不另开一页

`deepseek-ai/DeepSeek-V4.1-Flash-1017` 与 `deepseek-ai/DeepSeek-V4.1-Flash` 的对账结果：

- `config.json`、`inference/config.json` 逐字节相同。
- `inference/` 下的 `model.py / convert.py / engram.py / kernel.py / vision.py / README.md` 逐字节相同。
- `index.json` 相同：`total_size` 都是 `510,286,023,000`，96,085 个张量的 shard 映射相同。
- 48 个 shard 的 safetensors header 逐个相同（每个张量的 `dtype / shape / data_offsets`）。

只有权重值变了。显存与 roofline 只看结构与字节数，所以引擎算出的每个数都相同。2026-10-30 用户确认：不新建页面，在 V4.1-Flash 页面的 blurb 与 notes 第一条里注明，首页卡片随 blurb 显示。

## 2. GLM-5.3：78 层全 DSA，IndexCache 让 57 层不存 indexer key

| 项 | 值 |
|---|---|
| 结构 | 78 层全是 DSA（MLA latent 512 + RoPE 64 = 576 元素/token）。不是 GLM-5.3-Flash 那种 KDA 混合 |
| IndexCache | `indexer_types` 有 21 个 `full`、57 个 `shared`。shared 层复用上一个 full 层的 top-k，不建 indexer |
| indexer key | vLLM 的 `DeepseekV32IndexerCache`：`128 + 128 ÷ 128 × 4 = 132 B`（FP8 值 + fp32 scale），不受 `--kv-cache-dtype` 影响 |
| 每 token KV | 92,628 B（BF16）= 78 × 576 × 2 + 21 × 132。六个模型里最大 |
| 权重 | 703.72 GiB，原生 FP8。expert 684.17 GiB（76 层 = 75 主干 + 1 MTP） |

full 层是哪 21 层，三处逐层一致：config 的 `indexer_types`；vLLM 按 `index_topk_freq = 4`、`index_skip_topk_offset = 3` 算出的 `skip_topk`；checkpoint 里带 `indexer.wk` 的层（另有 MTP 层 78）。

引擎改动（不新增 family）：

- `dsa.perTokenFixedBytes = indexerElems × indexerBytes + (indexerScaleBytes || 0)`。GLM-5.3-Flash 没有 `indexerScaleBytes`，输出不变。
- shared 层写成一个 `indexerElems: 0` 的层组。显存只算 latent；decode 只读 top-k 个 latent，不扫 indexer。`decodeReadDesc` 对这种层组单独写一句。

候选机型：P 系 4 个（与 K3 / GLM-Flash / DSv4.1 对齐），2026-10-30 用户确认。起始切分是两台 p5en TP8/DP2/EP16。vLLM recipe 的标准配置是一台 8×H200 TP8，但它在默认口径下只有 2 路，比并发下限 8 还低。这一档留在第二个预设里。

## 3. Qwen3.8-Flash-Next：QSA = GQA 主 KV + 一份按 TP 复制的压缩 indexer key

| 项 | 值 |
|---|---|
| 结构 | 48 层 = 36 Gated DeltaNet + 12 QSA（3:1）。512 experts / top-10，1 shared |
| QSA 主 KV | GQA，24 q / **2** kv head，head_dim 256。`12 × 2 × 2 × 256 × 2 B = 24,576 B/token`（BF16） |
| QSA indexer key | vLLM 的 `QSACompressedKeyCache`：每 4 个 token 一行（`indexer_compress_ratio`），128 维，**只有 1 个 head**。dtype 走 `indexer_kv_dtype`，vLLM 对本模型默认 BF16。`12 × 128 × 2 ÷ 4 = 768 B/token` |
| decode 读取 | indexer 扫全部压缩 key；主 KV 只读 top-2048 个 token（`indexer_budget`）。vLLM 的 QSA 不带滑窗、不带 sink |
| n-gram 表 | 51.2B 参数，95.37 GiB（BF16）。第三桶，vLLM 默认 offload（ADR-0013） |
| 权重 | 335.28 GiB BF16；官方 FP8 172.76 GiB（expert 与 n-gram 表都量化，其余保留 BF16） |

两部分 KV 的切法相反，现有 family 表达不了：

- 主 KV 按 kv head 切，与 `gqa` 相同（`kvShards = min(TP, n_kv_heads)`）。
- indexer key 只有 1 个 head，TP 切不掉，每卡一份完整副本。

### 3.1 `kvShards` 只切元素部分

```
kvBytesPerTokenOnGpu(kvB, tp) = Σ_g count × (perTokenElems(g) × kvB ÷ kvShards(g, tp) + perTokenFixedBytes(g))
```

ADR-0012 原来的式子是 `(elems × kvB + fixed) ÷ kvShards`。旧 family 里，有 `kvShards` 的（gqa）没有 `fixed`，有 `fixed` 的（dsa / csa2）没有 `kvShards`，所以新式子对旧模型逐字节相同。

### 3.2 decode 读取也拆成两部分

新钩子 `decodeReadFixedPerReq(g, ctx)` 是每卡都要读的部分，不除。`decodeReadPerReq` 仍是可按 head 切的部分，调用方除以 `kvShards`。两处调用（`roofTerms` 与 `renderRoof`）改用同一个 helper `decodeReadOnGpu`，与显存侧是同一条规则。

```js
qsa: { perTokenElems:         g => 2 * g.kvHeads * g.headDim,
       perTokenFixedBytes:    g => g.indexerElems * g.indexerBytes / g.compressRatio,
       kvShards:              (g, tp) => Math.min(tp, g.kvHeads),
       decodeReadPerReq:      (g, ctx, kvB) => Math.min(ctx, g.indexTopk) * 2 * g.kvHeads * g.headDim * kvB,
       decodeReadFixedPerReq: (g, ctx) => ctx / g.compressRatio * g.indexerElems * g.indexerBytes, ... }
```

### 3.3 复制因子只按主 KV 数

`compute()` 的 `kvCopies` 改为 `TP ÷ kvShards(主 KV 层组, TP)`，没有 `kvShards` 的模型仍是 TP。原来的式子是加权平均，在 QSA 上 TP=2 会得到 1.03，于是「TP 超过 n_kv_heads」的告警会误报。旧模型上两个式子的结果相同。indexer key 的复制在 banner 与假设区里单独写一句。

### 3.4 其他

- `ppUnsupported`：vLLM recipe 写明 n-gram 嵌入目前不支持 PP。选 PP > 1 时出红字，数字照机械除法算。
- 算力行的高估比例：`nonMatmulParams` 原来除以总参数。Qwen3.8-Flash-Next 的总参数里有 51B 查表，这个比例会显示成 0.6%。改为除以每 token 参与矩阵乘的参数（非 expert + top-k 个 expert × MoE 层数），得 13.8%。Qwen3.8-27B 是 dense，仍是 6.2%。

候选机型：P 系 4 个 + p5（H100 80GB），2026-10-30 用户确认。起始切分是 vLLM recipe 给 8×H200 的 TEP8（TP8 + EP8）。

## 4. 两处顺带的文案修正

- **indexer key 那条假设的后半句按模型各写各的（`indexerAsm`）。** 原来引擎里写死「pooling 方式与 dtype 都没有一手依据，不池化则是 4 倍」。这句话只对 GLM-5.3-Flash 成立，已挪进它的数据文件，页面输出不变。
- **没有 state 的模型不显示 state 那一档。** GLM-5.3 是第一个既没有 recurrent state、也没有滑窗环的模型。原来图例与明细表里会留一行 0，provenance 注释还写着「state dtype 假设为 FP32」。现在与 dense 模型的 expert 桶同样处理：整条 series 不出现。

## 5. 并发下限造成的一处撞红

`CONC_FLOOR = 8` 在 GLM-5.3 的「一台 TP8/DP1（官方 recipe）」预设上撞红：它的上限是 2 路。这是下限造成的唯一一处撞红，刻意保留。按钮上现算出 2 路，点开看到红字，正是这个预设要讲的事。`app.html` 里 `CONC_FLOOR` 的注释已更新。

## 后果与验证

`verify-app.js` 从 23 条用例增加到 43 条：

- GLM-5.3 7 条、Qwen3.8-Flash-Next 10 条，各含一条 PP > 1。期望值全部来自独立手算（按 ADR-0010 §2 与显存侧的定义重写，不读引擎代码）。手算脚本先复现了 DSv4.1 原有的全部期望值，再用于新模型。
- 新增断言：QSA 的合法 TP 只有 1 / 2 / 4 / 8；TP > 2 有复制告警、TP = 2 有不复制的 good banner、TP8 的 hero 写复制 4 份；PP > 1 有「起不来」的红字；官方 checkpoint 那一档写成实测；BF16 → FP8 写成「降到」；GLM-5.3 的图例与明细表里没有 state 那一行。最后一条先在旧代码上确认会失败，再修复。
- K3 / GLM-5.3-Flash / Qwen3.8-27B 的断言逐字不变；DSv4.1 的变化全部来自 ADR-0013。

## 仍未解决

- **MTP 层的 KV** 在两个新模型上都未计（GLM-5.3 约 +1.4%，Qwen3.8-Flash-Next 是一层 QSA 约 +8%）。两者的 recipe 都开了 MTP。
- **vLLM 对 DSA 的 FP8 KV 用 `fp8_ds_mla` 布局**，每层不是 576 B。本页的 FP8 档仍按每元素 1 B 计，与 K3 / GLM-5.3-Flash 同口径，未核对。
- **QSA 每层的原始 key 环**（约 13 KiB/请求）与两个短卷积 state（约 2.3 MiB/请求）未计。
- **每卡 12 GiB overhead** 在两个新模型上都没有可借的实测。Qwen3.8-Flash-Next 有 `hc_count = 4`，激活峰值可能更大。
