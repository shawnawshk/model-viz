# ADR-0009：权重加第三桶（按 TP 切的查表），新增 CSA2 family，KV dtype 承认例外

- 状态：**Accepted → 已在 `app.html` + `index.html` + 三个 data 文件实施**（2026-09-10）
- 触发：收录第三个模型 DeepSeek-V4.1-Flash（`deepseek-ai/DeepSeek-V4.1-Flash`）
- 影响：data 契约的 `weights` 块（**又一次不兼容改动**）、新增 `csa2` / `swa` family、新增「不可 TP 切的每请求固定字节」、`kvDtypes` 允许声明为架构固定
- 相关：[ADR-0007](0007-kv-dtype-is-an-input.md)（本 ADR 给它记一条例外）、[ADR-0008](0008-two-bucket-measured-weights-and-dsa.md)（两桶权重 / `perTokenFixedBytes`）、[ADR-0004](0004-provenance-first-class.md)（provenance）、[ADR-0003](0003-scope-memory-feasibility-only.md)（只算显存）

## 触发原因：第三个模型把「两桶」和「KV dtype 是引擎选项」同时打破

DeepSeek-V4.1-Flash 的一手件对账结果如下。对 48 个 shard 逐张量读 header，共读到 96,085 个张量。字节合计 `510,286,023,000`，与 `index.json` 的 `metadata.total_size` **精确相等**，等于 475.24 GiB：

| 桶 | 字节 | GiB | 逻辑参数量 | B/参数 | 切法 |
|---|---|---|---|---|---|
| routed experts（40 层 × 384 + 3 个 DSpark 层 × 128） | 295,997,276,160 | 275.67 | 557,171,343,360 | 0.53125 | EP |
| **engram 哈希查表（2 张）** | 203,073,076,240 | **189.13** | 196,928,504,320 | 1.03125 | **TP，且不按 PP 切** |
| 其余全部 | 11,215,670,600 | 10.45 | 9,105,468,058 | — | TP，DP 上复制 |

问题不在于「多了一项」。问题在于**第三项的切法与前两项都不同**：

- 它不按 EP 切。它不是专家，而是一张按 n-gram 哈希查行的嵌入表。
- 它按 TP 切。`model.py` 的 `ParallelEngramEmbedding` 按行切在 `world_size` 上，查完做 `all_reduce`。**因此它的切分组就是 TP 组。** 它不能跨 DP replica 切。原因：那个 all_reduce 会把 replica 耦合起来，DP 就不再起作用。
- 它**不按 PP 切**。两张表挂在 layer 1 和 layer 14，都在 20 层 encoder 里。PP=2 时，这两张表整体落在 stage 0。stage 1 一份都没有。

用 ADR-0008 的两桶口径，这 189 GiB 只能并进「非 expert 桶」。后果是：这部分权重会被当成可按 PP 切的东西，也会被当成受「非 expert 权重 dtype」开关影响的东西。这两条都错。错的方向是**低估**。

同时，这个模型打破了 ADR-0007 的前提：这个模型的 KV cache dtype **不是**引擎启动参数。详见 §3。

## 决议

### 1. `weights` 加第三桶 `tpShardedBytes`

```js
weights: {
  totalBytes,            // 全部张量字节(= index.json 的 metadata.total_size)
  expertBytes,           // routed expert 桶            → 按 EP 切、按 PP 切
  expertParams,
  tpShardedBytes,        // 第三桶:按 TP 切、不按 PP 切、dtype 开关不动它
  tpShardedParams,
  nonExpertParams,       // 其余参数量(不含第三桶)
  nonExpertBf16Params,   // 其中能被「非 expert 权重」开关改成 FP8 的部分
  nonExpertFixedBytes,   // 其余非 expert 字节(F32 + 原生 FP8)
}
```

三桶的切法必须在**一张表里同时写清**。原因：「归错桶」这个失效模式，在 ADR-0008 里已经发生过一次。这一次还多了 PP 这一维：

| 桶 | ÷TP | ÷PP | ÷EP | dtype 开关 |
|---|---|---|---|---|
| expert | — | ✓ | ✓ | expert 格式 |
| tpSharded | ✓ | **✗** | — | ✗（原生 FP8 固定） |
| nonExpert | ✓ | ✓ | — | 非 expert 格式 |

引擎的闭合断言相应改为。写错一位数字，立刻白屏，不会静默出错：

```
nonExpertBf16Params × 2 + nonExpertFixedBytes === totalBytes − expertBytes − tpShardedBytes
```

对 DeepSeek-V4.1-Flash：`1,976,359,936 × 2 + 7,262,950,728 = 11,215,670,600` = `510,286,023,000 − 295,997,276,160 − 203,073,076,240` ✓

`tpShardedBytes` 缺省为 `0`。K3 与 GLM 的数字**一个都不变**。

### 2. engram 整张表算进 HBM，标为最坏情况，反事实与结论同屏

这是本模型最大的软数字，量级 **40%**。这个量级远超此前任何一项。

**取「计入」而不是「省略」**。理由与 ADR-0008 §6 处理 indexer key cache 时相同。如果省略，并发会被高估到离谱：189 GiB 是 p5en 单卡显存的 1.34 倍。而且页面上不会有任何色块提示这一项存在。如果计入，至少会有一个带 `估算` 标记的可见色块。读者由此知道该去质疑什么。

按 ADR-0007 建立的惯例，反事实必须**与结论同屏**出现，共三处：banner、边界声明块、假设区。文案是：「若引擎把 engram pin 在 host memory 上按需查（它每 token 只查 24 行），这 189 GiB 从显存里消失，本页所有数字推翻重画」。

**不做成界面开关。** 原因与 KV dtype 不同。KV dtype 是引擎里**已经存在**的参数（`--kv-cache-dtype`）。把它提为输入，是把隐含假设显式化。engram 的放置方式，在任何生产引擎里**都还不存在**（vLLM / SGLang 目前都没有这个结构）。给一个不存在的引擎行为做配置项，就是把猜测伪装成选项。

### 3. ADR-0007 的例外：本模型的 KV dtype 是架构固定的，是一手依据

ADR-0007 的结论是「KV cache 的 dtype 是引擎启动参数，不是模型属性」。**这条在 DeepSeek-V4.1-Flash 上不成立。**

`inference/model.py` 里 FP4 量化的 block 大小与 scale 格式是**写死的字面量**，不是可配置项：

```python
fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)   # 主 KV:block 16 + E4M3 scale
fp4_act_quant(k, fp4_block_size, True)                             # indexer key:block 32 + E8M0 scale
```

而且这两处 block 大小**故意不同**。源码注释自己写明：「Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0」。换 dtype 不是改一个启动参数，而是改架构。

所以：

- `kvDtypes` 允许只有一档。新增 `kvDtypeFixed: true`。
- 引擎在 `kvDtypeFixed` 为真时，**隐藏下拉框、隐藏反事实**。引擎同时在假设区，把这一项从「估算」提为「一手件（参考实现源码）」。
- 边界声明块的动态列举，因此对本模型**少列一项**。ADR-0008 §7 把「声明假设的数量」改成动态列举，目的正是让清单能双向变化。这里第一次用到「变短」这个方向。

**不要把 ADR-0007 改掉。** 它对 K3 / GLM 依然成立。它记录的失效模式（隐含的 FP8 假设让头号数字虚高一倍）与本例无关。ADR 是决策时点的记录，按惯例不回改。本 ADR 就是 ADR-0007 的例外条目。

### 4. 新增 `csa2` / `swa` family：KV 不再是「层数 × 每层元素数」

前两个模型的 KV，都能写成「持有 KV 的层数 × 每层每 token 元素数」。这个模型不能这样写。原因有三个：

1. **40 层里只有 4 层持有全局 KV**（`kv_source_layers = [2, 8, 14, 20]`）。其余 36 层从这 4 份里读。这就是 CSA2（Compressed Sparse Attention 2）的 Reindex / Reuse 模式。
2. **这 4 层的 `compress_ratio` 不同**。层 2/8/14 是 2（两个 token 池化成一个 latent）。层 20 是 1。所以「每 token 的 latent 数」是 `0.5+0.5+0.5+1 = 2.5`，不是整数。
3. **FP4 加两种 scale 布局**。「元素数 × dtype 字节数」这个乘法本身不成立。

改法是复用 ADR-0008 已经建立的 `perTokenFixedBytes(g)` 钩子。这个钩子当初是为 GLM 的 indexer key cache 加的。改法把 4 个 source 层拆成按 `compressRatio` 分组的 layer group：

```js
layers: [
  { count: 3,  family: "csa2", compressRatio: 2, latentElems: 512, latentScaleBytes: 32,
                               indexerElems: 128, indexerScaleBytes: 4, ... },
  { count: 1,  family: "csa2", compressRatio: 1, /* 同上 */ },
  { count: 36, family: "swa",  /* 只有环形 buffer,不持有全局 KV */ },
]
```

```
csa2.perTokenFixedBytes(g) = (latentElems×0.5 + latentScaleBytes
                            + indexerElems×0.5 + indexerScaleBytes) / g.compressRatio
```

引擎侧 `M.kvFixedBytesPerToken += g.count × perTokenFixedBytes(g)` 这一行**不用改**，直接得：

```
3 × (288 + 68)/2  +  1 × (288 + 68)/1  =  534 + 356  =  890 B/token
```

**这个 890，与 model card 的头号数字（"890 bytes per token"）精确吻合。** 一手件反推的结果，与官方声明独立对上。所以本模型的 KV 是 provenance 里最高的一级，不是估算。

`swa` family（`perTokenElems: () => 0`、`perTokenFixedBytes: () => 0`）不是冗余。原因：36 个层里**没有**全局 KV，这件事本身要在页面上显示出来。否则，读者会以为 890 是 40 层摊出来的。

### 5. 新增「不可 TP 切的每请求固定字节」，与既有的 `stateBytesPerReq` 区分

每一层都有一个 128 槽的 SWA 环形 buffer（`window_size = 128`）。这个 buffer **与上下文长度无关**：

```
40 层 × 128 槽 × 512 维 × 2 B(BF16) = 5,242,880 B = 5.000 MiB / 请求
```

引擎里已经有一个每请求固定量 `stateBytesPerReq`，是给 GLM 的 KDA recurrent state 用的。但这个量**是 ÷TP 的**：KDA 的 state 按 head 切，TP 能分摊。SWA 环形 buffer 不同。它存的是所有 head 共享的 512 维 latent（`wkv` 只输出一个向量）。**TP 切不掉这个 buffer，每张卡要存一份完整副本。** 这个行为与 KV cache 相同，与 recurrent state 相反。

所以新增一个独立字段，两者**不许合并**：

| 字段 | 来源 | ÷TP | ÷PP | 随上下文 |
|---|---|---|---|---|
| `stateBytesPerReq` | GLM 的 KDA recurrent state | ✓ | ✓ | ✗ |
| `windowBytesPerReq` | DSv4.1 的 SWA 环形 buffer | **✗** | ✓ | ✗ |

峰值单卡占用因此是：

```
ceil(并发 ÷ DP) × (上下文 × kvBytesPerToken + windowBytesPerReq) ÷ PP
```

**为什么这项不能省略**：临界上下文是 `5,242,880 ÷ 890 = 5,891 tokens`。**上下文短于约 5.9K 时，这个固定 buffer 比真正的 KV 还大。** 1K 上下文下，它占单请求的 85%。省略它会让 1K 的并发虚高 6.75 倍。这与 ADR-0008 §5 记录的现象是同一类：「GLM 的 recurrent state 在短上下文下反超 KV」。区别是：那一次的量可以按 TP 切，这一次不能。

### 6. runtime 字节 ≠ checkpoint 字节，第一次出现

本项目的权重口径是「实测 checkpoint 字节」（ADR-0008 §1）。这个模型第一次让 runtime 字节与 checkpoint 字节不相等。原因：`convert.py` 在转换时，把 `wo_a` 从 FP8 **反量化成 BF16 落盘**。源码注释自己写明原因：「an fp8 grouped GEMM would halve the memory」。另外，3 个 compressor 的 `wkv` 用的是 FP32。

```
runtime − checkpoint = +1,341,431,552 + 15,728,640 ≈ +1.25 GiB
```

**决议：页面继续算 checkpoint 字节，差额写进 `notes`。** 理由：checkpoint 字节是可三方对账的硬数据（`total_size` / HF API / 逐张量求和）。runtime 字节取决于转换脚本的选择，换一个引擎就会变。1.25 GiB 在 475 GiB 里占 0.26%，不值得为它引入一个「运行时膨胀」字段。但值得在 `notes` 里写明，因为这是「实测」二字第一次出现裂缝。

### 7. expert 格式：FP4 原生 + FP8，不编造 BF16 档

`convert.py --expert-dtype` 只接受 `fp4 | fp8`。所以只给这两档：

| 档 | B/参数 | expert 桶 | 依据 |
|---|---|---|---|
| FP4（原生） | 0.53125 | 275.67 GiB | 实测；= 0.5 + 每 32 元素一个 E8M0 scale ✓ 精确 |
| FP8 | 1.0009765625 | 519.36 GiB | `cast_e2m1fn_to_e4m3fn()`；= 1 + 每 32×32 块一个 scale，与本 checkpoint 里原生 FP8 的 shared expert 实测值逐位相同 ✓ |

GLM 那份给了一个非原生的 BF16 档，用来演示「一台立刻装不下」。这里**不给**。原因：转换脚本没这个选项。编一个出来，就是把想象写成数据。FP8 那一档已经足够演示同一件事。

**FP4 的 shape 陷阱写进 `notes`**。checkpoint 里 expert 张量的 dtype 标记是 `I8`。K 维是真实维度的**一半**（`w1 (2304, 2560)` 实际是 `(2304, 5120)`），因为两个 FP4 值打包进一个字节。逻辑参数量必须 ×2 才对得上：`2 × 278,585,671,680 = 557,171,343,360 = 15,744 experts × 3 × 2304 × 5120`，一个不差。**照字面读 shape 会把 expert 参数量低估整整一倍。** 这与 ADR-0006 记录的「141 GiB 当成 141e9」是同一类单位/口径陷阱：厂商或格式给出的字面数字，不能直接当语义值用。

ADR-0008 §1 的第二条断言（`expertParams ÷ (experts × 3 × latent × inter)` 必须是整数）在这里得到 `41`。**41 不是层数。** 40 个 backbone 层，每层 384 个专家。另外 3 个 DSpark 层，每层 128 个专家。`3 × 128 = 384`，恰好等价于一层。断言仍然有效，它只要求结果是整数。但引擎派生的 `M.expertWeightLayers`，在本模型上是「等效层数」，不是物理层数。这一点要在假设区里说明。否则，那个 41 会被读成层数。

### 8. PP 在本模型上不再是均匀的，页面必须说出来

engram 挂在 layer 1 和 layer 14。PP=2 时，这两张表都在 stage 0。引擎目前假设各 stage 均匀，即所有权重一律 ÷PP。这个假设在本模型上不成立。

**决议：`tpShardedBytes` 不除 PP。`PP > 1` 时出一条 `warn`。** 这样，单卡数字取的是**最坏 stage**。对「装不装得下」这个问题，这个方向保守且正确。但这样会让 stage 1 的余量被低估。所以必须由告警文案说明这一点，不能让读者以为那是均值。

按 ADR-0005，这里**不做**「建议怎么摆 stage」的推荐。推荐怎么摆 stage，是求解器的工作。本 ADR 只报告不均匀这个事实。

### 9. 预设：这个模型要讲的是「TP 有下限」，且下限随卡的显存移动

前两个模型的故事，是 TP 的**代价**。TP 会复制 KV latent，所以 TP 越小越好。GLM 上，TP1×DP8 的并发比 TP8/DP1 高 6 倍。这个模型的故事**方向相反**：KV 只有 890 B/token，复制它的代价几乎为零。限制并发的是 engram：engram 只能按 TP 切。所以 **TP 有下限，这个下限随单卡显存移动**。

手算（util 0.90、每卡 12 GiB overhead、128K、EP=卡数）：

| 机型 | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| p5en 1 台（141 GiB） | 装不下 | 装不下 | **532** | 487 |
| p6-b200 1 台（179 GiB） | 装不下 | 500 | **1,134** | 1,020 |
| p6-b300 1 台（269 GiB） | 差 5.56 GiB 装不下 | **3,340** | 2,554 | 1,498 |

三个机型的最优 TP 分别是 4 / 4 / 2。TP=1 在三个机型上全部装不下，b300 上只差 5.56 GiB。GLM 页上「点一下 TP1×DP8，看并发跳 6 倍」的演示，在这一页会变成「点一下，直接红字装不下」。**这个对照本身就是页面最值得讲的东西**。两页放在同一个引擎上，才能看出这一点。

预设名字按 ADR-0008 §9 的规矩：只描述**结构与机制**，不写绝对路数，不写「推荐 / 最优」。原因：上表已经证明排序随机型变，标「推荐」就是越界。括号里标该切分下 engram 被切成几份。这是差异的机制，恒真。路数由 `markPreset()` 按当前口径现算。

`defaultParallel` 取 `TP8/DP1/EP8`。官方 `inference/README.md` 明写 `MP=8`，这是唯一有一手依据的起始切分。读者先看到官方配置，再点 TP4，看到官方配置其实不是这一页口径下最高的那档。

### 10. 每卡 12 GiB overhead 的猜测在本模型上更不牢，要单独说

`hc_mult = 4`：残差流以 4 份并行副本携带（Single-Pass mHC），激活量因此直接 ×4。再加上 indexer scratch：`index_topk = 512`，`candidate_topk_blocks = 2048 × candidate_block_size = 8`。

这不改变「12 GiB 是纯猜测」这个既有标记。但**敏感度不同**，要在本模型的 `notes` 里写明：同一个常数，在三个模型上的可信度不一样。不要让读者以为这个常数已经被验证过。

（把 `overhead` 提为界面输入，这个待办项本 ADR 仍不解决。见 CLAUDE.local.md。）

## 后果与验证

`verify-app.js` 从 8 个用例扩到 **13 个**。新增的 5 条，都是本模型的用例：

| 用例 | 钉住的是什么 | 路数 |
|---|---|---|
| 官方 MP=8 `1×p5en TP8/DP1/EP8` | 890 B/token（= model card 头号数字）+ 整组预设 | 488 |
| `1×p5en TP4/DP2/EP8` | TP4 反超官方 TP8，且这个反超是本页口径下的结论 | 538 |
| **1K 上下文** `TP8/DP1` | `windowBytesPerReq`（见下） | 9,682 |
| **PP2** `TP4/DP1/PP2/EP4` | PP 分支渲染 + `expectLookupGiB = 47.28`（engram 不 ÷PP） | 561 |
| **装不下** `TP1×DP8` | `csa2` 路径下「权重本身就超预算」+ `expectLookupGiB = 189.13` | 0 |

- **1K 那条用例不可删。** 如果 `windowBytesPerReq` 被漏乘，1K 的期望值会差 **6.75 倍**。128K 只差 4.5%，1M 只差 0.6%。三个上下文里，只有 1K 能抓住这个漏乘。预设整组 `[9682, 10658, 0, 0, 66640, 33340]`，与 128K 那组 `[488, 538, 0, 0, 3364, 1680]` 对照，也顺带再证一次：预设路数是现算的。
- **那两个 `0` 不是写错。** 128K 与 1K 两组预设的第 3、4 项（`TP2/DP4` 与 `TP1×DP8`），在 p5en 上都是 0。原因：engram 只能按 TP 切，TP<4 时单卡就放不下 engram。这正是本模型最该被钉住的行为。不要当成 bug 删掉。
- 新增 `kvFixed` 断言：在 `kvDtypeFixed` 为真的模型上，探针那两轮 `bf16 / fp8` 必须给出**完全相同**的 B/token 与路数。这条断言，把 §3 的决议变成机器可查的东西。如果哪天有人给这种模型加了第二档 dtype，这条断言会当场报错。
- 新增 `expectLookupGiB`：直接钉住第三桶的单卡字节数。原因：第三桶的除法规则（÷TP，不 ÷PP）与另两桶都不同，混进总数里看不出来。
- 期望值一律先手算，再与引擎对账。手算与引擎的差在 1%–2%，例如 TP4/DP2 手算 532、引擎 538。差的来源是：手算用了 runtime 字节，页面用 checkpoint 字节（§6）。这不是 bug。
- **K3 与 GLM 的 8 条断言逐字不变通过。** `tpShardedBytes` / `tpShardedParams` 缺省 0，`kvDtypeFixed` 缺省 false，`windowBytesPerReq` 恒为 0。原因：这两个模型的 family 都不产生滑窗环。

顺带补齐三处既有空洞。这三处都是被本模型第一次触发暴露的，不是主动重构：

- `.pv-m::before` 字形。`PV_LABEL` 里一直有 `m: "实测"`。但在 DSv4.1 的 KV 之前，没有任何一项用得上它。所以这条字形从来没写过。
- `index.html` 的模型卡片改为 `totalBytes − expertBytes − tpShardedBytes`。有第三桶时，模型卡片多显示一格。如果不改，索引页会把 189 GiB 并进「非 expert」，与详情页自相矛盾。
- 集群权重总存量那块 tile，原先只加 `attn + expert`，会把第三桶整个漏掉：显示 0.28 TiB，真实值是 0.46 TiB。

**没有动的**：`:root[data-theme="dark"]` 那个块缺 `--series-kda`。暗色主题走 attribute 时，KDA 用的是浅色值。这是既有的不一致，与本次改动无关。只记录，不修复。

**引擎侧口径变化**：对前两个模型的输出无影响，但改了内部结构。`M.stateShardableByTp` 这个全局布尔量被删掉了。这个布尔量原先的规则是：「所有 family 里只要有一个可 TP 切，就整体可切」。这个规则在同时存在两类 family 的模型上会算错。现在改为：在 load 时，就按 family 分进 `stateBytesPerReq` / `windowBytesPerReq` 两个累加器。`compute()` 里，各累加器按自己的规则除。

## 仍未解决

- **engram 的放置方式，是本项目至今最大的软数字（40%）。** 消除办法与前两项相同：在目标机型上起一次服务，读引擎自报的 KV block 数反解（ADR-0004 / 0007 / 0008 都指向这个办法）。但这一项更麻烦：现在还没有生产引擎支持这个结构。所以短期内**无法**用这个办法消除。
- **SWA 环形 buffer 的存储 dtype 是估算。** 参考实现分配的是默认 dtype（BF16）。但注释说数值上「stays fp8」。真实引擎大概会直接存 FP8。那样的话，5.000 MiB 会变成 2.500 MiB，1K 上下文下的并发会差 1.8 倍。
- **3 个 DSpark 层按「加载」算。** 不开投机解码时，这些层不加载：expert 桶少 6.72 GiB，SWA 环少 3 层。这个口径与 ADR-0008 §4 处理 GLM 的 MTP 层一致：全算，差额写 `notes`。
- **vision 塔按「常驻」算**（0.90 GiB，含 aligner）。图像 token 的额外显存不算（`vision_max_n_token = 1024`/图）。按 ADR-0003，这部分属于吞吐侧的 batch 构成，不是本工具的问题。
- **PP 的 stage 不均匀，只报告、不建模。** 见 §8。真要建模，需要引擎知道每层的权重分布。这是比本 ADR 大得多的改动。
- **GQA family 仍未落地**，这一点与 ADR-0008 留下的那条相同。注意：本 ADR 新增的 `csa2` **不是** GQA 的替代。`csa2` 的每 token 字节数是静态的，source 层与 ratio 都写在 config 里。GQA 的复制因子 `max(1, TP/n_kv_heads)` 是**并行配置的函数**，仍然需要给 `perTokenElems(g)` 传当前切分。三件事不要混：`perTokenFixedBytes`、`windowBytesPerReq`、GQA 的动态复制因子。
