# ADR-0010：在同一引擎上加 decode 的 roofline 下界；部分取代 ADR-0003 的「不算吞吐」

- 状态：**Accepted → 已在 `app.html` + `index.html` + `data/` 实施**（2026-09-15）
- 触发：用户要一个「给定模型 / 机型 / 并行策略 → 理论上界」的模型，用来给部署选型和 benchmark 设计定方向
- 影响：**取代** [ADR-0003](0003-scope-memory-feasibility-only.md) 后果里的「不回答哪个 instance 吞吐更高」与 [ADR-0005](0005-validator-not-solver-and-instance-list.md) §4 的「不算 FLOPS、不算吞吐、不算延迟」；**不动**成本 / 可得性那一条，**不动**校验器语义。data 契约：`instances` 加 `hbmGBs` / `denseTflops`，稀疏 family 的层组加 `indexTopk`
- 相关：[ADR-0002](0002-page-per-model.md)（单引擎）、[ADR-0004](0004-provenance-first-class.md)（provenance）、[ADR-0007](0007-kv-dtype-is-an-input.md)（KV dtype 是输入）、[ADR-0009](0009-third-weight-bucket-and-csa2.md)（family 钩子）

## 为什么重开 ADR-0003

ADR-0003 把吞吐排除的理由有两条：成本没依据，所以不能回答「哪个更值」；instance 轴先做。**两条都不是反对吞吐本身的论证。** ADR-0005 §4 那句「不算 FLOPS、不算吞吐、不算延迟」的标题是「防止范围回弹」——那是纪律，不是论据。

这次让纪律让路，理由是三件已经存在的东西：

1. **显存侧已经算出了 roofline 需要的全部静态量**：每卡三桶权重（含复制因子）、每 token KV 字节、每请求固定 state、EP 复制组、跨域通信次数。另起一个工具意味着复制这些公式——正是 ADR-0002 反对的事。
2. **并发和上下文两个滑块就是 roofline 的两个横轴。** decode 的每步耗时是 `f(这一步的 token 数, 每请求上下文长度)`，页面上已经有这两个输入。
3. **provenance 五级和「任何数字必须连口径一起报」**正是「理论上界 ≠ 预测」需要的标记体系。`measured` 那一级至今是空的，roofline 的校准系数将是它的第一批住户。

**不取代的：**

- 成本 / 可得性仍然不算。roofline 给的是每卡 tok/s 上界，不是 $/token。
- 校验器不是求解器（ADR-0005 §1）。roofline 只对**当前配置**给数，不排序、不推荐、不搜索。
- 每卡 12 GiB overhead 等既有软数字的处理一律不动。

## 决议

### 1. 算什么：decode 一个 micro-step 在最忙那张卡上的耗时下界

```
step ≥ max( HBM 读取字节 ÷ HBM 带宽,  矩阵乘 FLOPs ÷ dense 算力,  通信字节 ÷ 链路带宽 )
```

三条线都取厂商 spec 峰值；取 `max` 假设三者完美重叠。**没有效率系数，没有每步固定开销，没有 attention FLOPs**——所以它是下界，不是预测（§6）。

由它派生的输出：

| 输出 | 式子 | 含义 |
|---|---|---|
| ITL 下界 | `PP × step` | 一个请求相邻两个 token 之间至少隔多久 |
| 集群吞吐上界 | `⌈并发 ÷ PP⌉ ÷ step` | 每个 step 有一个 micro-batch 完成 |
| 单用户 tok/s 上界 | `1 ÷ ITL` | |
| regime | 三项里哪个是 max | HBM 带宽受限 / 算力受限 / 通信受限 |
| 拐点 B\* | `tComp(B) = tMem(B)` 的解 | 低于它加并发几乎不加 ITL，高于它 ITL 随并发线性涨 |
| 临界上下文 L\* | `KV/state 读取 = 权重读取` 的解 | 短于它 ITL 基本不随上下文变 |

**v1 不算 prefill / TTFT。** prefill 的 attention 项是 `O(S²)`，且 chunked prefill 让它与 decode 混在一个 step 里，需要另一组假设，不塞进这一版。

### 2. 三个量怎么数

**并发滑块读作「这一步的 token 数」**（每请求 1 token，不含 MTP / 投机解码）；**上下文滑块读作「每请求当前上下文长度」**（§4）。记

```
stageTok = ⌈并发 ÷ PP⌉            一个 micro-batch 落在一个 stage 的 token 数
rankTok  = ⌈stageTok ÷ DP⌉        最忙的那个 DP rank 上的 token 数
groupTok = stageTok × EP ÷ stage   到达本 expert 复制组的 token 数(EP = stage 时 = stageTok)
```

**(1) HBM 读取**（每卡，每 step）

| 项 | 式子 | 级别 |
|---|---|---|
| 非 expert 权重 | 显存侧的 `attn` 桶原样（`÷TP ÷PP`），整读 | 推导 |
| routed expert 权重 | `expert 桶 × (主干 MoE 层 ÷ 带 expert 权重的层) × 命中率`，命中率 `= 1 − (1 − k/E)^groupTok` | **估算**（均匀路由） |
| KV / state | `rankTok × [ Σ family.decodeReadPerReq(ctx) ÷ PP + 2 × state ÷ TP ÷ PP + window ÷ PP ]` | 推导 / 估算，随 KV dtype |

三点说明：主干 MoE 层数用 `moe.layers`，不用 `expertWeightLayers`——MTP / DSpark 那套 expert 权重占显存但不参与 decode（与 all-to-all 计数口径一致）；recurrent state 每步读一次写一次所以 ×2，滑窗环只读；`÷TP / ÷PP` 的规则与显存侧完全相同（MLA latent 每个 TP rank 各读一份完整副本，state 按 head 切）。engram 查表每 token 只查几十行，KB 量级，不计。

**(2) 矩阵乘 FLOPs**（每卡，每 step，`2 × 参数 × token`）

```
expert   = 2 × (3 × latent × inter) × (主干 MoE 层 ÷ PP) × stageTok × k ÷ stage
非 expert = 2 × nonExpertBf16Params ÷ TP ÷ PP × rankTok   按「非 expert 权重」格式选的 dtype 行
         + 2 × (nonExpertParams − nonExpertBf16Params) ÷ TP ÷ PP × rankTok   按 FP8 行(原生 FP8 模块 + F32 norms)
```

expert 按 stage 均摊——`stageTok × k` 个 expert-token 对分给 stage 张卡，EP 是否等于 stage 不改变这个均值。`nonExpertParams` 含 embedding 与 vision 塔，它们不参与矩阵乘，高估不到 3%，不单独扣。

**dtype 行的选择规则**与 banner 里「无原生 FP4 的两条出路」的 (a) 路一致：机型有原生就用原生（MXFP4 / NVFP4 → FP4 行）；无原生 → dequant 到 FP8 行，没有 FP8 → BF16 行。假设激活与权重同精度。

**(3) 通信**（每卡，每 step，只算字节 ÷ 带宽，**没有延迟项**）

```
TP all-reduce  = (层 ÷ PP) × 2 次 × 4(TP−1)/TP × rankTok × hidden × 2 B      ring,收 + 发合计
EP all-to-all  = (MoE 层 ÷ PP) × 4 × stageTok × k × moe.latent × 2 B × (EP−1)/EP ÷ stage   dispatch + combine,收 + 发
```

链路：`TP ≤ NVLink 域` 用 NVSwitch 双向合计带宽（`p2p.gbs`）；`TP > 域` 整段按 EFA 线速的每卡份额（`interDomainGbps ÷ 8 ÷ 每实例卡数`）。all-to-all 跨域时把非本卡流量按 `(EP − 域) ÷ (EP − 1)` 分给 EFA、其余给 NVSwitch，两条链路并行，取 `max`。PCIe 机型没有卡间带宽数据，通信一行按 0 计并在页面明说。

dispatch 与 combine 都按 BF16 计（估算）。引擎若用 FP8 dispatch，该项少 25%。

### 3. 稀疏 attention 的 decode 读取走 family 钩子，不能按「全量 KV」算

这是本 ADR 唯一一处**必须**做细的地方。GLM-5.3-Flash 的 DSA 在 128K 上下文下：

| 读法 | 每请求每步 |
|---|---|
| 全量读 latent（错） | `131,072 × 11 层 × 512 × 2 B ≈ 1.44 GB` |
| indexer 全扫 + top-k latent（对） | `131,072 × 352 B + 2,048 × 11 × 1,024 B ≈ 68 MB` |

差 20 倍，而且方向是**高估耗时**——这会让「下界」不再是下界。所以 `FAMILY` 表加一个 `decodeReadPerReq(g, ctx, kvBytes)` 钩子：

| family | decode 每请求读多少 |
|---|---|
| `mla` | `ctx × (kv_lora_rank + rope) × kvBytes`（full attention，全读） |
| `dsa` | `ctx × indexer 字节 + min(ctx, indexTopk) × latent 字节` |
| `csa2` | 池化后 `n = ctx ÷ compress_ratio`：`n × indexer 字节 + min(n, indexTopk) × latent 字节` |
| `swa` / `linear` | 0（它们的读取在 `perReqBytes` 那条：滑窗环整读，recurrent state 读 + 写） |

为此层组要多一个字段 `indexTopk`（`config.json` 的 `index_topk`，GLM 2048、DSv4.1 512——两个值原本就写在各自 `notes` 里）。**引擎加载时硬断言**稀疏 family 必须有它，缺了白屏，不静默按全量算。

CSA2 的 36 个 reuse 层是否重读 source 层选出的 top-k latent，参考实现没看透，**未建模**；量级 `36 × 512 × 288 B ≈ 5.3 MB/请求`，与 indexer 全扫比是小项，但要记着。

### 4. 上下文滑块的双重口径

显存侧把上下文读作「容量规划的最大长度」；roofline 读作「每请求此刻的长度」。**同一个滑块，两种读法**，页面文字两处都说了。

不加第二个滑块的理由：多一个输入会让预设 / 口径联动再复杂一层，而「所有请求都处在这个长度」恰好是 ITL 最差的那个点——对一个只给下界的工具来说，看极限点比看平均更对路。

### 5. 新增数据字段与 provenance

`data/instances.js` 每个机型加：

```js
hbmGBs: <十进制 GB/s>,                       // HBM / GDDR 带宽
denseTflops: { bf16: <T>, fp8: <T>, fp4: <T> },   // dense(非 sparsity)张量核算力;没有那一档就不写
```

三条规则：

1. **只取 dense 行。** NVIDIA datasheet 通常印的是带 sparsity 的数字，有的加星号，有的写成 `a | b`。抄错一倍与 ADR-0006 记的 GiB/GB 是同一类陷阱。每个数字旁边注明出处 URL 与「datasheet 原文是否为 sparse」，对不上的写 NOT FOUND，不从记忆里补。
2. **十进制。** 带宽 / 算力是厂商单位（GB/s、TFLOPS），内部按 `× 10⁹ / × 10¹²`；字节仍按 GiB 显示（ADR-0006），时间 = 字节 ÷ (GB/s × 10⁹)。两套单位在页面的口径条里写明。
3. **级别是 `spec`。** provenance 五级里 `spec` 原本只用在 `describe-instance-types` 的显存上；这次它第一次出现在 UI 标记里，字形 `▲ spec`。

`docs/instance-specs.md` 那 57 个实例的总目录**不加**这两列——它以 API 的 MiB 为真值，而带宽 / 算力没有 API 来源；只有被列为候选、进了 `instances.js` 的机型才有这两个字段。

**2026-09-15 抓取时发现的三件事**（都写进了 `instances.js` 的注释）：

- **HGX B300 的 FP4 dense 不是 sparse 的一半。** datasheet 印 `18 PFLOPS | 14 PFLOPS`，脚注 1 写 `Sparse | Dense`；同一张表的 FP8 / BF16 只印 sparse，脚注 2 写 dense 取 ½。所以 B300 填 `fp4: 14000, fp8: 4500, bf16: 2250`。B200 的 FP4 只印 sparse 18，dense 从板级行 `Total NVFP4 144 | 72 PFLOPS` ÷ 8 卡 = 9 反推，与脚注一致。
- **两张 RTX PRO 的 datasheet 完全没有 sparse / dense 字样。** 数字按原文照录，加 `tflopsSparsityUnstated: true`，页面在算力行旁标「未标注」并说明若实为 sparse 则松 2 倍。不从记忆里判定。
- **datasheet 印的显存与 API 对不上。** HGX B300 印每卡 `270 GB`，HGX B200 印 `180 GB`；`describe-instance-types` 给 `275,040 MiB`（≈ 288 GB）与 `183,359 MiB`（≈ 192 GB）。显存真值仍按 ADR-0006 取 API，这里只记录差异，不解释。

### 6. 没有效率系数、没有固定开销项：v1 是纯上界

`measured` 一级至今为空。填一个「一般能到 70%」的系数是把猜测伪装成校准——与 ADR-0009 §2 不给 engram 做开关是同一条理由。所以 v1 明确输出**未校准的 spec 上界**，并在页面上把「低并发下实测比下界慢一个量级以上是正常的」写成固定文案，不是脚注。

**获得校准的路径**（对应 ADR-0004 的「获得 `measured` 的路径」）：在目标机型上跑一次 genai-perf，在拐点 B\* 两侧各取一个并发点，记录 ITL 与总吞吐，`实测 ÷ 下界` 就是这台机器这个引擎的效率比。同机器换并行策略这个比例大致可复用；换机器要重测。拿到之后再决定是否加成一个输入——那是 ADR-0011 的事。

### 7. 校验器语义不变，预设按钮不加 tok/s

roofline 只对当前配置算，不做「哪个切分吞吐更高」的排序。**预设按钮 v1 不加吞吐上界**：点预设会把并发重置到 1 路，1 路的吞吐上界是「单用户 ITL 下界」的倒数，跨预设比它没有意义；要加就得先决定预设用哪个并发口径显示，那是另一个决定。

### 8. 页面边界声明改写

两个页面顶部的「这张图只算显存」改成「算显存，外加 decode 的 roofline 下界」，第一条改为「吞吐与延迟只给理论下界，不是预测」，并列出不含的项（固定开销、attention FLOPs、通信延迟、prefill）。成本那条、12 GiB 那条、动态列举那条不动。

## 后果与验证

- `app.html`：`FAMILY` 加 `decodeReadPerReq`；加载时断言 `indexTopk` 与 `hbmGBs / denseTflops.bf16`；新增 `roofTerms / roofRidge / roofKvCross / renderRoof`；新增面板（`roofhead / roofsum / roofwarn / rooftbl / roofnotes`）。显存侧 `compute()` **一行没动**。
- `verify-app.js`：五个新容器进 `REQUIRED` 与 `CLEAN`（`NaN / undefined` 检查）；每个用例都断言 step 为正有限数、regime 合法；6 个用例带 `roof` 期望（step ms 取 3 位有效数字、regime、拐点、临界上下文 ±1%）。**期望值由一个独立于引擎的 Python 脚本按 §2 的式子手算**（输入抄自 `data/` 与 datasheet），再与引擎对账——6 条全部一致：

| 用例 | 并发 × 上下文 | step 下界 | regime | 拐点 | 临界上下文 |
|---|---|---|---|---|---|
| K3 `4×p5en TP8/DP4/EP32` | 64 × 128K | 21.9 ms | HBM | 范围内无 | ≈ 98K |
| GLM `p5en TP1/DP8/EP8` | 1 × 128K | 3.75 ms | HBM | 范围内无 | 范围内无 |
| GLM `p5en TP1/DP4/PP2/EP4` | 16 × 128K | 3.40 ms | HBM | 范围内无 | 范围内无 |
| DSv4.1 `p5en TP8/DP1/EP8` | 32 × 128K | 3.46 ms | HBM | 范围内无 | ≈ 2.85M |
| DSv4.1 `p5en TP8/DP1/EP8` | 256 × 1K | 8.00 ms | HBM | 范围内无 | ≈ 813K |
| DSv4.1 `p5en TP2/DP4/EP8` | 4096 × 1K | 10.1 ms | **算力** | **≈ 4,022 路** | ≈ 205K |

- 最后一条是专门找出来的：**三个模型在 p5en / b300 上的所有常规切分，1 至 4,096 路之内都到不了算力受限**，扫描下来只有这一个配置（显存上装不下的那个）的拐点落在滑块范围内，所以用它钉住「算力受限」regime 与拐点二分那条路径。
- **K3 / GLM / DSv4.1 显存侧的 13 条断言逐字不变通过**——`indexTopk` 只进 `decodeReadPerReq`，不进 `perTokenElems / perTokenFixedBytes`。
- `index.html`：机型表加「HBM GB/s」与「dense TFLOPS」两列；边界声明同步。

### 第一次算出来就值得记的东西（是模型的推论，不是实测）

**这三个模型在 H200 / B300 上按本 ADR 的口径几乎永远是 HBM 带宽受限**，加并发到 4,096 路也到不了算力受限。原因不在权重，在「每 token 都要读一遍、随并发线性涨」的那一项：

- K3 / GLM 的 **KDA recurrent state**：每层每 head 一个 128×128 的 FP32 矩阵，每步读一次写一次。GLM 34 层 × 64 head = 每 token 285 MB 的读写；K3 69 层 × 96 head = 每 token 868 MB（÷TP）。这一项的每 token 斜率（GLM 约 60 µs/token/rank）比算力项的斜率（约 18 µs/token）大 3 倍，所以 `tComp − tMem` 永远为负。
- K3 的 24 个 **full-attention MLA 层**在 128K 下每请求每步读 3.6 GB latent，比 state 还大。
- DSv4.1 没有 state，但 **每层一个 5 MB 的滑窗环**每步整读，斜率与算力项接近，拐点被推到 4,000 路以外。

这个推论建立在两个本 ADR 自己引入的假设上：**state 是 FP32**（显存侧 ADR-0008 的既有假设）与**每步读 + 写各一次**（本 ADR §2）。若引擎把 state 存成 BF16 且做原地更新，这一项减半；若 KDA kernel 在 SRAM 里做多步 chunk 更新，还会更少。所以「永远带宽受限」是**按最保守的 state I/O 口径**得出的，标 `估算`。它的实际意义是：**在这三个模型上，光看 FLOPS 选机器会选错方向；该比的是 HBM 带宽，以及引擎怎么处理 recurrent state。**

## 仍未解决

- **attention 本身的 FLOPs 未计。** full-attention MLA 层在 128K 以上与矩阵乘同量级（K3：`2 × 96 head × ctx × 576 × 24 层` 对 `2 × 104B`），算力那一行在长上下文下偏松，regime 判断可能把算力受限误判成带宽受限。需要给 MLA 层组补 `heads` 字段并核对 config.json，v2。
- **prefill / TTFT 未算。**
- **通信没有延迟项，PP 没有 send/recv 与气泡。** 低并发下延迟项主导，这是「实测比下界慢 10 倍」的主要来源之一。
- **效率系数 / 校准数据**：见 §6 的路径。拿到第一组实测前，本页所有 roofline 数字都是 `spec × 推导`，没有一个是 `measured`。
- **投机解码（MTP / EAGLE）**：改变每步 token 数，接受率是实测量，不能算。
- **all-to-all 的 dtype**、**DeepEP 的 node-limited routing**（会降低跨域份额）：都按最简假设。
- **KDA state 的读写口径**（FP32、每步读 + 写各一次）决定了上面「永远带宽受限」的推论。这是本页第二大的软数字（第一仍是 engram），消除办法与其他项相同：在目标机型上测一次，对比 ITL 随并发的斜率。
- **预设按钮上的 tok/s**：见 §7。
- **GQA family**：继承自 ADR-0008 / 0009。
