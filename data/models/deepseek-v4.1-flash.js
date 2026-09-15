// DeepSeek-V4.1-Flash —— deepseek-ai/DeepSeek-V4.1-Flash
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数              derived  —— config.json(text_config)+ inference/config.json
//   权重字节数/参数量      derived  —— 逐张量读 48 个 shard 的 safetensors header 求和,
//                                    与 index.json 的 total_size 精确相等
//   每 token KV 字节数     derived  —— 从 inference/model.py 的 buffer 形状与量化调用反推,
//                                    结果与 model card 的头号数字「890 bytes per token」精确吻合。
//                                    **这是本项目第一个 dtype 也有一手依据的模型**,见 ADR-0009 §3
//   engram 是否常驻 HBM    guessed  —— 本页按「整张常驻」计。量级 40%,是本页最大的软数字
//   SWA 环形 buffer dtype  estimated —— 参考实现分配 BF16,注释说数值上是 FP8
//
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["deepseek-v4.1-flash"] = {
  name: "DeepSeek-V4.1-Flash",
  hf: "deepseek-ai/DeepSeek-V4.1-Flash",
  blurb: "552B backbone(+196B engram)· 8B prefill / 16B decode 激活 · 40 层 CED(20 encoder + 20 decoder)"
       + " · CSA2 稀疏注意力,每 token KV 仅 890 B · 384 routed experts / top-6 · expert 原生 FP4 · 多模态",
  hidden: 5120,                              // dim,只用于 PP 的 send/recv 激活量

  // ---- 层结构 ----
  // 40 层,但只有 4 层持有全局 KV(kv_source_layers = [2, 8, 14, 20]),其余 36 层从这 4 份里读 ——
  // 这就是 CSA2 的 Reindex / Reuse 模式。这 4 层的 compress_ratio 不同:
  //   层 2/8/14 在 encoder 里,ratio 2(2 个 token 池化成 1 个 latent)
  //   层 20 是 decoder 的第一层,ratio 1(decoder 的全局 KV 从 encoder 末态投影而来)
  // 每 token 的 latent 数因此是 0.5+0.5+0.5+1 = 2.5,不是整数 —— 旧的「层数 × 每层元素数」算不出来。
  //
  // latent 512 维 FP4,每 16 通道一个 E4M3 scale → 512×0.5 + 32 = 288 B
  // indexer key 128 维 FP4,每 32 元素一个 E8M0 scale → 128×0.5 + 4 = 68 B
  // 合计每 latent 356 B × 2.5 = 890 B/token,与 model card 的头号数字精确吻合。
  layers: [
    { count: 3, family: "csa2", compressRatio: 2,
      latentElems: 512, latentBytesPerElem: 0.5, latentScaleBytes: 32,
      indexerElems: 128, indexerBytesPerElem: 0.5, indexerScaleBytes: 4,
      windowSlots: 128, windowElems: 512, windowBytesPerElem: 2,
      note: "encoder 里的 3 个 KV source。ratio 2 = 两个 token 由一个学习到的 softmax gate 池化成一个 latent。" },
    { count: 1, family: "csa2", compressRatio: 1,
      latentElems: 512, latentBytesPerElem: 0.5, latentScaleBytes: 32,
      indexerElems: 128, indexerBytesPerElem: 0.5, indexerScaleBytes: 4,
      windowSlots: 128, windowElems: 512, windowBytesPerElem: 2,
      note: "layer 20 = decoder 的全局 KV source。CED 架构:decoder 的 KV 从 encoder 末态投影,不是每层自己算。" },
    { count: 36, family: "swa",
      windowSlots: 128, windowElems: 512, windowBytesPerElem: 2,
      note: "不持有全局 KV,只有自己那个 128 槽的滑窗环。SWA Bounded Replay:滑窗 KV 不落盘,靠重放最近 n_win 个 token 重建。" },
  ],
  totalLayers: 40,

  // ---- KV dtype:本模型是 ADR-0007 的例外 ----
  // FP4 的 block 大小与 scale 格式写死在 inference/model.py 里,而且主 KV 与 indexer key
  // 两处故意不同(16/E4M3 与 32/E8M0)。换 dtype 不是改启动参数,是改架构。
  // kvDtypeFixed 让引擎隐藏下拉框、隐藏反事实,并把这一项从「假设」提为「一手依据」。
  kvDtypeFixed: true,
  kvDtypes: [
    // label 里不写「架构固定」—— 引擎的 banner 与假设区已经在句子里说了,写进 label 会套两层括号
    { id: "fp4", label: "FP4 E2M1", bytes: 1 },   // bytes 不参与计算:CSA2 的字节数全走 perTokenFixedBytes
  ],

  moe: {
    layers: 40,            // 40 层全是稀疏 MLP,没有 dense 前缀层
    experts: 384, topk: 6, sharedExperts: 1,
    latent: 5120, inter: 2304,   // expert 直接吃 dim,没有额外的 MoE latent 投影
  },

  // ---- 权重:三桶实测字节(口径见 ADR-0009 §1)----
  weights: {
    totalBytes:  510_286_023_000,      // 96,085 个张量逐个求和,与 index.json 的 metadata.total_size 精确相等
    // 桶 1:routed experts。40 层 × 384 + 3 个 DSpark 层 × 128 = 15,744 个专家。
    expertBytes: 295_997_276_160,      // = 275.67 GiB(FP4 259.45 + E8M0 scale 16.22)
    expertParams:    557_171_343_360,  // 15,744 × 3 × 2304 × 5120。**注意这是 I8 numel 的 2 倍** —— 见 notes 第 2 条
    // 桶 2:engram 哈希查表。按 TP 切、不按 PP 切、dtype 开关不动它。
    tpShardedBytes:  203_073_076_240,  // = 189.13 GiB(2 张表,FP8 183.40 + E8M0 scale 5.72)
    tpShardedParams: 196_928_504_320,  // 与 model card 的「Engram conditional memory (196B parameters)」一致
    // 桶 3:其余全部。
    nonExpertParams:     9_105_468_114,   // BF16 1,976,359,936 + F32 42,307,282 + 原生 FP8 7,086,800,896
    nonExpertBf16Params: 1_976_359_936,   // embed / head / norms / vision / aligner / compressor / hc_* 里的 BF16 部分
    nonExpertFixedBytes: 7_262_950_728,   // F32 169,229,128 + 原生 FP8 7,086,800,896 + E8M0 scale 6,920,704
    // 三条闭合校验,前两条引擎加载时会硬断言:
    //   1,976,359,936 × 2 + 7,262,950,728 = 11,215,670,600 B = 10.45 GiB
    //     = totalBytes − expertBytes − tpShardedBytes ✓
    //   557,171,343,360 ÷ (384 × 3 × 5120 × 2304) = 41(整数)✓ —— 但 41 是**等效层数**,见 expertLayerDesc
    //   557,171,343,360 + 196,928,504,320 + 9,105,468,114 = 763,205,315,794 总参数
    //     其中 backbone(去掉 engram / MTP / vision)= 551,566,180,464 ≈ model card 的「552B backbone」✓
  },
  // 引擎默认会把 expertParams ÷ 结构 的商当层数显示。这里那个商是 41,但它不是物理层数 ——
  // 40 层各 384 个专家,外加 3 个 DSpark 层各 128 个(3 × 128 = 384,恰好等价于一层)。
  expertLayerDesc: "41 层等效(40 层 × 384 + 3 个 DSpark 层 × 128)",

  // ---- 第三桶的文案 ----
  tpShardedLabel: "engram 哈希查表(按 TP 切,不按 PP 切)",
  tpShardedShort: "engram 查表",
  tpShardedNote: "字节数实测(2 × 384M 行 × 256 维 FP8);可疑的是放置方式 —— 「整张常驻 HBM」是猜测,量级 40%",

  // ---- 每请求固定占用的文案(滑窗环,不是 recurrent state)----
  stateNote: "滑窗环形 buffer:结构由 config 推出(window_size 128),但存储 dtype 是假设(BF16 或 FP8)",

  // ---- 量化 ----
  // 原生格式是 E2M1 + 每 32 元素一个 E8M0 scale(config 的 expert_dtype "fp4" + scale_fmt "ue8m0",
  // 实测 scale 形状 (2304,160) 对 in=5120 → block 32)—— 这正是 MXFP4 的布局,所以按机型目录的
  // 词汇写 "mxfp4",不写 "fp4"。词对不上会让 b200/b300 被误判成「无原生 FP4」。
  // 实测每参数 0.53125 B,与 K3 的 MXFP4 逐位相同。非 expert 是 FP8 E4M3 + 32×32 块 scale。
  quantFormat: "mxfp4",
  quantExcludes: "vision 塔与 aligner / embedding / lm_head / 全部 norms / compressor / indexer.wk / hc_* / engram 的 q,k_weight",

  // expert 权重可选格式。只给 convert.py 真正支持的两档(--expert-dtype fp4|fp8),
  // 不编造 BF16 那一档 —— 转换脚本没这个选项。bytesPerParam=null 表示「用实测字节数」。
  expertFormats: [
    { id: "mxfp4", label: "MXFP4 E2M1(原生)", bytesPerParam: null },
    // cast_e2m1fn_to_e4m3fn():1 B + 每 32×32 块一个 scale = 1.0009765625,
    // 与本 checkpoint 里原生 FP8 的 shared expert 实测值逐位相同。275.67 → 519.36 GiB。
    { id: "fp8",  label: "FP8 E4M3", bytesPerParam: 1.0009765625 },
  ],
  // 非 expert 是混合 dtype:大头(attention 的 wq/wkv/wo、shared expert)已经是 FP8,
  // 剩下 19.8 亿 BF16 参数是 embedding、lm_head、vision 塔、aligner、norms、compressor、hc_*。
  nonExpertFormats: [
    { id: "native", label: "原生混合(实测)",     bf16Bytes: 2 },
    { id: "fp8",    label: "BF16 部分也压成 FP8", bf16Bytes: 1 },
  ],

  // ---- 候选实例:业务判断,与用户逐个确认,不从 instance 目录自动推导(ADR-0005 §2)。
  // 与 K3 / GLM 两页对齐,只保留 P 系这 4 个,便于三个模型并排比较。
  // 本模型与前两个的关键差别:475.24 GiB 权重里有 189.13 GiB 只能按 TP 切,于是
  // **TP 有下限,且下限随单卡显存移动** —— p5en/p5e 上 TP 必须 ≥4,b200 上 ≥2,b300 上 ≥2
  // (b300 的 TP=1 差 5.56 GiB)。H100 80GB 一档整台都装不下 189 GiB ÷ 8 + 275 GiB ÷ 8,不在候选里。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // 起始切分取官方 inference/README.md 明写的 MP=8 —— 这是唯一有一手依据的切分。
  // 点一下旁边的 TP4/DP2 就能看到它在本页口径下其实不是最高的那档。
  defaultParallel: { n: 1, tp: 8, dp: 1, pp: 1, ep: 8 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  //
  // 名字里「不要」写绝对路数,也「不要」写「推荐 / 最优」(ADR-0008 §9 / ADR-0005)。
  // 本模型尤其不能标:最优 TP 随机型变(见下表),标它就是把一个随口径漂移的结论钉死。
  // 128K / util 0.90 / 原生 FP4 下手算(路数):
  //     机型            TP1      TP2      TP4     TP8
  //     p5en 1 台     装不下   装不下      532     487
  //     b200 1 台     装不下      500    1,134   1,020
  //     b300 1 台   差 5.56G    3,340    2,554   1,498
  // 三个机型的最优 TP 分别是 4 / 4 / 2,而 TP=1 在三个上全部装不下。
  // 这与 GLM 那页的教训方向**相反**:GLM 上 TP 越小越好(TP 复制 KV latent),
  // 这里 KV 只有 890 B/token,复制它几乎不疼,掐住你的是只能按 TP 切的 engram。
  // 所以名字只描述结构,括号里标该切分下 engram 被切成几份 —— 这是差异的机制,恒真。
  presets: [
    { name: "TP8/DP1(官方 MP=8,engram ÷8)", inst: "p5en.48xlarge",    n: 1, tp: 8, dp: 1, pp: 1, ep: 8  },
    { name: "TP4/DP2(engram ÷4)",            inst: "p5en.48xlarge",    n: 1, tp: 4, dp: 2, pp: 1, ep: 8  },
    // 这两个在 p5en 上装不下 —— 留着不是凑数:GLM 那页点 TP1×DP8 是并发跳 6 倍,
    // 这一页点它是直接红字。两页对照才看得出 engram 把选择空间砍掉了一半。
    { name: "TP2/DP4(engram ÷2)",            inst: "p5en.48xlarge",    n: 1, tp: 2, dp: 4, pp: 1, ep: 8  },
    { name: "TP1×DP8(engram 不切)",          inst: "p5en.48xlarge",    n: 1, tp: 1, dp: 8, pp: 1, ep: 8  },
    // b300 显存大到 TP=2 能塞进 engram 的一半,于是最优 TP 反而降下来
    { name: "b300 TP2/DP4(engram ÷2)",       inst: "p6-b300.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8  },
    // 两台:EP 翻倍摊薄 expert,但 engram 仍只跟 TP 走 —— 所以 TP 不能跟着降
    { name: "两台 TP4/DP4/EP16",              inst: "p5en.48xlarge",    n: 2, tp: 4, dp: 4, pp: 1, ep: 16 },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  notes: [
    `<b>权重全部是实测值,与 index.json 精确相等</b>。做法是对 48 个 shard 各发两个 Range 请求读 `
    + `safetensors header,把 <b>96,085</b> 个张量的 <code>dtype / shape / data_offsets</code> 求和:`
    + `字节合计 <code>510,286,023,000</code>,与 <code>index.json</code> 的 `
    + `<code>metadata.total_size</code> <b>精确相等</b> = 475.24 GiB。<br>`
    + `注意 HF <code>/api/models/</code> 的 <code>safetensors.parameters</code> 在本模型上`
    + `<b>对不上</b>(差 23.56 GB):它不认识 <code>F8_E8M0</code> 这个 scale dtype,`
    + `整整 23,563,015,184 个 scale 字节没被算进去。GLM 那次三方对账在这里只剩两方。`,

    `<b>expert 是 FP4,而 checkpoint 里的 shape 是骗人的。</b>expert 张量的 dtype 标记是 `
    + `<code>I8</code>,且 K 维只有真实维度的<b>一半</b>(<code>w1 (2304, 2560)</code> 实际是 `
    + `<code>(2304, 5120)</code>)—— 两个 FP4 值打包进一个字节。逻辑参数量必须 ×2 才对得上:`
    + `<code>2 × 278,585,671,680 = 557,171,343,360 = 15,744 × 3 × 2304 × 5120</code>,一个不差。<br>`
    + `<b>照字面读 shape 会把 expert 参数量低估整整一倍</b>。每参数实测 <code>0.53125</code> B `
    + `= FP4 的 0.5 + 每 32 元素一个 E8M0 scale,与 <code>quantization_config</code> 声明的 `
    + `<code>expert_dtype: fp4</code> + <code>scale_fmt: ue8m0</code> <b>精确吻合</b>。`,

    `<b>890 B/token 是先从参考实现的代码反推出来、再与 model card 对上的。</b>`
    + `<code>inference/model.py</code> 里:<code>compress_kv_cache</code> 只在 4 个 `
    + `<code>kv_source_layers</code> 上注册,形状是 <code>[bsz, max_seq_len ÷ compress_ratio, 512]</code>,`
    + `量化调用是 <code>fp4_act_quant(latent, 16, scale_dtype=float8_e4m3fn)</code> → `
    + `<code>512×0.5 + 512÷16 = 288 B</code>;indexer 的 <code>k_cache</code> 同样只在这 4 层上,`
    + `<code>fp4_act_quant(k, 32)</code>(E8M0)→ <code>128×0.5 + 128÷32 = 68 B</code>。<br>`
    + `<code>compress_ratios</code> 给出层 2/8/14 是 2、层 20 是 1,于是 `
    + `<code>(288+68) × (0.5+0.5+0.5+1) = 356 × 2.5 = <b>890 B/token</b></code>,`
    + `与 model card 的「<b>890 bytes per token</b>」<b>精确吻合</b>。一手件反推与官方声明独立对上,`
    + `所以这一项不需要任何估算 —— 比 K3 / GLM 的 KV 口径都硬。`,

    `<b>CSA2 是稀疏注意力,而这次它<u>确实</u>省了显存 —— 但省法与你想的不一样。</b>`
    + `GLM 那页写过「DSA 稀疏不省显存」:<code>index_topk</code> 决定每个 query 去<b>看</b>多少 token,`
    + `不是<b>存</b>多少。这一条在本模型上<b>仍然成立</b> —— <code>index_topk = 512</code> 同样不省显存。<br>`
    + `真正省下来的是另外三件事:(1) <b>跨层共享</b> —— 40 层只有 4 层持有 KV,其余层读同一份;`
    + `(2) <b>时间维压缩</b> —— encoder 那 3 层的 <code>compress_ratio = 2</code>,两个 token 池化成一个 latent;`
    + `(3) <b>FP4</b> —— latent 从 BF16 的 2 B/元素降到 0.5 B + scale。`
    + `三者叠起来才是 890 B/token(K3 是 27,648,GLM 是 11,616)。<b>稀疏度本身一分都没省。</b>`,

    `<b>滑窗环形 buffer 在短上下文下是主导项。</b>每层一个 128 槽的定长环(<code>window_size = 128</code>),`
    + `40 层合计 <b>5.000 MiB/请求</b>,<b>与上下文长度无关</b>。与 KV 相等的临界上下文是 `
    + `<code>5,242,880 ÷ 890 ≈ 5,891 tokens</code> —— <b>1K 上下文下它占单请求的 85%</b>,`
    + `漏掉它会让 1K 的并发虚高 6.75 倍。<br>`
    + `而且它<b>切不掉</b>:环里存的是所有 head 共享的 512 维 latent,每张卡一份完整副本,`
    + `行为与 KV cache 相同,与 GLM 的 per-head recurrent state <b>相反</b>。`,

    `<b>runtime 字节比 checkpoint 多 1.25 GiB,本页算的是 checkpoint。</b>`
    + `<code>convert.py</code> 转换时把 <code>wo_a</code> 从 FP8 <b>反量化成 BF16 落盘</b>`
    + `(源码注释自陈「an fp8 grouped GEMM would halve the memory」),另有 3 个 compressor 的 `
    + `<code>wkv</code> 提到 FP32,合计 <code>+1,341,431,552 + 15,728,640 ≈ +1.25 GiB</code>。<br>`
    + `本页继续算 checkpoint 字节 —— 它可与 <code>total_size</code> 对账,而 runtime 字节取决于`
    + `转换脚本的选择,换个引擎就变。1.25 GiB 在 475 GiB 里是 0.26%,但这是「实测」二字第一次出现裂缝。`,

    `<b>3 个 DSpark 层与 vision 塔都按「加载」算。</b>DSpark(投机解码的 draft 头)有自己的 128 个专家,`
    + `expert 桶里 <b>6.72 GiB</b> 是它们的;不开投机解码时引擎不会加载,滑窗环也少 3 层。`
    + `vision 塔 + aligner 共 <b>0.90 GiB</b>(BF16,4.12 亿参数)按常驻算;`
    + `图像 token 本身的显存(<code>vision_max_n_token = 1024</code>/图)<b>不算</b> —— 那属于 batch 构成。`,

    `<b>每卡 12 GiB overhead 那个猜测在本模型上比在前两个上更不牢。</b>`
    + `<code>hc_mult = 4</code>:残差流以 <b>4 份并行副本</b>携带(Single-Pass mHC),激活量直接 ×4;`
    + `再加 <code>index_topk = 512</code> 与 <code>candidate_topk_blocks 2048 × block 8</code> 的 `
    + `indexer scratch。同一个常数在三个模型上的可信度不一样,别把它当成被验证过的值。`,
  ],
};
