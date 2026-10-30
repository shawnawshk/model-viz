// Qwen3.8-Flash-Next —— Qwen/Qwen3.8-Flash-Next(model_type qwen4_exp,Qwen4 架构的预览版)
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数              derived  —— config.json(text_config)
//   权重字节数/参数量      derived  —— 逐张量读 131 个 shard 的 safetensors header 求和,
//                                    与 index.json 的 total_size、HF API 的 parameters 三方精确对账
//   FP8 那一档的字节数     derived  —— 官方 FP8 checkpoint(Qwen/Qwen3.8-Flash-Next-FP8)逐张量实测,expert 与 n-gram 表都有
//   n-gram 表放在哪       引擎参数 —— vLLM 默认 offload 到 host(ADR-0013),已提为界面输入
//   indexer key 的 dtype  estimated —— vLLM 对本模型的默认值 BF16(indexer_kv_dtype),是引擎实现,见 notes 第 3 条
//   GDN state dtype       estimated —— config.json 写 float32,但 vLLM 默认落到 BF16,同 Qwen3.8-27B
//   KV cache dtype        不属于模型 —— 引擎运行时选项,已提为界面输入,见 kvDtypes 与 ADR-0007
//
// 第一个 QSA 模型(ADR-0014):full-attention 层是 GQA + 一份按 TP 复制的压缩 indexer key。
// 第二个带「按 TP 切的查表」的模型(第一个是 DSv4.1 的 engram),所以「查表放置」这个输入对它同样起作用。
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["qwen3.8-flash-next"] = {
  name: "Qwen3.8-Flash-Next",
  hf: "Qwen/Qwen3.8-Flash-Next",
  vllmRecipe: "https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next",
  // 官方模型卡 pipeline_tag = image-text-to-text,「Causal Language Model with Vision Encoder」;repo 带 video_preprocessor_config.json。
  modalities: {
    input: ["text", "image", "video"], output: ["text"],
    sources: [
      "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md",
      "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/video_preprocessor_config.json",
    ],
  },
  blurb: "180.0B checkpoint 参数(125B 主干 + 51B n-gram 嵌入 + 4B MTP)· 6B 激活 · 48 层混合注意力(36 Gated DeltaNet 线性 + 12 QSA 稀疏)"
       + " · 512 routed experts / top-10 · GQA 2 kv head · 原生 BF16,官方另有 FP8 · 多模态 · 262K 原生上下文",
  blurbEn: "180.0B checkpoint parameters (125B backbone + 51B n-gram embedding + 4B MTP) · 6B activated · 48 hybrid-attention layers (36 Gated DeltaNet linear + 12 QSA sparse)"
       + " · 512 routed experts / top-10 · GQA 2 kv heads · native BF16, official FP8 also available · multimodal · 262K native context",
  hidden: 2560,                              // hidden_size,只用于 PP 的 send/recv 激活量(hc_count 4 时残差流其实是 4 × 2560,见 notes)

  // ---- 层结构 ----
  // layer_types 逐层数出来的:linear_attention 36 层、full_attention 12 层,「3 个线性 + 1 个 full」循环(full_attention_interval = 4)。
  layers: [
    { count: 12, family: "qsa", qHeads: 24, kvHeads: 2, headDim: 256,
      indexerElems: 128, indexerBytes: 2, compressRatio: 4,   // indexer_head_dim / vLLM 默认 BF16 / indexer_compress_ratio
      indexTopk: 2048,                                        // indexer_budget:512 个 micro-block × 4 token
      note: "QSA = GQA 主 KV(24 q / 2 kv head,head_dim 256,按 kv head 切)+ 一份压缩 indexer key "
          + "(每 4 个 token 一行 128 维,1 个共享 head → TP 切不掉)。decode 只读 top-2048 个 token 的主 KV;"
          + "vLLM 的 QSA 不带滑窗、不带 sink(遇到就抛 NotImplementedError)" },
    { count: 36, family: "linear", label: "Gated DeltaNet 线性注意力", labelEn: "Gated DeltaNet linear attention", heads: 48, kHeads: 16, headDim: 128, stateDtypeBytes: 4,
      note: "Gated DeltaNet。recurrent state 形状 [48 v-head × 128 × 128],与序列长度无关,per-head 可按 TP 切。"
          + "另有 kernel=4 的短卷积 state(36 层 × 10240 通道 × 3 × 2 B ≈ 2.1 MiB/请求),量级可忽略,未计入" },
  ],
  totalLayers: 48,

  // 边界声明里 indexer key cache 那一条的后半句(前半句由引擎拼)
  indexerAsm: `按 vLLM 对本模型的默认 <code>indexer_kv_dtype</code> = BF16 计(每 4 个 token 一行 128 维,只有 1 个 head,TP 切不掉)—— `
    + `vLLM recipe 的命令都传了 <code>--attention-config.indexer_kv_dtype fp8</code>,那样这一项减半;它不受 <code>--kv-cache-dtype</code> 影响`,
  indexerAsmEn: `per vLLM's default <code>indexer_kv_dtype</code> = BF16 for this model (one 128-dim row per 4 tokens, only 1 head, cannot be split by TP) — `
    + `the vLLM recipe commands all pass <code>--attention-config.indexer_kv_dtype fp8</code>, which halves this item; it is not affected by <code>--kv-cache-dtype</code>`,

  // ---- KV cache 的 dtype:不是模型属性,是引擎的运行时选项(--kv-cache-dtype)----
  // 理由同 ADR-0007。第一项即默认值。recipe 的 NVIDIA 命令都显式传了 --kv-cache-dtype fp8 —— 那是部署选择,不是默认值。
  kvDtypes: [
    { id: "bf16", label: "BF16(引擎默认)",   bytes: 2, labelEn: "BF16 (engine default)" },
    { id: "fp8",  label: "FP8(需显式开启)", bytes: 1, labelEn: "FP8 (opt-in)" },
  ],

  moe: {
    layers: 48,            // 48 层全是稀疏 MLP(「12 × (3 × (GDN → MoE) → 1 × (QSA → MoE))」)
    experts: 512, topk: 10, sharedExperts: 1,
    latent: 2560, inter: 640,    // expert 直接吃 hidden_size
  },
  expertLayerDesc: "49 层(48 层主干 + 1 层 MTP)",
  expertLayerDescEn: "49 layers (48 backbone + 1 MTP)",

  // ---- 权重:三桶实测字节(口径见 ADR-0009 §1)----
  weights: {
    totalBytes:  359_999_963_128,      // 1,658 个张量逐个求和,与 index.json metadata.total_size 精确相等 = 335.28 GiB
    // 桶 1:routed experts。checkpoint 里是融合的 gate_up_proj / down_proj,48 层 + MTP 1 层。
    expertBytes: 246_625_075_200,      // = 229.69 GiB,全 BF16
    expertParams:    123_312_537_600,  // 49 × 512 × 3 × 2560 × 640,与逐张量 numel 精确相等
    // 桶 2:n-gram 嵌入表(vLLM 叫 PLE)。常驻时按 TP 切(ETP 组 = TP 组)、不按 PP 切;vLLM 默认 offload 到 host。
    tpShardedBytes:  102_400_491_520,  // = 95.37 GiB(128 个 shard × 2,500,012 行 × 160 维 BF16)
    tpShardedParams:  51_200_245_760,  // 与 model card 的「51B n-gram embedding」一致
    // 桶 3:其余全部,全是 BF16。另有 3 个 I64 的哈希常量张量(35 个元素 = 280 B),不算参数。
    nonExpertParams:     5_487_198_064,
    nonExpertBf16Params: 5_487_198_064,   // 与官方 FP8 checkpoint 里保留 BF16 的参数数逐个相同 —— FP8 版不动这一桶
    nonExpertFixedBytes: 280,             // ple_embedding 的 ngram_heads_offsets / ngram_heads_vocab_sizes / layer_multipliers(I64)
    // 三条闭合校验,前两条引擎加载时会硬断言:
    //   5,487,198,064 × 2 + 280 = 10,974,396,408 B = 10.22 GiB = totalBytes − expertBytes − tpShardedBytes ✓
    //   123,312,537,600 ÷ (512 × 3 × 2560 × 640) = 49(整数层数)✓
    //   expertParams + tpShardedParams + nonExpertParams = 179,999,981,424 = HF API 的 safetensors.parameters.BF16 ✓
    nonMatmulParams: 1_084_630_256,       // embedding 635,699,200 + vision 塔 448,931,056:不参与 decode 矩阵乘,只用于标注算力行的高估比例
  },
  nonExpertFixedDesc: "n-gram 哈希用的 3 个 I64 常量张量,280 B",
  nonExpertFixedDescEn: "3 I64 constant tensors used for n-gram hashing, 280 B",

  // ---- 第三桶的文案 ----
  tpShardedLabel: "n-gram 嵌入表(PLE;常驻时按 TP 切,不按 PP 切)",
  tpShardedLabelEn: "n-gram embedding table (PLE; sharded by TP when resident, not by PP)",
  tpShardedShort: "n-gram 表",
  tpShardedShortEn: "n-gram table",
  tpShardedNote: "字节数实测(128 个 shard 共 3.2 亿行 × 160 维 BF16;官方 FP8 版 47.68 GiB);vLLM 默认放 host,常驻 HBM 时按 TP 切",
  tpShardedNoteEn: "Byte count measured (128 shards, 320 million rows in total × 160 dims BF16; official FP8 build 47.68 GiB); vLLM offloads it to host by default, and when resident in HBM it is sharded by TP",
  // 常驻那一档的局限,显示在「常驻 HBM」那条 banner 里
  lookupHbmCaveat: `<b>这一档的路数是上限。</b>vLLM recipe 写着:4×H100 上 TP4 不开 offload 会在启动时 OOM,H100 上 TP / TEP 都必须 offload;`
    + `可按「÷TP 常驻」算,4×H100 TP4(FP8)每卡扣完 12 GiB overhead 还剩 16.8 GiB —— 说明常驻的真实开销比 ÷TP 更大。`
    + `另外 DEP(TP1 × DP)不支持常驻,只能 offload。`,
  lookupHbmCaveatEn: `<b>The number of concurrent requests at this setting is an upper bound.</b> The vLLM recipe states that on 4×H100, TP4 without offload runs out of memory at startup, and on H100 both TP and TEP must offload; `
    + `the figure can be computed as "resident ÷ TP", but on 4×H100 TP4 (FP8) each GPU has only 16.8 GiB left after the 12 GiB overhead — so the real cost of keeping the table resident is larger than ÷TP. `
    + `In addition, DEP (TP1 × DP) does not support a resident table and can only offload.`,

  // ---- 每请求固定占用的文案(GDN recurrent state)----
  stateNote: "结构由 config 推导(36 层 × 48 head × 128 × 128);dtype 按 config 的 float32,vLLM 默认 auto 会落到 BF16(减半)",
  stateNoteEn: "Structure derived from the config (36 layers × 48 heads × 128 × 128); dtype follows the config's float32, while vLLM's default auto falls back to BF16 (half the size)",

  // ---- 引擎不支持的 PP ----
  ppUnsupported: `vLLM recipe:N-gram Embedding 目前不支持 pipeline parallelism,启动会报错,只能用单机 TP / TEP。`,
  ppUnsupportedEn: `vLLM recipe: N-gram Embedding does not currently support pipeline parallelism and fails at startup; use single-node TP / TEP only.`,

  // ---- 量化 ----
  // 原生 checkpoint 是 BF16。官方 FP8 版的 quantization_config:fmt e4m3、weight_block_size [128,128],
  // modules_to_convert = ["ple.ple_embedding.ngram_embedding"],另有 943 条 modules_to_not_convert。
  quantFormat: "bf16",
  quantExcludes: "官方 FP8 版只量化 routed expert 与 n-gram 表;attention / Gated DeltaNet / shared expert / router / Gated Residual / vision / embedding / lm_head / norms 全部保留 BF16",

  // expert 权重可选格式。FP8 一档就是官方 FP8 checkpoint,逐张量实测:
  //   expert:FP8 e4m3 + 每 128×128 块一个 BF16 scale → 1 + 2/16384 = 1.0001220703125 B/参数,123,327,590,400 B,精确;
  //   n-gram 表:FP8,整张表只有一个 BF16 的 per-tensor scale(shape [1])→ 51,200,245,760 + 2 = 51,200,245,762 B。
  //   所以这一档要连第三桶一起换(tpShardedBytes),否则「常驻 + FP8」会多算 47.7 GiB ÷ TP。
  expertFormats: [
    { id: "bf16", label: "BF16(原生)", bytesPerParam: null, labelEn: "BF16 (native)" },
    { id: "fp8",  label: "FP8(官方 FP8 checkpoint,expert 与 n-gram 表都换)", bytesPerParam: 1.0001220703125,
      labelEn: "FP8 (official FP8 checkpoint: experts + n-gram table)",
      tpShardedBytes: 51_200_245_762, measured: true },
  ],
  // 非 expert 只有 10.22 GiB,而且官方 FP8 版不动它。第二档是推算(没有 checkpoint),留着只为与另几个模型的同名开关对齐。
  nonExpertFormats: [
    { id: "bf16", label: "BF16(原生,官方 FP8 版也不动)", bf16Bytes: 2, labelEn: "BF16 (native; official FP8 keeps it)" },
    { id: "fp8",  label: "BF16 部分也压成 FP8(推算)",     bf16Bytes: 1, dtype: "fp8", labelEn: "BF16 part also quantized to FP8 (extrapolated)" },
  ],

  // ---- 候选实例:业务判断,与用户逐个确认(ADR-0005 §2)。2026-10-30 确认:P 系 4 个 + p5(H100 80GB)。
  // 加 p5 的理由:offload 之后 240 GiB 的 BF16 权重一台 p5 就装得下,vLLM recipe 也单独给了 8×H100 的配置;
  // 而且「H100 上 n-gram 表必须 offload」正好是「查表放置」这个输入最有说服力的一个机型。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge", "p5.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // 起始切分取 vLLM recipe 给 8×H200 的配置:TEP8 = TP8 + EP8。recipe 说 FP8 checkpoint 上 plain TP8 起不来
  // (moe_intermediate_size 640 ÷ 8 = 80,不是 128 宽量化块的倍数),必须把 expert 交给 EP —— 本页的 EP 本来就是这个口径。
  defaultParallel: { n: 1, tp: 8, dp: 1, pp: 1, ep: 8 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  // 名字里「不要」写绝对路数,也「不要」写「推荐 / 最优」(ADR-0008 §9 / ADR-0005),路数由 markPreset() 现算。
  // 手算(BF16 KV、原生 BF16、每卡 12 GiB overhead;路数,依次为下面七个预设):
  //                       TEP8  TP4×DP2  TP2×DP4  TP1×DP8  p5 TEP8  p5 TP2×DP4  b300 TP2×DP4
  //   1K/0.90             3312    4308     4972     4688     1170      1604        12020
  //   128K/0.90             52     102      196      184       18        60          472
  //   1M/0.90                6      12       24       24        2         8           60
  //   128K/0.90 · 常驻       45      72       80        0       11         0          360
  // 切分故事是 Qwen3.8-27B 与 DSv4.1 两页的叠加:
  //   offload(默认)时,TP 的甜点在 n_kv_heads = 2 上 —— TP2 既切权重又不复制 KV,官方 TEP8 复制 4 份,只有它的约 1/4;
  //   常驻时,95 GiB 的 n-gram 表只能按 TP 切,TP1×DP8 直接装不下 —— DSv4.1 那条「TP 有下限」回来了。
  // 括号里标该切分下主 KV 被复制几份 —— 这是差异的机制,恒真。
  presets: [
    { name: "TEP8:TP8/DP1(官方 H200 配置,KV ×4)", nameEn: "TEP8: TP8/DP1 (official H200 config, KV ×4)", inst: "p5en.48xlarge",    n: 1, tp: 8, dp: 1, pp: 1, ep: 8 },
    { name: "TP4×DP2(KV ×2)", nameEn: "TP4×DP2 (KV ×2)",                      inst: "p5en.48xlarge",    n: 1, tp: 4, dp: 2, pp: 1, ep: 8 },
    { name: "TP2×DP4(TP = n_kv_heads,KV ×1)", nameEn: "TP2×DP4 (TP = n_kv_heads, KV ×1)",      inst: "p5en.48xlarge",    n: 1, tp: 2, dp: 4, pp: 1, ep: 8 },
    // 非 expert 只有 10 GiB,复制 8 份也便宜;但它对「查表放置」最敏感:常驻时每卡背整张 95 GiB 的表
    { name: "TP1×DP8(KV ×1;常驻时 n-gram 表不切)", nameEn: "TP1×DP8 (KV ×1; n-gram table not sharded when resident)", inst: "p5en.48xlarge",    n: 1, tp: 1, dp: 8, pp: 1, ep: 8 },
    // recipe 的 8×H100 配置:H100 上 n-gram 表必须 offload
    { name: "p5 H100 TEP8(官方 H100 配置)", nameEn: "p5 H100 TEP8 (official H100 config)",          inst: "p5.48xlarge",      n: 1, tp: 8, dp: 1, pp: 1, ep: 8 },
    { name: "p5 H100 TP2×DP4(KV ×1)", nameEn: "p5 H100 TP2×DP4 (KV ×1)",               inst: "p5.48xlarge",      n: 1, tp: 2, dp: 4, pp: 1, ep: 8 },
    { name: "b300 TP2×DP4(KV ×1)", nameEn: "b300 TP2×DP4 (KV ×1)",                  inst: "p6-b300.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8 },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  notes: [
    `<b>权重全部是实测值,三方对账。</b>对 131 个 shard 逐个发 Range 请求读 safetensors header,把 <b>1,658</b> 个张量的 `
    + `<code>dtype / shape / data_offsets</code> 求和:字节合计 <code>359,999,963,128</code>,与 <code>index.json</code> 的 `
    + `<code>metadata.total_size</code> <b>精确相等</b> = 335.28 GiB(vLLM recipe 写的也是 335.28 GiB);参数合计 `
    + `<code>179,999,981,424</code>,与 HF API 的 <code>safetensors.parameters.BF16</code> <b>精确相等</b>,另有 35 个 I64 元素(n-gram 哈希常量)。<br>`
    + `构成:routed expert <b>229.69 GiB(68.5%)</b>、n-gram 表 <b>95.37 GiB(28.4%)</b>、其余 <b>10.22 GiB(3.0%)</b>。`
    + `最后这一桶里:Gated DeltaNet 投影 3.89、embedding + lm_head 2.37(<code>tie_word_embeddings = false</code>)、`
    + `Gated Residual 的 hyper-connection 1.19、QSA 投影 1.15、vision 塔 0.84、shared expert + router 0.56 GiB。`
    + `model card 的「125B + 51B n-gram + 4B MTP」合计 180B,与逐张量求和一致。`,

    `<b>「expert → FP8」那一档就是官方 FP8 checkpoint,逐张量实测。</b><code>Qwen/Qwen3.8-Flash-Next-FP8</code> 的 152,089 个张量同样逐个读了,`
    + `合计 <code>185,502,232,570</code> B = 172.76 GiB(与它的 <code>index.json</code> 精确相等;recipe 写 172.78 GiB,差 0.02 GiB,来源未查)。`
    + `它的 <code>modules_to_convert</code> 只有 n-gram 表,再加上全部 routed expert:expert 是 FP8 + 每 128×128 块一个 <b>BF16</b> scale,`
    + `每参数 <code>1 + 2/16384 = 1.0001220703125</code> B,精确;n-gram 表是 FP8,整张表<b>只有一个</b> BF16 的 per-tensor scale(shape <code>[1]</code>),`
    + `<code>51,200,245,760 + 2</code> B = 47.68 GiB。其余 5,487,198,064 个参数与 BF16 版<b>逐个相同,保留 BF16</b>。<br>`
    + `所以本页把这一档做成「换 checkpoint」:expert 与 n-gram 表一起换。n-gram 表默认在 host 里,这只在你把它切成常驻时才影响 HBM。`,

    `<b>QSA 的每 token KV:主 KV 24,576 B(BF16)+ indexer key 768 B,两部分切法相反。</b>`
    + `主 KV 是 GQA:<code>12 层 × 2 (K+V) × 2 kv head × 256 × 2 B = 24,576 B</code>,按 kv head 切,复制因子 <code>max(1, TP ÷ 2)</code> —— `
    + `kv head 只有 2 个,TP 的甜点在 2 上,比 Qwen3.8-27B(4 个 kv head)还低一档。`
    + `indexer key 是 vLLM 的 <code>QSACompressedKeyCache</code>:每 4 个 token 一行、128 维、<b>只有 1 个共享 head</b>,`
    + `<code>12 × 128 × 2 B ÷ 4 = 768 B/token</code>,每张卡一份完整副本(TP 切不掉),dtype 走 <code>indexer_kv_dtype</code>,`
    + `vLLM 对本模型的默认值是 BF16(<code>resolve_indexer_kv_dtype("bf16")</code>),与 <code>--kv-cache-dtype</code> 无关。<br>`
    + `vLLM 还给每层留一个 4 槽的原始 key 环(MRoPE 时每槽 140 个 BF16),12 层合计约 13 KiB/请求,量级可忽略,未计入。见 <code>docs/adr/0014</code>。`,

    `<b>QSA 稀疏省的是读,不是存。</b>全部 token 的主 KV 都留在 cache 里(显存与普通 GQA 一样),但 decode 每步只读 top-2048 个 token `
    + `(<code>indexer_budget</code>,512 个 micro-block × 4 token),再加 indexer 扫一遍全部压缩 key。128K 下每请求每步(TP1):`
    + `indexer <code>32,768 行 × 256 B × 12 层 = 96 MiB</code> + 主 KV <code>2,048 × 2,048 B × 12 层 = 24 MiB</code>,`
    + `全量读则是 3 GiB,差 25 倍。这与 DSA / CSA2 的道理相同,roofline 走 <code>decodeReadPerReq</code> 钩子,不按全量 KV 算。<br>`
    + `也因此 attention 本身的 FLOPs 在这个模型上不大:每 token 只对 2,048 个 token 做 attention,约 0.6 GFLOP(indexer 打分在 128K 下再加约 0.4),`
    + `而矩阵乘按 6B 激活参数约 12 GFLOP —— 与 Qwen3.8-27B 上「attention 与矩阵乘同量级」那条待办方向相反。`,

    `<b>Gated DeltaNet 的 recurrent state = 108.0 MiB/请求,与上下文无关。</b><code>36 层 × 48 v-head × 128 × 128 × 4 B = 113,246,208 B</code>。`
    + `dtype 与 Qwen3.8-27B 同样有两个答案:<code>config.json</code> 写 <code>mamba_ssm_dtype: "float32"</code>,本页按 FP32 计;`
    + `vLLM <code>--mamba-ssm-cache-dtype auto</code> 对 Gated DeltaNet 落到模型 dtype(BF16),那样是 54 MiB。<br>`
    + `与 KV 相等的临界上下文:TP1 时 <code>113,246,208 ÷ 25,344 ≈ 4,468 tokens</code>;官方 TEP8 时 state ÷8 而主 KV 只 ÷2,降到约 1,084 tokens。`
    + `1K 上下文下 TEP8 的单请求里 state 占 51%,所以那一档 dtype 在短上下文下能改并发约 1.3 倍,128K 下不到 1%。`
    + `另有两个短卷积 state(Gated DeltaNet 约 2.1 MiB、PLE 的空洞卷积约 0.2 MiB/请求),未计入。`,

    `<b>n-gram 嵌入表(vLLM 叫 PLE):51.2B 参数,只挂在 layer 2,默认根本不进 HBM。</b>`
    + `16 个哈希 head(bigram + trigram 各 8 个)× 每 head 约 2,000 万个质数大小的桶 = 320,001,536 行 × 160 维;每 token 查 16 行,共 5 KiB。`
    + `vLLM 的实现:表按 ETP 组(默认 = TP 组)做 vocab-parallel 切分;<code>EngramConfig.cpu_offload</code> 默认 <code>true</code>,`
    + `整张表放在 pinned host memory 上,在前一层计算时预取。recipe 写明:H100 上必须 offload,H200 及以上可选;`
    + `host 内存要「至少 51 GB 再加运行时余量」(FP8 版)。<br>`
    + `PLE 层自己的 <code>key_proj / value_proj</code>(3,277 万参数)在 vLLM 里不按 TP 切(<code>disable_tp=True</code>),本页按 ÷TP 算,差 0.06 GiB,可忽略。`,

    `<b>官方配置与本页数字的对照。</b>recipe 给 8×H200 的是 <b>TEP8</b>(TP8 + EP8,FP8 checkpoint、FP8 KV、FP8 indexer、MTP 3),`
    + `给 4×GB300 的是 TEP4;vLLM 的 CI 精度测试用 FP8 + TP4。本页默认停在 TEP8 + BF16 原生权重 + 引擎默认 dtype,`
    + `点 TP2×DP4 那个预设看 KV 不再复制的效果。<br>`
    + `<b>本页不建模的两条 vLLM 限制</b>:PP 目前起不来(选 PP&gt;1 会出红字);DEP(TP1 × DP)必须 offload。`,

    `<b>每卡 12 GiB overhead 这个猜测在本模型上更不牢。</b>Gated Residual 的 <code>hc_count = 4</code>:残差流是 4 份并行的 2560 维(共 10240 维),`
    + `与 DSv4.1 的 <code>hc_mult = 4</code> 同类,激活峰值本该更大;recipe 的命令都开了 MTP 3 个 draft token,还要再加 draft 的激活与状态。`
    + `而 TEP8 时每卡权重只有 30 GiB,overhead 是它的 40%。唯一一档实测(<code>docs/measurements.md</code> M-001,GLM-5.3-Flash)<b>不能搬到这里</b>。`,
  ],
  notesEn: [
    `<b>All weights are measured values, reconciled three ways.</b> The safetensors header of each of the 131 shards was read with a Range request, and the `
    + `<code>dtype / shape / data_offsets</code> of <b>1,658</b> tensors were summed: total bytes <code>359,999,963,128</code>, `
    + `<b>exactly equal</b> to <code>metadata.total_size</code> in <code>index.json</code> = 335.28 GiB (the vLLM recipe also states 335.28 GiB); total parameters `
    + `<code>179,999,981,424</code>, <b>exactly equal</b> to <code>safetensors.parameters.BF16</code> from the HF API; in addition there are 35 I64 elements (n-gram hash constants).<br>`
    + `Breakdown: routed experts <b>229.69 GiB (68.5%)</b>, n-gram table <b>95.37 GiB (28.4%)</b>, everything else <b>10.22 GiB (3.0%)</b>. `
    + `Within that last bucket: Gated DeltaNet projections 3.89, embedding + lm_head 2.37 (<code>tie_word_embeddings = false</code>), `
    + `Gated Residual hyper-connection 1.19, QSA projections 1.15, vision tower 0.84, shared expert + router 0.56 GiB. `
    + `The model card's "125B + 51B n-gram + 4B MTP" adds up to 180B, consistent with the tensor-by-tensor sum.`,

    `<b>The "expert → FP8" option is the official FP8 checkpoint, measured tensor by tensor.</b> The 152,089 tensors of <code>Qwen/Qwen3.8-Flash-Next-FP8</code> were read the same way, `
    + `totaling <code>185,502,232,570</code> B = 172.76 GiB (exactly equal to its <code>index.json</code>; the recipe states 172.78 GiB, a 0.02 GiB difference whose source was not checked). `
    + `Its <code>modules_to_convert</code> lists only the n-gram table; on top of that, every routed expert is quantized: experts are FP8 + one <b>BF16</b> scale per 128×128 block, `
    + `<code>1 + 2/16384 = 1.0001220703125</code> B per parameter, exact; the n-gram table is FP8 with <b>only one</b> BF16 per-tensor scale (shape <code>[1]</code>), `
    + `<code>51,200,245,760 + 2</code> B = 47.68 GiB. The other 5,487,198,064 parameters are <b>identical, one by one, to the BF16 build and stay BF16</b>.<br>`
    + `So this page models this option as "switching checkpoint": experts and the n-gram table are switched together. The n-gram table is in host memory by default, so this affects HBM only when you set it to resident.`,

    `<b>QSA KV per token: main KV 24,576 B (BF16) + indexer key 768 B, and the two are sharded in opposite ways.</b> `
    + `The main KV is GQA: <code>12 layers × 2 (K+V) × 2 kv head × 256 × 2 B = 24,576 B</code>, sharded by kv head, with replication factor <code>max(1, TP ÷ 2)</code> — `
    + `there are only 2 kv heads, so the TP sweet spot is at 2, one step lower than Qwen3.8-27B (4 kv heads). `
    + `The indexer key is vLLM's <code>QSACompressedKeyCache</code>: one row per 4 tokens, 128 dims, <b>only 1 shared head</b>, `
    + `<code>12 × 128 × 2 B ÷ 4 = 768 B/token</code>, one full copy on every GPU (TP cannot split it); its dtype follows <code>indexer_kv_dtype</code>, `
    + `which vLLM defaults to BF16 for this model (<code>resolve_indexer_kv_dtype("bf16")</code>), independent of <code>--kv-cache-dtype</code>.<br>`
    + `vLLM also keeps a 4-slot raw key ring per layer (140 BF16 per slot with MRoPE), about 13 KiB/request across 12 layers, negligible and not counted. See <code>docs/adr/0014</code>.`,

    `<b>QSA sparsity saves reads, not storage.</b> The main KV of all tokens stays in the cache (GPU memory is the same as for plain GQA), but each decode step reads only the top-2048 tokens `
    + `(<code>indexer_budget</code>, 512 micro-blocks × 4 tokens), plus the indexer scanning all compressed keys. At 128K, per request per step (TP1): `
    + `indexer <code>32,768 rows × 256 B × 12 layers = 96 MiB</code> + main KV <code>2,048 × 2,048 B × 12 layers = 24 MiB</code>, `
    + `versus 3 GiB for a full read, a 25x difference. The reasoning is the same as for DSA / CSA2: the roofline goes through the <code>decodeReadPerReq</code> hook and does not use the full KV size.<br>`
    + `As a result, attention FLOPs themselves are small for this model: each token attends to only 2,048 tokens, about 0.6 GFLOP (indexer scoring adds about 0.4 more at 128K), `
    + `while the matmuls take about 12 GFLOP for 6B activated parameters — the opposite direction from the open to-do on Qwen3.8-27B that "attention and matmul are of the same order".`,

    `<b>The Gated DeltaNet recurrent state = 108.0 MiB/request, independent of context.</b> <code>36 layers × 48 v-head × 128 × 128 × 4 B = 113,246,208 B</code>. `
    + `As with Qwen3.8-27B, the dtype has two answers: <code>config.json</code> says <code>mamba_ssm_dtype: "float32"</code>, which this page uses (FP32); `
    + `vLLM <code>--mamba-ssm-cache-dtype auto</code> falls back to the model dtype (BF16) for Gated DeltaNet, which gives 54 MiB.<br>`
    + `The context at which state equals KV: at TP1, <code>113,246,208 ÷ 25,344 ≈ 4,468 tokens</code>; with the official TEP8 the state is ÷8 while the main KV is only ÷2, so it drops to about 1,084 tokens. `
    + `At 1K context, state is 51% of a single request under TEP8, so this dtype choice can change concurrency by about 1.3x at short context and by less than 1% at 128K. `
    + `There are also two short-convolution states (about 2.1 MiB for Gated DeltaNet and about 0.2 MiB/request for the PLE dilated convolution), not counted.`,

    `<b>The n-gram embedding table (called PLE in vLLM): 51.2B parameters, attached only to layer 2, and by default it never enters HBM.</b> `
    + `16 hash heads (8 each for bigram and trigram) × about 20M prime-sized buckets per head = 320,001,536 rows × 160 dims; each token looks up 16 rows, 5 KiB in total. `
    + `The vLLM implementation: the table is vocab-parallel sharded over the ETP group (default = TP group); <code>EngramConfig.cpu_offload</code> defaults to <code>true</code>, `
    + `so the whole table sits in pinned host memory and is prefetched while the previous layer computes. The recipe states: offload is required on H100 and optional on H200 and above; `
    + `host memory must be "at least 51 GB plus runtime headroom" (FP8 build).<br>`
    + `The PLE layer's own <code>key_proj / value_proj</code> (32.77M parameters) is not sharded by TP in vLLM (<code>disable_tp=True</code>); this page computes it as ÷TP, a 0.06 GiB difference, negligible.`,

    `<b>Official configuration versus the numbers on this page.</b> For 8×H200 the recipe gives <b>TEP8</b> (TP8 + EP8, FP8 checkpoint, FP8 KV, FP8 indexer, MTP 3), `
    + `for 4×GB300 it gives TEP4; the vLLM CI accuracy test uses FP8 + TP4. This page defaults to TEP8 + native BF16 weights + engine default dtypes; `
    + `click the TP2×DP4 preset to see the effect of KV no longer being replicated.<br>`
    + `<b>Two vLLM limits that this page does not model</b>: PP cannot start up today (selecting PP&gt;1 shows red text); DEP (TP1 × DP) must offload.`,

    `<b>The per-GPU 12 GiB overhead guess is even less solid for this model.</b> Gated Residual has <code>hc_count = 4</code>: the residual stream is 4 parallel copies of 2560 dims (10240 dims in total), `
    + `of the same kind as <code>hc_mult = 4</code> in DSv4.1, so the activation peak should be larger; the recipe commands all enable MTP with 3 draft tokens, which adds the draft's activations and state. `
    + `And with TEP8 the weights are only 30 GiB per GPU, so overhead is 40% of that. The only measurement (<code>docs/measurements.md</code> M-001, GLM-5.3-Flash) <b>cannot be carried over here</b>.`,
  ],
};
