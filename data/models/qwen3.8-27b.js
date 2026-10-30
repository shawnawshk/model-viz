// Qwen3.8-27B —— Qwen/Qwen3.8-27B
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数              derived  —— config.json(text_config)
//   权重字节数/参数量      derived  —— 逐张量读 18 个 shard 的 safetensors header 求和,
//                                    与 index.json 的 total_size、HF API 的 parameters 三方精确对账
//   FP8 那一档的字节数     derived  —— **官方 FP8 checkpoint(Qwen/Qwen3.8-27B-FP8)逐张量实测**,
//                                    不是按参数量推算。本项目第一次在「非 expert 权重」开关上拿到实测值
//   GDN state dtype       estimated —— config.json 写 float32,但 vLLM 默认落到 BF16,见 notes 第 4 条
//   KV cache dtype        不属于模型 —— 引擎运行时选项,已提为界面输入,见 kvDtypes 与 ADR-0007
//
// 这是第一个 dense 模型(没有 moe 块)、第一个 GQA 模型(KV 复制因子随 TP 变),两件事都要引擎配合,见 ADR-0012。
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["qwen3.8-27b"] = {
  name: "Qwen3.8-27B",
  hf: "Qwen/Qwen3.8-27B",
  vllmRecipe: "https://recipes.vllm.ai/Qwen/Qwen3.8-27B",
  // 官方模型卡 Vision-Language Understanding 与图像/视频输入示例;pipeline_tag=image-text-to-text。
  modalities: {
    input: ["text", "image", "video"], output: ["text"],
    sources: ["https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md"],
  },
  blurb: "27.8B dense(无 MoE,全部激活)· 64 层混合注意力(48 Gated DeltaNet 线性 + 16 Gated Attention,3:1)"
       + " · GQA 24 q / 4 kv head,head_dim 256 · 原生 BF16,官方另有 FP8 · 多模态 · 262K 原生上下文(YaRN 至 1M)",
  blurbEn: "27.8B dense (no MoE, all params activated) · 64-layer hybrid attention (48 Gated DeltaNet linear + 16 Gated Attention, 3:1)"
         + " · GQA 24 q / 4 kv heads, head_dim 256 · native BF16, official FP8 also available · multimodal · 262K native context (YaRN up to 1M)",
  hidden: 5120,                              // hidden_size,只用于 PP 的 send/recv 激活量

  // ---- 层结构 ----
  // layer_types 逐层数出来的:linear_attention 48 层、full_attention 16 层,
  // 排布是「3 个线性 + 1 个 full」循环(full_attention_interval = 4,full 层在 3,7,11,…,63)。
  layers: [
    { count: 16, family: "gqa", qHeads: 24, kvHeads: 4, headDim: 256,
      note: "Gated Attention = GQA + 输出门。KV 按 kv head 切:TP ≤ 4 时每卡只存自己那 4/TP 个 head,"
          + "TP 组合起来恰好一份;TP=8 时每卡 1 个 head、复制 2 份。partial_rotary_factor 0.25 → RoPE 只占 64 维,"
          + "不影响 KV 字节数(K 整个 256 维都进 cache)" },
    { count: 48, family: "linear", label: "Gated DeltaNet 线性注意力", labelEn: "Gated DeltaNet linear attention", heads: 48, kHeads: 16, headDim: 128, stateDtypeBytes: 4,
      note: "Gated DeltaNet。recurrent state 形状 [48 v-head × 128 k-dim × 128 v-dim],与序列长度无关,"
          + "per-head 可按 TP 切分,不复制。另有 kernel=4 的短卷积 state(48 层 × 10240 通道 × 3 × 2 B ≈ 2.8 MiB/请求),量级可忽略,未计入" },
  ],
  totalLayers: 64,

  // ---- KV cache 的 dtype:不是模型属性,是引擎的运行时选项(--kv-cache-dtype)----
  // 理由与 K3 / GLM 相同,见 ADR-0007。第一项即默认值(auto = 模型 dtype = BF16)。
  // vLLM 官方 recipe 的所有启动命令都显式传了 --kv-cache-dtype fp8,并明说「fp8 KV is a choice here, not a requirement」。
  kvDtypes: [
    { id: "bf16", label: "BF16(引擎默认)",   labelEn: "BF16 (engine default)", bytes: 2 },
    { id: "fp8",  label: "FP8(需显式开启)", labelEn: "FP8 (opt-in)",          bytes: 1 },
  ],

  // 没有 moe 块 = dense 模型(ADR-0012)。引擎据此隐藏 EP 与 expert 权重两个控件,expert 桶恒为 0。

  // ---- 权重:实测字节(口径见 ADR-0008 / 0009;dense 模型只有「非 expert」这一桶)----
  weights: {
    totalBytes:  55_562_855_904,       // 1,199 个张量逐个求和,与 index.json metadata.total_size 精确相等 = 51.75 GiB
    expertBytes: 0, expertParams: 0,   // dense:没有 routed expert
    nonExpertParams:     27_781_427_952,   // 全部 BF16,与 HF API safetensors.parameters.BF16 精确相等;**零 F32 张量**
    // 下面两个字段在本模型上的含义与 K3 / GLM 不同:切分线不是「BF16 vs F32/原生 FP8」,
    // 而是「官方 FP8 checkpoint 量化了哪些 / 保留了哪些」—— 这样 FP8 那一档才能与实测对上。
    nonExpertBf16Params: 24_699_207_680,   // 官方 FP8 会量化的部分:64 层 MLP + 16 层 attention 投影 + 48 层 GDN 的 qkv/z/out 投影 + MTP 的同类,共 407 个张量
    nonExpertFixedBytes:  6_164_440_544,   // 3,082,220,272 × 2 B:embed / lm_head / vision 塔 / 全部 norm / GDN 的 A_log·dt_bias·conv1d·in_proj_a·in_proj_b —— 官方 FP8 也保留 BF16
    // 两条闭合校验,引擎加载时会硬断言:
    //   24,699,207,680 × 2 + 6,164,440,544 = 55,562,855,904 = totalBytes ✓
    //   expertBytes = expertParams = 0(dense)✓
    // 第三条(FP8 档)见 nonExpertFormats:24,699,207,680 × 1.0001220703125 + 6,164,440,544 = 30,866,663,264 = FP8 checkpoint 逐张量求和 ✓
    nonMatmulParams: 1_732_128_496,        // embedding 1,271,398,400 + vision 塔 460,730,096:不参与 decode 矩阵乘。
                                           // **只用于在 roofline 口径里标注算力行的高估比例(6.2%),不从 FLOPs 里扣** ——
                                           // 扣了就与另三个模型(没有这个字段、一律含 embedding)口径不一致,ADR-0010 当时选的是「含,高估不到 3%」
  },
  nonExpertFixedDesc: "embed / lm_head / vision / norms,官方 FP8 也保留 BF16",
  nonExpertFixedDescEn: "embed / lm_head / vision / norms, kept in BF16 even in the official FP8 checkpoint",
  nonExpertFixedDtype: "bf16",             // 上面那部分是 BF16,roofline 算力行按 BF16 算(缺省会按 FP8/F32 算)

  // ---- 量化 ----
  // 原生 checkpoint 是 BF16,没有量化。字段值用机型目录 nativeDtypes 的词汇,所有机型都有原生 BF16,不会触发告警。
  quantFormat: "bf16",
  // 官方 FP8 checkpoint 的 quantization_config.modules_to_not_convert(882 条)归纳:
  quantExcludes: "vision 塔全部 27 层 / embedding / lm_head / 全部 norms / GDN 的 A_log·dt_bias·conv1d·in_proj_a·in_proj_b",

  // 没有 expertFormats:dense 模型没有 expert 桶。
  // 「非 expert 权重」两档都是实测:
  //   BF16 —— 本 checkpoint;
  //   FP8  —— Qwen/Qwen3.8-27B-FP8,fmt e4m3、weight_block_size [128,128]、scale 是 **BF16**(GLM 是 F32),
  //           故量化部分每参数 1 + 2/(128×128) = 1.0001220703125 B,精确;总字节 30,866,663,264 = 28.75 GiB。
  //           不是「一半」:6.16 GiB 的 embed / lm_head / vision 官方也没量化,所以只省 44%。
  nonExpertFormats: [
    { id: "bf16", label: "BF16(原生 checkpoint)",           labelEn: "BF16 (native checkpoint)", bf16Bytes: 2 },
    { id: "fp8",  label: "FP8 E4M3(官方 FP8 checkpoint,实测)", labelEn: "FP8 E4M3 (official FP8 checkpoint)", bf16Bytes: 1.0001220703125, dtype: "fp8" },
  ],

  // ---- 候选实例:业务判断(ADR-0005 §2)。2026-09-16 由本次会话拟定,未逐个与用户确认,可删。
  // 前三个模型都是 P 系那 4 个;这个模型 51.75 GiB 权重一张卡就装得下,G 系与 H100 80GB 才是它真正会被部署的地方,
  // 所以在 4 个 P 系(便于四个模型并排比较)之外加了 p5 / g7e:
  //   p5(H100 80GB):TP4×DP2 @128K/0.90 = 46 路,TP1×DP8 只剩 8 路(每卡 63.75 GiB 权重 + overhead,只剩 8 GiB 给 KV)
  //   g7e(RTX PRO 6000 96GB,无 NVLink):TP 上限 1,TP1×DP8 @128K/0.90 = 16 路 —— ADR-0001 第一次实际咬人的机型。
  //       余量 22.6(BF16)/ 45.7(FP8)GiB,远大于 overhead 假设的误差,所以那个猜测在这台机器上
  //       翻不了「装得下」的结论,只影响路数(12 → 6.98 GiB 会让 128K 从 16 变 24 路)。
  //
  // 2026-09-17 移除 g6e(L40S 44.7GB)。理由不是「它装不下」,而是**本工具在这台机器上答不了**:
  //   util 0.90 的预算只有 40.23 GiB,官方 FP8 权重 28.75 GiB,余量 11.5 GiB —— 而 2026-09-16 在 p5en 上
  //   实测到的非权重非 KV 开销是 6.98 GiB(non-torch 1.74 + 激活峰值 4.04 + CUDA graph 1.20),
  //   与本页硬写的 12 GiB 差 5 GiB,而这 5 GiB 恰好跨过 g6e 的生死线:12 GiB 下判「装不下,0 路」,
  //   6.98 GiB 下 1K 上下文是 176 路。红字的方向由一个未校准的常量单独决定,而摆一台答不了的机器
  //   配一个自信的红字是最坏的选项,所以先撤掉。
  //   注意那 6.98 是 GLM / TP8 / p5en 的值,**不能直接搬到 Qwen / TP1 / g6e** —— TP1 没有 NCCL buffer 那 1.74,
  //   但激活不分片又可能更大。所以这不是「换个数就对了」,是「这台机器上没有可信的数」。
  //   等 overhead 提为界面输入(实测细节见 docs/measurements.md M-001)再考虑加回。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge",
    "p5.48xlarge", "g7e.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // 起始切分故意选「传统」的 TP8/DP1(8 卡机器上的反射动作):TP 超过 n_kv_heads=4 之后 KV 开始复制 2 份,
  // 页面一打开就有那条告警;点一下旁边的 TP4×DP2 看并发接近翻倍(128K/0.90 下 53 → 100 路)。
  defaultParallel: { n: 1, tp: 8, dp: 1, pp: 1, ep: 1 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  //
  // 名字里「不要」写绝对路数,也「不要」写「推荐 / 最优」(ADR-0008 §9 / ADR-0005)。路数由 markPreset() 按当前口径现算。
  // 手算(BF16 KV、原生 BF16 权重、每卡 12 GiB overhead;路数,依次为下面六个预设):
  //     1K/0.90    3265  4014  3504  2480  1852   888
  //   128K/0.90      53   100    84    56    46    16
  //     1M/0.90       6    12     8     0     4     0    ← TP1×DP8 在 1M 下装不下(单路 64 GiB KV)
  // 三个口径下 TP4×DP2 都是 p5en 上最高的一档,但这是「本页口径下」的结论,不写进名字。
  //
  // 这个模型的切分权衡是第四种方向:K3 要 TP(非 expert 权重大)、GLM 要 DP(TP 复制 latent)、
  // DSv4.1 的 TP 有下限(engram 只能按 TP 切);这里 **TP 的甜点在 n_kv_heads 上** ——
  // TP ≤ 4 时 KV 按 head 切、不复制,TP 同时还切权重,所以 TP 越大越好;TP=8 越过 4 个 kv head,
  // KV 开始复制 2 份、DP 少一半,并发掉回 TP1 的水平。括号里标该切分下 KV 被复制几份 —— 这是差异的机制,恒真。
  presets: [
    { name: "TP8/DP1(TP > n_kv_heads,KV ×2)",   nameEn: "TP8/DP1 (TP > n_kv_heads, KV ×2)", inst: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 1 },
    { name: "TP4×DP2(TP = n_kv_heads,KV ×1)",   nameEn: "TP4×DP2 (TP = n_kv_heads, KV ×1)", inst: "p5en.48xlarge", n: 1, tp: 4, dp: 2, pp: 1, ep: 1 },
    { name: "TP2×DP4(KV ×1,权重 ×4)",           nameEn: "TP2×DP4 (KV ×1, weights ×4)", inst: "p5en.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 1 },
    { name: "TP1×DP8(KV ×1,权重 ×8)",           nameEn: "TP1×DP8 (KV ×1, weights ×8)", inst: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1 },
    // 80 GB 一档:同样的 TP4×DP2,预算从 126.9 降到 72 GiB,权重 + overhead 占掉 35%
    { name: "p5 H100 80GB TP4×DP2",               nameEn: "p5 H100 80GB TP4×DP2", inst: "p5.48xlarge",   n: 1, tp: 4, dp: 2, pp: 1, ep: 1 },
    // 无 NVLink:TP 只能是 1,每卡背满 51.75 GiB 权重 —— 这是 ADR-0001 在本项目里第一次实际起作用
    { name: "g7e TP1×DP8(无 NVLink,TP 上限 1)",  nameEn: "g7e TP1×DP8 (no NVLink, TP limit 1)", inst: "g7e.48xlarge",  n: 1, tp: 1, dp: 8, pp: 1, ep: 1 },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  notes: [
    `<b>权重全部是实测值,三方对账,而且是至今最干净的一份。</b>对 18 个 shard 逐个发 Range 请求读 safetensors header,`
    + `把 <b>1,199</b> 个张量的 <code>dtype / shape / data_offsets</code> 求和:字节合计 <code>55,562,855,904</code>,`
    + `与 <code>index.json</code> 的 <code>metadata.total_size</code> <b>精确相等</b> = 51.75 GiB;参数合计 `
    + `<code>27,781,427,952</code>,与 HF <code>/api/models/Qwen/Qwen3.8-27B</code> 的 <code>safetensors.parameters.BF16</code> `
    + `<b>精确相等</b>;<b>1,199 个张量全是 BF16,一个 F32 都没有</b>(K3 / GLM 的 norm 是 F32)。<br>`
    + `构成:64 层 MLP 31.88 GiB(61.6%)、48 层 GDN 投影 10.36、16 层 attention 投影 3.13、embedding 2.37 + lm_head 2.37`
    + `(<code>tie_word_embeddings = false</code>,两张 248,320 × 5120 的表,合计 9.2%)、MTP 0.79、vision 塔 0.86。`,

    `<b>「非 expert 权重 → FP8」那一档在本模型上是实测,不是假设。</b>官方 <code>Qwen/Qwen3.8-27B-FP8</code> 的 1,606 个张量同样逐个读了:`
    + `407 个 <code>F8_E4M3</code> 权重共 <code>24,699,207,680</code> 参数,每个配一张 <b>BF16</b> 的 <code>weight_scale_inv</code>`
    + `(<code>weight_block_size [128,128]</code>,GLM 的 scale 是 F32),故量化部分每参数 <code>1 + 2/16384 = 1.0001220703125</code> B,精确;`
    + `其余 792 个张量 <code>3,082,220,272</code> 参数保留 BF16。合计 <code>30,866,663,264</code> B = <b>28.75 GiB</b>,`
    + `与 HF API 对该 repo 给出的 <code>BF16 3,082,220,272 + F8_E4M3 24,699,207,680 = 27,781,427,952</code>(与本 checkpoint 同一总数)<b>精确相等</b>。<br>`
    + `<b>不是「一半」</b>:embed / lm_head / vision / norms / GDN 的几张小张量共 6.16 GiB 官方也没量化,所以 FP8 只把 51.75 压到 28.75(−44%)。`
    + `这也是数据文件里 <code>nonExpertBf16Params / nonExpertFixedBytes</code> 的切分线 —— 在本模型上它们的含义是「官方 FP8 量化了哪些 / 保留了哪些」,`
    + `不是 K3 / GLM 那种「BF16 vs F32 / 原生 FP8」。roofline 的算力行把保留 BF16 的那 3.08B 参数按 BF16 算,不按 FP8。`,

    `<b>KV cache 是四个模型里最大的:65,536 B/token(BF16)。</b>16 层 GQA × 2 (K+V) × 4 kv head × head_dim 256 = `
    + `<b>32,768 元素/token</b> → BF16 64 KiB、FP8 32 KiB。对比 K3 27,648 B、GLM 11,616 B、DSv4.1 890 B —— `
    + `是 K3 的 2.4 倍、DSv4.1 的 74 倍。head_dim 256 是主因(常见的 128 只要一半);3:1 的混合层排布是把它压在 16 层上的原因,`
    + `64 层全 GQA 会是 256 KiB/token。<br>`
    + `<b>复制因子是 <code>max(1, TP ÷ 4)</code>,不是 TP</b>:KV 按 kv head 切,只有 4 个 head 可分。TP ≤ 4 时每卡只存自己那几个 head、TP 组合起来恰好一份;`
    + `TP=8 每卡至少拿 1 个 head,于是复制 2 份。这就是 p5en 上 TP4×DP2 同时赢过 TP8/DP1 与 TP1×DP8 的机制(128K/0.90 下 100 vs 53 vs 56 路)。`
    + `MLA 系那句「latent 每卡一份完整副本」在这里<b>不成立</b>,见 ADR-0012。<br>`
    + `<b>MTP 层的 KV 未计</b>:开投机解码时引擎会给 draft 层(也是一层 GQA)再分配一份 cache,每 token 多 1/16。`,

    `<b>Gated DeltaNet 的 recurrent state = 144.0 MiB/请求,与上下文无关,而它的 dtype 有两个「官方」答案。</b>`
    + `48 层 × 48 v-head × 128 × 128 × 4 B = <code>150,994,944</code> B。<code>config.json</code> 写 <code>mamba_ssm_dtype: "float32"</code>,`
    + `HF 参考实现按此分配,本页按 FP32 计。<b>但 vLLM 不是</b>:<code>--mamba-ssm-cache-dtype</code> 默认 <code>auto</code>,`
    + `对 Gated DeltaNet 这条路(<code>MambaStateDtypeCalculator._mamba_state_dtype</code>)auto → conv state dtype → <b>模型 dtype BF16</b>;`
    + `KDA 那条路(K3 / GLM)auto 反而是 float32。所以在 vLLM 上不传参就是 <b>72 MiB/请求</b>,要显式传 <code>float32</code> 才与 config 一致。<br>`
    + `与 KV 相等的临界上下文 = <code>150,994,944 ÷ 65,536</code> = <b>2,304 tokens</b>(TP=1);TP=8 时 state ÷8 而 KV 只 ÷4,临界降到 1,152。`
    + `<b>1K 下 state 占单请求的 69%</b>(TP ≤ 4;TP=8 时 state ÷8 而 KV 只 ÷4,降到 53%),所以那一档 dtype 在短上下文下能改并发约 1.5 倍,128K 下不到 1%。`
    + `vLLM recipe 里有一个实测印证:1× RTX 5090 @32K,KV 切 FP8 只把池子从 76,458 涨到 91,022 tokens(1.19×,不是 2×)—— `
    + `因为 state 不受 <code>--kv-cache-dtype</code> 影响,却与 KV 共用同一个池。`,

    `<b>MTP 层与 vision 塔都按「加载」算</b>,各 0.79 / 0.86 GiB(合计 3.2%)。<code>mtp_num_hidden_layers = 1</code>:`
    + `一层完整的 GQA + MLP 外加 <code>fc [5120, 10240]</code>,不开投机解码时引擎不加载;vision 塔 27 层 ViT + merger,`
    + `纯文本服务可用 <code>--language-model-only</code> 不加载 —— vLLM recipe 里这个开关让 1× 5090 @32K 的 KV 池从 91,022 涨到 135,926 tokens,`
    + `多出的 44,904 tokens × 16 KiB(FP8)≈ 0.7 GiB,与 vision 塔的 0.86 GiB 同量级,算是这一项的间接实测。`,

    `<b>TP 只能取 1 / 2 / 4 / 8。</b>24 个 q head、4 个 kv head、GDN 的 16 个 k head / 48 个 v head 都要被 TP 整除`
    + `(kv head 允许 TP 是它的倍数,那就是复制)。TP=16(两台)被 <code>24 % 16 ≠ 0</code> 挡住,vLLM / SGLang 会拒绝启动 —— `
    + `本页选到这类切分会出红字,数字按机械除法算出,不代表可部署。所以这个模型<b>跨机器只能靠 DP 或 PP</b>,而一张卡就装得下它,跨机器本来也没必要。`,

    `<b>每卡 12 GiB overhead 那个猜测在四个模型里对本模型影响最大,而它在本模型上的方向未知。</b>TP=8 时每卡权重只有 6.47 GiB,`
    + `overhead 是权重的 <b>1.9 倍</b>。2026-09-16 在 p5en 上对 GLM-5.3-Flash 起了一次服务,引擎自报的非权重非 KV 开销是 `
    + `<b>6.98 GiB</b>(non-torch 1.74 + 激活峰值 4.04 + CUDA graph 1.20),<b>在那个配置上</b>本页高了约 72%。`
    + `<b>但这个差值搬不到本模型头上</b>:同样把 12 换成 6.98,GLM 在 p5en/TP8 上是 53 → 56 路(5%),`
    + `而本模型在 g7e/TP1 上是 16 → 24 路(<b>50%</b>)—— 幅度差一个量级,正因为小模型上 overhead `
    + `占固定开销的比重大得多。<b>方向也不保证</b>:TP=1 没有跨卡通信 buffer,可是激活不分片又可能更大,`
    + `所以本模型的真实开销未必低于 12 GiB。它<b>能翻转小显存机型的装不装得下</b>:原候选里的 g6e(44.7 GiB)`
    + `就是被这 5 GiB 单独决定的,已因此撤出候选,见 <code>candidateInstances</code> 的注释。<br>`
    + `vLLM recipe 还记了一件与本页口径相反的事:CUDA graph capture 的分配<b>在 util 预算之外</b>(1× 5090 上「0.80 和 0.93 都只剩 47 MiB」,`
    + `要 <code>--enforce-eager</code> 才起得来),而本页把 12 GiB 整个放在预算之内。两种口径都不算错,但反解 overhead 时要先对齐这一点。`
    + `recipe 里那组引擎自报数(2× 5090 TP2、FP8 权重 14.28 GiB/卡、FP8 KV、262K:KV 池 377,456 tokens)是本项目见到的第一组`
    + `<b>引擎自报</b>读数,但按 <code>docs/glossary.md</code> 的 provenance 定义它<b>不计入 <code>measured</code></b> —— 那一级要求`
    + `「目标硬件实测」,而 5090 既不在候选机型里、这组数也不是本项目跑出来的。<b>唯一进入 <code>measured</code> 的是`
    + `<code>docs/measurements.md</code> M-001</b>(p5en 上的 GLM-5.3-Flash)。`
    + `加上 vLLM 混合池(attention KV 与 GDN state 共享、按页对齐)的分配规则本页不建模,这组 5090 数字也不能直接拿来反解。`,

    `<b>roofline 的算力行在这个模型上比在 MoE 模型上更松。</b>两件事:(1) attention 本身的 FLOPs 未计,而 128K 下它是 `
    + `<code>4 × 131,072 × 24 head × 256 × 16 层 ≈ 51.5 GFLOP/token</code>,与权重矩阵乘的 <code>2 × 27.78B = 55.6 GFLOP</code> <b>同量级</b> —— `
    + `dense 27B 的权重项本来就小,漏掉的这一项相对更大;(2) embedding(1.27B,查表)与 vision 塔(0.46B,decode 不跑)共 6.2% 的参数不做矩阵乘,`
    + `算力行按参数量算把它们也算进去了。两项方向相反,都不改变「HBM 带宽受限」的 regime:GDN state 每步读 + 写各 144 MiB/请求 ÷ TP,`
    + `这个随并发线性涨的斜率比算力项大,与 GLM / K3 一样到 4,096 路也到不了拐点。`,
  ],
  notesEn: [
    `<b>All weights are measured values, reconciled three ways, and this is the cleanest set so far.</b> We read the safetensors header of each of the 18 shards with Range requests `
    + `and summed the <code>dtype / shape / data_offsets</code> of <b>1,199</b> tensors: total bytes <code>55,562,855,904</code>, `
    + `<b>exactly equal</b> to <code>metadata.total_size</code> in <code>index.json</code> = 51.75 GiB; total params `
    + `<code>27,781,427,952</code>, <b>exactly equal</b> to <code>safetensors.parameters.BF16</code> from HF <code>/api/models/Qwen/Qwen3.8-27B</code>; `
    + `<b>all 1,199 tensors are BF16, not one F32</b> (the norms in K3 / GLM are F32).<br>`
    + `Breakdown: MLP of 64 layers 31.88 GiB (61.6%), GDN projections of 48 layers 10.36, attention projections of 16 layers 3.13, embedding 2.37 + lm_head 2.37 `
    + `(<code>tie_word_embeddings = false</code>, two 248,320 × 5120 tables, 9.2% together), MTP 0.79, vision tower 0.86.`,

    `<b>The "non-expert weights → FP8" option is measured on this model, not assumed.</b> We also read all 1,606 tensors of the official <code>Qwen/Qwen3.8-27B-FP8</code> tensor by tensor: `
    + `407 <code>F8_E4M3</code> weights with <code>24,699,207,680</code> params in total, each with a <b>BF16</b> <code>weight_scale_inv</code> `
    + `(<code>weight_block_size [128,128]</code>; the scale in GLM is F32), so the quantized part costs <code>1 + 2/16384 = 1.0001220703125</code> B per param, exactly; `
    + `the other 792 tensors, <code>3,082,220,272</code> params, stay BF16. Total <code>30,866,663,264</code> B = <b>28.75 GiB</b>, `
    + `which is <b>exactly equal</b> to the HF API figure for that repo, <code>BF16 3,082,220,272 + F8_E4M3 24,699,207,680 = 27,781,427,952</code> (the same total as this checkpoint).<br>`
    + `<b>It is not "half"</b>: embed / lm_head / vision / norms and a few small GDN tensors, 6.16 GiB in all, are not quantized in the official checkpoint either, so FP8 only reduces 51.75 to 28.75 (−44%). `
    + `This is also the split line of <code>nonExpertBf16Params / nonExpertFixedBytes</code> in the data file — on this model they mean "what the official FP8 quantized / what it kept", `
    + `not the "BF16 vs F32 / native FP8" split of K3 / GLM. The roofline compute line counts those 3.08B params that stay BF16 as BF16, not FP8.`,

    `<b>The KV cache is the largest of the four models: 65,536 B/token (BF16).</b> 16 GQA layers × 2 (K+V) × 4 kv heads × head_dim 256 = `
    + `<b>32,768 elements/token</b> → BF16 64 KiB, FP8 32 KiB. Compare K3 27,648 B, GLM 11,616 B, DSv4.1 890 B — `
    + `2.4 times K3 and 74 times DSv4.1. head_dim 256 is the main cause (the common 128 would give half); the 3:1 hybrid layout is what holds it to 16 layers, `
    + `and 64 full-GQA layers would be 256 KiB/token.<br>`
    + `<b>The replication factor is <code>max(1, TP ÷ 4)</code>, not TP</b>: KV is sharded by kv head and only 4 heads can be split. At TP ≤ 4 each GPU stores only its own heads and the TP group together holds exactly one copy; `
    + `at TP=8 each GPU needs at least 1 head, so it is replicated 2 times. This is the mechanism by which TP4×DP2 beats both TP8/DP1 and TP1×DP8 on p5en (100 vs 53 vs 56 concurrent requests at 128K/0.90). `
    + `The MLA-family statement "each GPU holds a full copy of the latent" does <b>not</b> hold here; see ADR-0012.<br>`
    + `<b>The KV of the MTP layer is not counted</b>: with speculative decoding on, the engine allocates another cache for the draft layer (also one GQA layer), adding 1/16 per token.`,

    `<b>The Gated DeltaNet recurrent state = 144.0 MiB/request, independent of context, and its dtype has two "official" answers.</b> `
    + `48 layers × 48 v-heads × 128 × 128 × 4 B = <code>150,994,944</code> B. <code>config.json</code> says <code>mamba_ssm_dtype: "float32"</code>, `
    + `the HF reference implementation allocates accordingly, and this page counts FP32. <b>But vLLM does not</b>: <code>--mamba-ssm-cache-dtype</code> defaults to <code>auto</code>, `
    + `and for the Gated DeltaNet path (<code>MambaStateDtypeCalculator._mamba_state_dtype</code>) auto → conv state dtype → <b>model dtype BF16</b>; `
    + `for the KDA path (K3 / GLM) auto is float32 instead. So on vLLM with no flag it is <b>72 MiB/request</b>, and you must pass <code>float32</code> explicitly to match the config.<br>`
    + `Break-even context where state equals KV = <code>150,994,944 ÷ 65,536</code> = <b>2,304 tokens</b> (TP=1); at TP=8 state is ÷8 while KV is only ÷4, so the break-even drops to 1,152. `
    + `<b>At 1K the state is 69% of a single request</b> (TP ≤ 4; at TP=8 state is ÷8 while KV is only ÷4, which lowers it to 53%), so at short context that dtype choice changes concurrency by about 1.5 times, and by less than 1% at 128K. `
    + `The vLLM recipe has a measured confirmation: on 1× RTX 5090 @32K, switching KV to FP8 only grows the pool from 76,458 to 91,022 tokens (1.19×, not 2×) — `
    + `because state is not affected by <code>--kv-cache-dtype</code> but shares the same pool with KV.`,

    `<b>The MTP layer and the vision tower are both counted as "loaded"</b>, 0.79 / 0.86 GiB (3.2% together). <code>mtp_num_hidden_layers = 1</code>: `
    + `one full GQA layer + MLP plus <code>fc [5120, 10240]</code>, which the engine does not load when speculative decoding is off; the vision tower is a 27-layer ViT + merger, `
    + `which text-only serving can skip with <code>--language-model-only</code> — in the vLLM recipe this switch grows the KV pool on 1× 5090 @32K from 91,022 to 135,926 tokens, `
    + `and the extra 44,904 tokens × 16 KiB (FP8) ≈ 0.7 GiB is the same order as the 0.86 GiB of the vision tower, an indirect measurement of this item.`,

    `<b>TP can only be 1 / 2 / 4 / 8.</b> The 24 q heads, 4 kv heads, and the 16 k heads / 48 v heads of GDN must all be divisible by TP `
    + `(a TP that is a multiple of the kv head count is allowed, and then the kv heads are replicated). TP=16 (two nodes) is blocked by <code>24 % 16 ≠ 0</code>, and vLLM / SGLang refuse to start — `
    + `selecting such a split on this page shows red text, and the numbers come from mechanical division and do not mean the split is deployable. So this model can <b>span nodes only through DP or PP</b>, and since one GPU holds it, spanning nodes is not needed anyway.`,

    `<b>The 12 GiB per-GPU overhead guess affects this model the most of the four models, and its direction on this model is unknown.</b> At TP=8 the weights are only 6.47 GiB per GPU, `
    + `so overhead is <b>1.9 times</b> the weights. On 2026-09-16 we started a service for GLM-5.3-Flash on p5en, and the non-weight, non-KV overhead reported by the engine was `
    + `<b>6.98 GiB</b> (non-torch 1.74 + activation peak 4.04 + CUDA graph 1.20), so <b>in that configuration</b> this page was about 72% too high. `
    + `<b>But that difference does not carry over to this model</b>: replacing 12 with 6.98 moves GLM on p5en/TP8 from 53 to 56 concurrent requests (5%), `
    + `while this model on g7e/TP1 goes from 16 to 24 (<b>50%</b>) — an order of magnitude apart, because on a small model overhead `
    + `is a much larger share of the fixed cost. <b>The direction is not guaranteed either</b>: TP=1 has no cross-GPU communication buffer, but activations are not sharded and may be larger, `
    + `so the real overhead of this model is not necessarily below 12 GiB. It <b>can flip whether a small-memory instance type fits</b>: g6e (44.7 GiB), originally a candidate, `
    + `was decided by these 5 GiB alone and was removed from the candidates for that reason; see the comment on <code>candidateInstances</code>.<br>`
    + `The vLLM recipe also records something that is the opposite of the assumption on this page: the CUDA graph capture allocation is <b>outside the util budget</b> (on 1× 5090 "0.80 and 0.93 both leave only 47 MiB", `
    + `and it only starts with <code>--enforce-eager</code>), while this page puts the whole 12 GiB inside the budget. Neither assumption is wrong, but align on this before back-solving overhead. `
    + `The engine-reported numbers in the recipe (2× 5090 TP2, FP8 weights 14.28 GiB/GPU, FP8 KV, 262K: KV pool 377,456 tokens) are the first set of `
    + `<b>engine-reported</b> readings this project has seen, but by the provenance definition in <code>docs/glossary.md</code> they are <b>not counted as <code>measured</code></b> — that level requires `
    + `"measured on the target hardware", and the 5090 is not among the candidate instance types and the numbers were not produced by this project. <b>The only entry in <code>measured</code> is `
    + `<code>docs/measurements.md</code> M-001</b> (GLM-5.3-Flash on p5en). `
    + `In addition, the allocation rules of the vLLM hybrid pool (attention KV and GDN state share it, page-aligned) are not modeled on this page, so these 5090 numbers cannot be used directly for back-solving either.`,

    `<b>The roofline compute line is looser on this model than on MoE models.</b> Two things: (1) the FLOPs of attention itself are not counted, and at 128K they are `
    + `<code>4 × 131,072 × 24 heads × 256 × 16 layers ≈ 51.5 GFLOP/token</code>, the <b>same order</b> as the weight matmul <code>2 × 27.78B = 55.6 GFLOP</code> — `
    + `the weight term of a dense 27B model is small to begin with, so the omitted term is relatively larger; (2) embedding (1.27B, table lookup) and the vision tower (0.46B, not run in decode), 6.2% of the params together, do no matmul, `
    + `and the compute line counts them anyway because it works from param count. The two effects have opposite directions, and neither changes the "HBM-bandwidth-bound" regime: the GDN state is read and written 144 MiB/request each per step ÷ TP, `
    + `and this slope, which grows linearly with concurrency, is larger than the compute term, so as with GLM / K3 the knee is not reached even at 4,096 concurrent requests.`,
  ],
};
