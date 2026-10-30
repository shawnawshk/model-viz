// GLM-5.3 —— zai-org/GLM-5.3
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数              derived  —— config.json(没有 text_config,纯文本模型)
//   权重字节数/参数量      derived  —— 逐张量读 141 个 shard 的 safetensors header 求和,
//                                    与 index.json 的 total_size、HF API 的 parameters 三方精确对账
//   indexer key cache     estimated —— 布局取自 vLLM 源码(每个 full 层 128 B FP8 + 4 B fp32 scale),
//                                    是引擎实现,不在 config / checkpoint 里;哪些层存 key 则是三方一致的一手依据,见 notes 第 3 条
//   KV cache dtype        不属于模型 —— 引擎运行时选项,已提为界面输入,见 kvDtypes 与 ADR-0007
//
// 与 GLM-5.3-Flash 是两种结构:Flash 是 34 KDA 线性 + 11 DSA 的混合,这里 78 层全是 DSA,没有线性层。
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["glm-5.3"] = {
  name: "GLM-5.3",
  hf: "zai-org/GLM-5.3",
  vllmRecipe: "https://recipes.vllm.ai/zai-org/GLM-5.3",
  // 官方模型卡 pipeline_tag = text-generation;config.json 里没有 vision_config。
  modalities: {
    input: ["text"], output: ["text"],
    sources: ["https://huggingface.co/zai-org/GLM-5.3/blob/main/README.md"],
  },
  blurb: "753.3B 总参数(含 MTP 层)· 39B 激活 · 78 层全 DSA 稀疏注意力(21 层带 indexer,57 层复用 top-k)"
       + " · 256 routed experts / top-8 · 原生 FP8 · 纯文本",
  blurbEn: "753.3B total parameters (including the MTP layer) · 39B activated · 78 layers, all DSA sparse attention (21 layers with an indexer, 57 reusing the top-k)"
       + " · 256 routed experts / top-8 · native FP8 · text only",
  hidden: 6144,                              // hidden_size,只用于 PP 的 send/recv 激活量

  // ---- 层结构 ----
  // 78 层全是 DSA(MLA latent 512 + RoPE 64 = 576 元素/token,与 DeepSeek 系相同,与 GLM-5.3-Flash 的 512/无 RoPE 不同)。
  // IndexCache:config 的 indexer_types 有 21 个 "full"、57 个 "shared"。shared 层复用上一个 full 层选出的 top-k,
  // 自己不建 indexer、不存 indexer key。full 层是 0,1,2,6,10,…,74 —— 与 vLLM 按 index_topk_freq = 4、
  // index_skip_topk_offset = 3 算出来的那组、以及 checkpoint 里带 indexer 权重的层(另加 MTP 层 78)逐层相同。
  layers: [
    { count: 21, family: "dsa", kvLoraRank: 512, qkRopeDim: 64,
      indexerElems: 128, indexerBytes: 1, indexerScaleBytes: 4,   // vLLM:head_dim 128 + 128 ÷ 128 × 4 = 132 B
      indexTopk: 2048,
      note: "indexer_types = full:自己算 top-k,存自己的 indexer key。key 的布局按 vLLM 的 DeepseekV32IndexerCache "
          + "(128 维 FP8 + 每 128 元素一个 fp32 scale),不受 --kv-cache-dtype 影响" },
    { count: 57, family: "dsa", kvLoraRank: 512, qkRopeDim: 64,
      indexerElems: 0, indexerBytes: 0,
      indexTopk: 2048,
      note: "indexer_types = shared(IndexCache):复用上一个 full 层的 top-k,不存 indexer key。latent 照存,decode 照读 top-k 个" },
  ],
  totalLayers: 78,

  // 边界声明里 indexer key cache 那一条的后半句(前半句由引擎拼)
  indexerAsm: `布局取自 vLLM 源码(每个 full 层 128 B FP8 + 4 B scale,57 个 shared 层不存)—— 是引擎实现,不在 checkpoint 里,`
    + `别的引擎可能不同;没有 IndexCache 的话是 78 层 × 132 = 10,296 B,KV 多约 8%`,
  indexerAsmEn: `layout taken from vLLM source (128 B FP8 + 4 B scale per full layer; the 57 shared layers store none) — it is an engine implementation detail, not in the checkpoint, `
    + `so other engines may differ; without IndexCache it would be 78 layers × 132 = 10,296 B, about 8% more KV`,

  // ---- KV cache 的 dtype:不是模型属性,是引擎的运行时选项(--kv-cache-dtype)----
  // 理由同 ADR-0007。第一项即默认值(auto = 模型 dtype = BF16)。
  // vLLM recipe 的所有 NVIDIA 命令都显式传了 --kv-cache-dtype fp8 / fp8_e4m3 —— 那是部署选择,不是默认值。
  kvDtypes: [
    { id: "bf16", label: "BF16(引擎默认)",   bytes: 2, labelEn: "BF16 (engine default)" },
    { id: "fp8",  label: "FP8(需显式开启)", bytes: 1, labelEn: "FP8 (opt-in)" },
  ],

  moe: {
    layers: 75,            // 78 层里的稀疏 MLP 层数(first_k_dense_replace = 3)。
                           // 带 expert 权重的层是 76 —— MTP 那层也有一整套 256 个专家,见 notes 第 4 条。
    experts: 256, topk: 8, sharedExperts: 1,
    latent: 6144, inter: 2048,   // expert 直接吃 hidden_size
  },
  expertLayerDesc: "76 层(75 层主干 + 1 层 MTP)",
  expertLayerDescEn: "76 layers (75 backbone + 1 MTP)",

  // ---- 权重:两桶实测字节 + 两个参数量(口径见 ADR-0008)----
  weights: {
    totalBytes:  755_617_140_416,      // 118,629 个张量逐个求和,与 index.json metadata.total_size 精确相等 = 703.72 GiB
    expertBytes: 734_618_714_112,      // 所有 .mlp.experts.* 张量 = 684.17 GiB(FP8 734,439,407,616 + F32 scale 179,306,496)
    expertParams:    734_439_407_616,  // 76 × 256 × 3 × 6144 × 2048,与逐张量 numel 精确相等
    nonExpertParams:  18_890_532_864,  // BF16 2,103,729,152 + 原生 FP8 16,786,784,256 + F32 19,456(不含 *_scale_inv)
    nonExpertBf16Params: 2_103_729_152,   // embedding / lm_head / router / MTP 的 eh_proj / indexer.weights_proj / norms
    nonExpertFixedBytes: 16_790_968_000,  // 原生 FP8 16,786,784,256 + F32 scale 4,105,920 + F32 e_score_correction_bias 77,824
    // 三条闭合校验,前两条引擎加载时会硬断言:
    //   2,103,729,152 × 2 + 16,790,968,000 = 20,998,426,304 B = 19.56 GiB = totalBytes − expertBytes ✓
    //   76 × 256 × 3 × 6144 × 2048 = expertParams(整数层数)✓
    //   expertParams + nonExpertParams = 753,329,940,480,与 HF API 的 safetensors.parameters.total 精确相等 ✓
  },

  // ---- 量化 ----
  // 原生 FP8 e4m3,weight_block_size [128, 128],scale 是 F32 → 实测每参数 1.000244140625 B,与 GLM-5.3-Flash 逐位相同。
  quantFormat: "fp8",
  // 与 config.json 的 quantization_config.modules_to_not_convert(541 条)逐类对应:
  quantExcludes: "全部 norms / router(mlp.gate 与 e_score_correction_bias)/ indexer 的 weights_proj 与 k_norm / MTP 的 eh_proj / embedding / lm_head",

  // expert 权重可选格式。BF16 一档有官方 checkpoint(zai-org/GLM-5.3-BF16),同样逐张量读了 282 个 shard 的 header:
  // 59,585 个张量合计 1,506,659,919,872 B,与它的 index.json 精确相等;expert 1,468,878,815,232 B = 734,439,407,616 × 2,
  // 一个不差(没有 scale 张量)。1,368 GiB,一台机器装不下,recipe 也写明 BF16 要多机。
  expertFormats: [
    { id: "fp8",  label: "FP8(原生)", bytesPerParam: null, labelEn: "FP8 (native)" },
    { id: "bf16", label: "BF16(官方 BF16 checkpoint)", bytesPerParam: 2, measured: true, labelEn: "BF16 (official BF16 checkpoint)" },
  ],
  // 非 expert 是混合 dtype,而且这次大头已经是 FP8(attention 全部投影、shared expert、前 3 层 dense MLP),
  // 仍是 BF16 的只有 2.10B 参数,其中 1.90B 是 embedding 与 lm_head。这个开关最多省 1.96 GiB。
  nonExpertFormats: [
    { id: "native", label: "原生混合(实测)",     bf16Bytes: 2, labelEn: "Native mixed (measured)" },
    { id: "fp8",    label: "BF16 部分也压成 FP8", bf16Bytes: 1, labelEn: "BF16 part also quantized to FP8" },
  ],

  // ---- 候选实例:业务判断,与用户逐个确认(ADR-0005 §2)。2026-10-30 确认:P 系这 4 个,与 K3 / GLM-Flash / DSv4.1 对齐。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // vLLM recipe 的标准配置是 8×H200 TP8(+ FP8 KV + MTP),但它在本页默认口径(128K / util 0.90 / BF16 KV)下只有 2 路,
  // 比并发下限 8 还低,页面一打开就是红字。所以起始切分取两台的 TP8/DP2 —— 同一个「每台 TP8」,多一台换一倍 DP。
  // recipe 那一档留在第二个预设里。
  defaultParallel: { n: 2, tp: 8, dp: 2, pp: 1, ep: 16 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  // 名字里「不要」写绝对路数,也「不要」写「推荐 / 最优」(ADR-0008 §9 / ADR-0005),路数由 markPreset() 现算。
  // 手算(BF16 KV、原生 FP8、每卡 12 GiB overhead;路数,依次为下面六个预设):
  //     1K/0.90    1576   304  9520 11288   888 11288
  //   128K/0.90      12     2    64    88     0    88
  //     1M/0.90       0     0     0     8     0     8
  // 这个模型的切分故事是 GLM-5.3-Flash 的「反面 + 回归」:
  //   一台 p5en 上,expert bank 684 GiB ÷ 8 = 85.5 GiB/卡 先吃掉预算的 67%,TP1×DP8 再复制 19.6 GiB 非 expert,
  //   剩 9.8 GiB 连一路 128K 的 KV(11.3 GiB)都放不下 —— Flash 页上最好的那一档,在这里一台上是红字;
  //   两台把 expert bank 摊到 16 卡,KV 复制的代价重新占主导,TP1×DP16 回到 TP8/DP2 的 5.3 倍,PP2 又比它高 38%。
  // 括号里标该切分下 KV latent 被复制几份 —— 这是差异的机制,恒真。
  presets: [
    { name: "两台 TP8/DP2(KV ×8)", nameEn: "Two nodes TP8/DP2 (KV ×8)",              inst: "p5en.48xlarge",    n: 2, tp: 8, dp: 2,  pp: 1, ep: 16 },
    { name: "一台 TP8/DP1(官方 recipe,KV ×8)", nameEn: "One node TP8/DP1 (official recipe, KV ×8)",  inst: "p5en.48xlarge",    n: 1, tp: 8, dp: 1,  pp: 1, ep: 8  },
    { name: "两台 TP1×DP16(KV ×1)", nameEn: "Two nodes TP1×DP16 (KV ×1)",             inst: "p5en.48xlarge",    n: 2, tp: 1, dp: 16, pp: 1, ep: 16 },
    // PP 在这个模型上同样最省显存:非 expert 只存 1 份,每个 stage 只持有自己那 39 层的 KV。代价是流水线气泡,这张图不算
    { name: "两台 PP2 TP1×DP8(KV ×1)", nameEn: "Two nodes PP2 TP1×DP8 (KV ×1)",          inst: "p5en.48xlarge",    n: 2, tp: 1, dp: 8,  pp: 2, ep: 8  },
    // 留着不是凑数:与 GLM-5.3-Flash 页的同名预设对照,看 expert bank 的体量怎样把最优方向翻过来
    { name: "一台 TP1×DP8(KV ×1)", nameEn: "One node TP1×DP8 (KV ×1)",              inst: "p5en.48xlarge",    n: 1, tp: 1, dp: 8,  pp: 1, ep: 8  },
    // 一台 b300 与两台 p5en 的 PP2 在上面三个口径下路数恰好相同 —— b300 的 241.7 GiB 预算约等于 p5en 的两倍减去一份 overhead
    { name: "b300 一台 TP1×DP8(KV ×1)", nameEn: "b300 one node TP1×DP8 (KV ×1)",         inst: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8,  pp: 1, ep: 8  },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  notes: [
    `<b>权重全部是实测值,三方对账。</b>对 141 个 shard 逐个发 Range 请求读 safetensors header,`
    + `把 <b>118,629</b> 个张量的 <code>dtype / shape / data_offsets</code> 求和:字节合计 <code>755,617,140,416</code>,`
    + `与 <code>index.json</code> 的 <code>metadata.total_size</code> <b>精确相等</b> = 703.72 GiB;参数合计(不含 <code>*_scale_inv</code>)`
    + `<code>753,329,940,480</code>,与 HF <code>/api/models/zai-org/GLM-5.3</code> 的 <code>safetensors.parameters</code> 三项之和`
    + `(BF16 2,103,729,152 + F8_E4M3 751,226,191,872 + F32 19,456)<b>精确相等</b>;expert 桶每参数 <code>1.000244140625</code> B,`
    + `恰好 = 1 B(FP8 e4m3)+ 4 B ÷ (128×128) 的块 scale,与 GLM-5.3-Flash 逐位相同。`,

    `<b>每 token 的 KV 是 GLM-5.3-Flash 的 8 倍:92,628 B(BF16)。</b>78 层全是 DSA,每层 <code>kv_lora_rank 512 + qk_rope 64 = 576</code> 元素 `
    + `→ <code>78 × 576 × 2 = 89,856 B</code>,再加 indexer key 2,772 B。Flash 只有 11 层持有 KV、而且不带 RoPE 维度(每层 512 元素),`
    + `所以只有 11,616 B。对比 K3 27,648 B、DSv4.1 890 B、Qwen3.8-27B 65,536 B —— 这是六个模型里<b>最大</b>的一份,`
    + `而且 latent 是所有 head 共享的,TP 组内每卡一份完整副本:TP8 时一路 128K 在每张卡上都要 <b>11.31 GiB</b>。`,

    `<b>IndexCache:78 层里只有 21 层存 indexer key。</b><code>indexer_types</code> 有 21 个 <code>full</code>、57 个 <code>shared</code>;`
    + `shared 层复用上一个 full 层选出的 top-2048,自己不建 indexer。full 层是 <code>0, 1, 2, 6, 10, …, 74</code> —— `
    + `这一组在三处<b>逐层相同</b>:config 的 <code>indexer_types</code>、vLLM 按 <code>index_topk_freq = 4</code> 与 `
    + `<code>index_skip_topk_offset = 3</code> 算出的 <code>skip_topk</code>、checkpoint 里带 <code>indexer.wk</code> 权重的层(另有 MTP 层 78)。<br>`
    + `每个 full 层的 key 按 vLLM 的 <code>DeepseekV32IndexerCache</code> 计:<code>head_dim 128 + 128 ÷ 128 × 4 = 132 B</code>`
    + `(FP8 值 + fp32 scale),与 <code>--kv-cache-dtype</code> 无关。<code>21 × 132 = 2,772 B/token</code>,约 latent 的 3%。`
    + `层数是一手依据,<b>每层的字节布局是 vLLM 的实现</b>,标记为 <span class="pv pv-e">估算</span>。`
    + `decode 时 shared 层不扫 indexer,只读 top-k 个 latent,所以 roofline 里那 57 层的读取与上下文长度无关。`,

    `<b>MTP 层算进显存了</b>。checkpoint 里有第 78 层(<code>num_nextn_predict_layers = 1</code>),带一整套完整的 256 个专家 —— `
    + `所以 expert 桶写的是「76 层」而不是 75 层。它的 expert 权重 <b>9.00 GiB</b> 计入 expert 桶,另有 0.34 GiB 的 `
    + `<code>eh_proj</code> / attention / shared expert / norms 计入非 expert 桶。vLLM recipe 的启动命令都开了 MTP(<code>num_speculative_tokens 5</code>),`
    + `所以这一层是会加载的;但 <b>MTP 层自己的 KV 未计</b>(它也是一层 DSA:576 × 2 + 132 = 1,284 B/token,约 +1.4%),all-to-all 次数也按主干 75 层算。`,

    `<b>非 expert 这次大头已经是 FP8,那个开关只动很小一块。</b>attention 的 <code>q_a / q_b / kv_a / kv_b / o_proj</code>、`
    + `75+1 层的 shared expert、前 3 层的 dense MLP、indexer 的 <code>wq_b / wk</code> 全是原生 FP8,共 16.79B 参数。`
    + `仍是 BF16 的只有 2.10B 参数:embedding 与 lm_head 各 0.95B(<code>tie_word_embeddings = false</code>)、`
    + `router 0.12B、MTP 的 <code>eh_proj</code> 0.08B、indexer 的 <code>weights_proj</code>、norms。`
    + `「BF16 部分也压成 FP8」最多省 1.96 GiB,再按 TP 切 —— 与 GLM-5.3-Flash 上省 6.45 GiB 不同,这里几乎不改变结论。`,

    `<b>与 vLLM recipe 的说法对得上,但这不是实测。</b>recipe 写的是:FP8 checkpoint「单台 8×H200 / 8×H20 就能服务」,`
    + `而「要跑满 1M 上下文」需要 8×B200 + FP8 KV。按本页口径(util 0.90、FP8 KV、每卡 12 GiB overhead):`
    + `一台 p5en TP8 在 1M 下 <b>0 路</b>、一台 p6-b200 TP8 在 1M 下 <b>1 路</b> —— 方向一致。`
    + `但 recipe 没给 NVIDIA 机型上引擎自报的 KV block 数,它给的两组(MI325X / MI350X)不在候选机型里,`
    + `所以按 <code>docs/glossary.md</code> 的 provenance 定义,这一条<b>不计入 measured</b>。`,

    `<b>每卡 12 GiB overhead 这个猜测在本模型上没有可借的实测。</b>唯一一档实测(<code>docs/measurements.md</code> M-001)是 `
    + `GLM-5.3-Flash / TP8 / p5en 上的 6.98 GiB,但那是另一种结构:45 层、hidden 4096、有 mHC 的 <code>hc_*</code> 残差流;`
    + `本模型 78 层、hidden 6144、没有 mHC,而且 recipe 开的 MTP 5 个 draft token 还要额外的激活与 draft 状态。`
    + `<b>不能拿 Flash 的 6.98 来修正本页。</b>一台 p5en TP8 只剩 26.9 GiB 给 KV,而一路 128K 要 11.3 GiB —— overhead 只要比猜的多 4.3 GiB,就从 2 路掉到 1 路。`,
  ],
  notesEn: [
    `<b>All weights are measured values, reconciled three ways.</b> The safetensors header of each of the 141 shards was read with a Range request, `
    + `and the <code>dtype / shape / data_offsets</code> of <b>118,629</b> tensors were summed: total bytes <code>755,617,140,416</code>, `
    + `<b>exactly equal</b> to <code>metadata.total_size</code> in <code>index.json</code> = 703.72 GiB; total parameters (excluding <code>*_scale_inv</code>) `
    + `<code>753,329,940,480</code>, <b>exactly equal</b> to the sum of the three entries of <code>safetensors.parameters</code> at HF <code>/api/models/zai-org/GLM-5.3</code> `
    + `(BF16 2,103,729,152 + F8_E4M3 751,226,191,872 + F32 19,456); the expert bucket is <code>1.000244140625</code> B per parameter, `
    + `exactly = 1 B (FP8 e4m3) + 4 B ÷ (128×128) block scale, bit-for-bit identical to GLM-5.3-Flash.`,

    `<b>KV per token is 8 times that of GLM-5.3-Flash: 92,628 B (BF16).</b> All 78 layers are DSA, each with <code>kv_lora_rank 512 + qk_rope 64 = 576</code> elements `
    + `→ <code>78 × 576 × 2 = 89,856 B</code>, plus 2,772 B of indexer key. Flash has KV in only 11 layers and no RoPE dimension (512 elements per layer), `
    + `so it is only 11,616 B. Compare K3 at 27,648 B, DSv4.1 at 890 B, Qwen3.8-27B at 65,536 B — this is the <b>largest</b> of the six models, `
    + `and the latent is shared by all heads, so each GPU in a TP group holds a full copy: at TP8 one 128K request needs <b>11.31 GiB</b> on every GPU.`,

    `<b>IndexCache: only 21 of the 78 layers store an indexer key.</b> <code>indexer_types</code> has 21 <code>full</code> and 57 <code>shared</code> entries; `
    + `a shared layer reuses the top-2048 chosen by the previous full layer and builds no indexer of its own. The full layers are <code>0, 1, 2, 6, 10, …, 74</code> — `
    + `this set is <b>identical layer by layer</b> in three places: the config's <code>indexer_types</code>, the <code>skip_topk</code> that vLLM computes from <code>index_topk_freq = 4</code> and `
    + `<code>index_skip_topk_offset = 3</code>, and the layers in the checkpoint that carry <code>indexer.wk</code> weights (plus MTP layer 78).<br>`
    + `The key of each full layer is computed per vLLM's <code>DeepseekV32IndexerCache</code>: <code>head_dim 128 + 128 ÷ 128 × 4 = 132 B</code> `
    + `(FP8 values + fp32 scale), independent of <code>--kv-cache-dtype</code>. <code>21 × 132 = 2,772 B/token</code>, about 3% of the latent. `
    + `The layer counts are a primary source, but <b>the per-layer byte layout is a vLLM implementation detail</b>, marked <span class="pv pv-e">estimated</span>. `
    + `During decode, shared layers do not scan the indexer and read only the top-k latents, so in the roofline the reads of those 57 layers do not depend on context length.`,

    `<b>The MTP layer is counted in GPU memory.</b> The checkpoint contains layer 78 (<code>num_nextn_predict_layers = 1</code>) with a full set of 256 experts — `
    + `so the expert bucket is written as "76 layers" rather than 75. Its expert weights, <b>9.00 GiB</b>, go into the expert bucket, and another 0.34 GiB of `
    + `<code>eh_proj</code> / attention / shared expert / norms goes into the non-expert bucket. The vLLM recipe start-up commands all enable MTP (<code>num_speculative_tokens 5</code>), `
    + `so this layer does get loaded; but <b>the KV of the MTP layer itself is not counted</b> (it is also a DSA layer: 576 × 2 + 132 = 1,284 B/token, about +1.4%), and the all-to-all count is also based on the 75 backbone layers.`,

    `<b>The bulk of the non-expert weights is already FP8 this time, so that switch moves only a small part.</b> The attention <code>q_a / q_b / kv_a / kv_b / o_proj</code>, `
    + `the shared experts of 75+1 layers, the dense MLP of the first 3 layers, and the indexer <code>wq_b / wk</code> are all native FP8, 16.79B parameters in total. `
    + `Only 2.10B parameters remain BF16: embedding and lm_head at 0.95B each (<code>tie_word_embeddings = false</code>), `
    + `router 0.12B, MTP <code>eh_proj</code> 0.08B, indexer <code>weights_proj</code>, and norms. `
    + `"BF16 part also quantized to FP8" saves at most 1.96 GiB, and that is then sharded by TP — unlike the 6.45 GiB saved on GLM-5.3-Flash, here it hardly changes the conclusion.`,

    `<b>This agrees with what the vLLM recipe says, but it is not a measurement.</b> The recipe states that the FP8 checkpoint "can be served on a single node of 8×H200 / 8×H20", `
    + `and that "running the full 1M context" needs 8×B200 + FP8 KV. Under this page's assumptions (util 0.90, FP8 KV, 12 GiB overhead per GPU): `
    + `one p5en node at TP8 gives <b>0 concurrent requests</b> at 1M, and one p6-b200 node at TP8 gives <b>1 concurrent request</b> at 1M — the direction is consistent. `
    + `But the recipe does not give the KV block count self-reported by the engine on NVIDIA instance types; the two it does give (MI325X / MI350X) are not among the candidate instance types, `
    + `so by the provenance definition in <code>docs/glossary.md</code>, this item is <b>not counted as measured</b>.`,

    `<b>There is no measurement to borrow for the 12 GiB per-GPU overhead guess on this model.</b> The only measurement (<code>docs/measurements.md</code> M-001) is `
    + `6.98 GiB on GLM-5.3-Flash / TP8 / p5en, but that is a different structure: 45 layers, hidden 4096, and an mHC residual stream with <code>hc_*</code>; `
    + `this model has 78 layers, hidden 6144, no mHC, and the MTP with 5 draft tokens that the recipe enables needs extra activations and draft state. `
    + `<b>Flash's 6.98 cannot be used to correct this page.</b> One p5en node at TP8 has only 26.9 GiB left for KV, while one 128K request needs 11.3 GiB — if the overhead is just 4.3 GiB more than guessed, concurrency drops from 2 to 1.`,
  ],
};
