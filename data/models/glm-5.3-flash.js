// GLM-5.3-Flash —— zai-org/GLM-5.3-Flash
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数            derived  —— config.json(text_config)
//   权重字节数/参数量    derived  —— 逐张量读 62 个 shard 的 safetensors header 求和,
//                                  与 index.json 的 total_size、HF API 的 parameters 三方对账
//   KDA state dtype     estimated —— 结构由 config 推出,但 dtype 是假设(按 FP32)
//   indexer key cache   estimated —— 元素数按 index_kpool=4 池化推算,dtype 按 FP8 假设;
//                                   两者都不在 config / checkpoint 里,见 notes 第 4 条
//   KV cache dtype      不属于模型 —— 引擎运行时选项,已提为界面输入,见 kvDtypes 与 ADR-0007
//
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["glm-5.3-flash"] = {
  name: "GLM-5.3-Flash",
  hf: "zai-org/GLM-5.3-Flash",
  blurb: "321.3B 总参数 · 18B 激活 · 45 层混合注意力(34 KDA 线性 + 11 DSA 稀疏)· 288 routed experts / top-8 · 原生 FP8",
  hidden: 4096,                              // hidden_size,只用于 PP 的 send/recv 激活量

  // ---- 层结构 ----
  // layer_types 逐层数出来的:linear_attention 34 层、deepseek_sparse_attention 11 层,
  // 排布是「3 个线性 + 1 个稀疏」循环(full_attn_layers = 3,7,11,…,43)。
  layers: [
    { count: 11, family: "dsa", kvLoraRank: 512, qkRopeDim: 0,
      indexerElems: 32, indexerBytes: 1,
      note: "DSA = MLA latent + lightning indexer。latent 被所有 head 共享 → TP 内每卡各存一份完整副本。"
          + "qk_rope_head_dim=0(mla_use_nope)→ 每 token 只有 512 个元素,不是 DeepSeek 系的 576。"
          + "indexer 的 key cache 另算,走自己的 dtype" },
    { count: 34, family: "linear", heads: 64, headDim: 128, stateDtypeBytes: 4,
      note: "KDA 线性注意力。recurrent state 与序列长度无关,per-head 可按 TP 切分,不复制。"
          + "另有 3 个 kernel=4 的短卷积 state,量级可忽略,未计入" },
  ],
  totalLayers: 45,

  // ---- KV cache 的 dtype:不是模型属性,是引擎的运行时选项(--kv-cache-dtype)----
  // 理由与 K3 完全相同,见 ADR-0007。第一项即默认值(auto = 模型 dtype = BF16)。
  // 注意这个开关不影响 indexer 的 key cache —— 那一份走自己的 dtype。
  kvDtypes: [
    { id: "bf16", label: "BF16(引擎默认)",   bytes: 2 },
    { id: "fp8",  label: "FP8(需显式开启)", bytes: 1 },
  ],

  moe: {
    layers: 42,            // 45 层里的稀疏 MLP 层数(first_k_dense_replace = 3)。
                           // 注意带 expert 权重的层是 43 —— MTP 那层也有一整套,见 notes 第 2 条。
    experts: 288, topk: 8, sharedExperts: 1,
    latent: 4096, inter: 2048,   // expert 直接吃 hidden_size,没有 K3 那样的 MoE latent 投影
  },

  // ---- 权重:两桶实测字节 + 两个参数量(口径见 ADR-0008)----
  weights: {
    totalBytes:  328_326_771_576,      // 逐张量求和,与 index.json metadata.total_size 精确相等
    expertBytes: 311_729_651_712,      // 所有 .mlp.experts.* 张量 = 290.32 GiB(含 MTP 层那 288 个)
    expertParams:    311_653_564_416,  // 43 × 288 × 3 × 4096 × 2048,与逐张量 numel 精确相等
    nonExpertParams:   9_669_466_974,  // 其余张量(不含 *_scale_inv)
    nonExpertBf16Params: 6_926_096_640,   // 其中可被「非 expert 权重」开关切成 FP8 的那部分
    nonExpertFixedBytes: 2_744_926_584,   // 原生 FP8 模块 2,743,074,816 B + F32 1,851,768 B,开关不动
    // 三条闭合校验,前两条引擎加载时会硬断言:
    //   6,926,096,640 × 2 + 2,744,926,584 = 16,597,119,864 B = 15.46 GiB = totalBytes − expertBytes ✓
    //   43 × 288 × 3 × 4096 × 2048 = expertParams(整数层数)✓
    //   expertParams + nonExpertParams = 321,323,031,390,与 HF API 的 safetensors.parameters 合计一致 ✓
  },

  // ---- 量化 ----
  // 原生格式是 FP8 e4m3(quantization_config.fmt = "e4m3",activation_scheme = "dynamic"),
  // 按 128×128 块一个 F32 scale,故实测每参数 1.000244140625 B。
  // 四个候选机型都有原生 FP8,所以不会触发 K3 那种「无原生 kernel」告警。
  quantFormat: "fp8",
  // 与 config.json 的 quantization_config.modules_to_not_convert 逐条对应:
  quantExcludes: "34 个 KDA 层的全部投影 / kv_b_proj / indexer / mHC 的 hc_* / lm_head / embedding / norms",

  // expert 权重可选格式。bytesPerParam=null 表示「用实测字节数」(此处为 1.000244140625)。
  // BF16 一档不是模型的原生形态,是把 expert 反量化物化后的推算值 —— 290 GiB 会涨到 580 GiB,
  // 一台机器立刻装不下,这就是它值得放在这里的原因。
  expertFormats: [
    { id: "fp8",  label: "FP8(原生)", bytesPerParam: null },
    { id: "bf16", label: "BF16",      bytesPerParam: 2 },
  ],
  // 非 expert 是混合 dtype(见 notes 第 5 条),所以第一档叫「原生混合」而不是「BF16」。
  // 第二档只把 BF16 的那 6.93B 参数压成 FP8,已经是 FP8 的部分不动。
  nonExpertFormats: [
    { id: "native", label: "原生混合(实测)",     bf16Bytes: 2 },
    { id: "fp8",    label: "BF16 部分也压成 FP8", bf16Bytes: 1 },
  ],

  // ---- 候选实例:业务判断,与用户逐个确认,不从 instance 目录自动推导(ADR-0005 §2)。
  // 2026-09-09 确认:与 K3 页对齐,只保留 P 系这 4 个,便于两个模型并排比较。
  // 305.78 GiB 权重 + 每卡 12 GiB overhead,这 4 个机型都能用「一台」装下 —— 这是本模型与 K3
  // 最大的差别。p5.48xlarge(H100 80GB)其实也跑得动(一台 72 GiB 预算下 42 路 @128K),
  // g7e 甚至因为最优解本来就是 TP=1 而不受「无 NVLink」拖累 —— 但这两个不在本次确认的范围内。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // 起始切分故意选「传统」的 TP8/DP1:让 KV 被复制 8 份的代价先摆出来,
  // 点一下旁边的 TP1×DP8 就能看到同一台机器上并发跳 6 倍(128K/util 0.90 下 53 → 325 路;
  // 这个倍数本身随口径变,见下面 presets 的注释)。
  defaultParallel: { n: 1, tp: 8, dp: 1, pp: 1, ep: 8 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  //
  // 名字里「不要」写绝对路数。路数取决于 util、上下文长度、KV dtype、两个权重格式,
  // 而点预设时这些口径故意不重置(你拖到 1M 就是想在 1M 下比切分)。写死在名字里必然与
  // 滑块脱钩 —— 按钮上标着 325 路(128K/0.90)而页面显示 44 路(1M/0.90)。第三行由 markPreset() 按当前
  // 口径现算并标注 @上下文/util;名字里要说就说「相对」关系,那个稳健得多。
  // 名字里也「不要」标「推荐 / 最优」。ADR-0005:这是校验器不是求解器,排序交给现算的数字。
  // 而且那个排序本身随口径变 —— 原生 FP8、四个角上实测(路数,依次为下面六个预设):
  //     1K/0.70   1751  1943  2203  2372  5904  6909   ← TP2×DP4 反超 TP1×DP8
  //     1K/1.00   3279  4295  4387  4724 10607 11389   ← 同上
  //   128K/0.90     53   325   191   365   838   918
  //     1M/1.00      7    53    29    59   132   142
  // TP1×DP8 对 TP8/DP1 的领先幅度从 6.1× 缩到 1.11×;短上下文下 KV 复制的代价变小,
  // 而 TP 能切非 expert 权重与 recurrent state,于是反过来占优。
  // 把 expert 切成 BF16(580 GiB)更会直接翻转:1M/0.70 下 TP1×DP8 装不下,TP8/DP1 还剩 1 路。
  // 所以名字只描述结构(括号里是该切分下 KV 被复制几份 —— 这是差异的机制,恒真)。
  presets: [
    // 与 K3 相反:这个模型的非 expert 权重只有 15.46 GiB(p5en 预算的 12%),复制它很便宜;
    // 而 KV latent 会被 TP 复制 TP 份。所以在长上下文下 TP 越大越亏。
    { name: "TP8/DP1(KV ×8)",       inst: "p5en.48xlarge",    n: 1, tp: 8, dp: 1,  pp: 1, ep: 8  },
    { name: "TP1×DP8(KV ×1)",       inst: "p5en.48xlarge",    n: 1, tp: 1, dp: 8,  pp: 1, ep: 8  },
    { name: "TP2×DP4(KV ×2)",       inst: "p5en.48xlarge",    n: 1, tp: 2, dp: 4,  pp: 1, ep: 8  },
    // PP 在这个模型上显存最省:非 expert 只存 1 份,且每个 stage 只持有自己那几层的 KV。
    // 四个角上它都是单台里最高的一档 —— 但代价是流水线气泡,而这张图不算那个。
    { name: "PP2 TP1×DP4(KV ×1)",   inst: "p5en.48xlarge",    n: 1, tp: 1, dp: 4,  pp: 2, ep: 4  },
    { name: "两台 TP1×DP16",         inst: "p5en.48xlarge",    n: 2, tp: 1, dp: 16, pp: 1, ep: 16 },
    // 相对比较比绝对路数稳健:b300 一台 > p5en 两台,在上面四个角上都成立(幅度 1.07–1.17×)。
    { name: "b300 一台 ⇄ p5en 两台", inst: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8,  pp: 1, ep: 8  },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  notes: [
    `<b>权重全部是实测值,三方对账</b>。做法是对 62 个 shard 逐个发 Range 请求读 safetensors header,`
    + `把 76,108 个张量的 <code>data_offsets</code> 与 <code>shape</code> 求和:字节合计 `
    + `<code>328,326,771,576</code>,与 <code>index.json</code> 的 <code>metadata.total_size</code> `
    + `<b>精确相等</b>;参数量合计 <code>321,323,031,390</code>,与 HF `
    + `<code>/api/models/zai-org/GLM-5.3-Flash</code> 的 <code>safetensors.parameters</code> 三项之和 `
    + `<b>精确相等</b>;expert 桶每参数 <code>1.000244140625</code> B,恰好 = 1 B(FP8 e4m3)+ `
    + `4 B ÷ (128×128) 的块 scale,与 <code>quantization_config</code> 声明的 e4m3 + 128 块量化`
    + `<b>精确吻合</b>。三条独立来源互相印证,所以这一桶不需要任何估算。`,

    `<b>MTP 层算进显存了</b>。checkpoint 里有第 45 层(<code>num_nextn_predict_layers = 1</code>),`
    + `带一整套完整的 288 个专家 —— 所以上面那条写的是「43 层 × 288 experts」而不是 42 层。`
    + `它的 expert 权重 <b>6.75 GiB</b> 计入 expert 桶,另有 0.23 GiB 的 eh_proj / norms / shared expert `
    + `计入非 expert 桶。<br><b>不开投机解码时引擎不会加载它</b>:那种情况下 expert 桶少 6.75 GiB`
    + `(2.3%),按 EP 切完对单卡的影响不到 1 GiB。all-to-all 次数按主干 42 层算,没把它算进去。`,

    `<b>DSA 是稀疏注意力,但它不省显存。</b><code>index_topk = 2048</code> 决定的是每个 query 去`
    + `<b>看</b>多少个 token,不是<b>存</b>多少个 —— 全部 token 的 MLA latent 依然要留在 cache 里,`
    + `显存开销与 full MLA 一模一样。DSA 省的是 attention 的算力与 KV 读带宽,而<b>这两样都不在这张图里</b>。`
    + `如果你是为了「稀疏注意力应该更省显存」来看这一页的,结论是:不省,一分都不省。`,

    `<b>indexer key cache 是本模型比 K3 多出来的一个估算项。</b>lightning indexer 的 <code>wk</code> 输出 128 维,`
    + `而 config 里 <code>index_kpool = 4</code> 且 <code>index_kpool_compress = true</code>,`
    + `本页按「4 个 token 池化成 1 个 key」计:<b>32 元素/token/层 × FP8 = 11 层合计 352 B/token</b>,`
    + `约 MLA latent 的 <b>+3%</b>。<br>若引擎实际不做池化压缩(128 元素/token/层),则是 `
    + `<b>1,408 B/token</b>,约 <b>+13%</b>。两个读数都不改变结论的量级,但<b>都不是一手依据</b> —— `
    + `pooling 方式与 dtype 在 <code>config.json</code> 和 checkpoint 里都查不到,`
    + `和 KV dtype 一样属于引擎实现细节。标记为 <span class="pv pv-e">估算</span>。`,

    `<b>非 expert 是混合 dtype,所以那个开关只动一半。</b>已经是 FP8 的:11 个 DSA 层的 `
    + `<code>q_a / q_b / kv_a / o_proj</code>、42+1 层的 shared expert、前 3 层的 dense MLP,`
    + `共 2,743,074,816 参数。仍是 BF16 的:34 个 KDA 层的全部投影、<code>kv_b_proj</code>、indexer、`
    + `mHC 的 <code>hc_*</code> 张量(共 0.07 GiB)、embedding 与 lm_head,共 6,926,096,640 参数。<br>`
    + `所以「非 expert 权重」切到 FP8 只把后者从 12.90 GiB 压到 6.45 GiB,前者不受影响 —— `
    + `一共省 6.45 GiB,再按 TP 切。<b>mHC 带来的额外残差流激活未单独计入</b>,归在那 12 GiB 猜测里。`,

    `<b>每 token 只有 512 个 KV 元素,不是 576。</b><code>mla_use_nope = true</code> 且 `
    + `<code>qk_rope_head_dim = 0</code> —— 这个模型的 MLA 完全不带 RoPE 维度,与 DeepSeek 系`
    + `和 Kimi K3 的 512+64 都不同。这是它每 token 的 KV 比 K3 小 2.4 倍的一半原因,`
    + `另一半是持有 KV 的层从 24 降到 11。`,
  ],
};
