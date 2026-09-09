// Kimi K3 —— moonshotai/Kimi-K3
//
// 数据分级(见 docs/glossary.md 的 provenance 五级):
//   结构参数        derived —— config.json
//   权重字节数      derived —— HF API safetensors.parameters + index.json metadata.total_size
//   KDA state dtype estimated —— 结构由 config 推出,但 dtype 是假设
//   KV cache dtype  不属于模型 —— 是引擎运行时选项,已提为界面输入,见 kvDtypes 与 ADR-0007
//
// 注意 index.json 必须走 resolve 端点,raw 只返回 git-lfs 指针。

REG.models["kimi-k3"] = {
  name: "Kimi K3",
  hf: "moonshotai/Kimi-K3",
  blurb: "2.7799T 总参数 · 104B 激活 · 93 层混合注意力(69 KDA + 24 Gated MLA)· 896 routed experts / top-16",
  hidden: 7168,                              // hidden_size,只用于 PP 的 send/recv 激活量

  // ---- 层结构:分段列表,混合注意力是常态而非特例 ----
  layers: [
    { count: 24, family: "mla",    kvLoraRank: 512, qkRopeDim: 64,
      note: "Gated MLA,full attention。latent 被所有 head 共享 → TP 内每卡各存一份完整副本" },
    { count: 69, family: "linear", heads: 96, headDim: 128, stateDtypeBytes: 4,
      note: "KDA 线性注意力。recurrent state 与序列长度无关,per-head 可按 TP 切分,不复制" },
  ],
  totalLayers: 93,

  // ---- KV cache 的 dtype:不是模型属性,是引擎的运行时选项(--kv-cache-dtype)----
  // config.json 里没有这一项,checkpoint 里也没有 —— 它由启动参数决定。
  // 而它把最大并发直接翻倍/砍半,量级超过那个 12 GiB 的 overhead 猜测值,
  // 所以必须是界面上可切的输入,不能作为常数藏在数据文件里。理由见 ADR-0007。
  // 第一项即默认值:引擎默认 auto = 模型 dtype = BF16,不加参数就是这一档。
  kvDtypes: [
    { id: "bf16", label: "BF16(引擎默认)",   bytes: 2 },
    { id: "fp8",  label: "FP8(需显式开启)", bytes: 1 },
  ],

  moe: {
    layers: 92,            // first_k_dense_replace = 1
    experts: 896, topk: 16, sharedExperts: 2,
    latent: 3584, inter: 3072,
  },

  // ---- 权重:两桶实测字节 + 两个参数量,全部 derived,不含残差估算(口径见 ADR-0008)----
  weights: {
    totalBytes:   1_560_860_324_864,   // index.json metadata.total_size
    expertBytes:  1_446_456_066_048,   // routed expert 桶 = 1347.0 GiB。K3 的 quantization_config
                                       // 排除了所有非 expert 模块,所以「总字节 − 非 expert」这个
                                       // 残差恰好就是纯 routed expert,无需逐张量求和
    expertParams:     2_722_740_830_208,   // 92 × 896 × 3 × 3584 × 3072,由 config 推出
    nonExpertParams:     57_191_006_976,   // HF safetensors.parameters:BF16 57,179,884,544 + F32 11,122,432
    nonExpertBf16Params: 57_179_884_544,   // 其中可被「非 expert 权重」开关切成 FP8 的那部分
    nonExpertFixedBytes:     44_489_728,   // F32 norms/scales(11,122,432 × 4 B),开关不动它
    // 两条闭合校验,引擎加载时会硬断言:
    //   57,179,884,544 × 2 + 44,489,728 = 114,404,258,816 B = 106.5 GiB = totalBytes − expertBytes ✓
    //   expertBytes ÷ expertParams = 0.53125,与 mxfp4(4bit + 每32元素一个8bit scale)吻合 ✓
  },

  // ---- 量化 ----
  // routed expert 的原生格式。用于与 instance.nativeDtypes 比对,不匹配则告警。
  quantFormat: "mxfp4",
  // 与 config.json 的 quantization_config.ignore 逐条对应:
  //   self_attn / shared_experts / mlp.(gate|up|gate_up|down)_proj / lm_head / vision_tower / mm_projector
  quantExcludes: "attention / shared experts / MLP projections / lm_head / vision",

  // expert 权重可选格式。bytesPerParam=null 表示「用 index.json 的 total_size 反解出的实测值」
  // (K3 上恰为 0.53125),其余是把 expert 重量化后的推算值 —— 后者不是模型的原生形态。
  // 为什么需要这个:Hopper 无原生 MXFP4,引擎有两条出路 ——
  //   (a) 保持 4bit 在显存里、on-the-fly dequant → 省显存,不省算力
  //   (b) 直接把权重物化成 FP8/BF16      → 拿到原生张量核,但显存也省不到了
  // 两条路的显存后果相反,工具必须都能表达。
  expertFormats: [
    { id: "mxfp4", label: "MXFP4(原生)", bytesPerParam: null },
    { id: "fp8",   label: "FP8",         bytesPerParam: 1 },
    { id: "bf16",  label: "BF16",        bytesPerParam: 2 },
  ],
  // 非 expert 权重:BF16 是实际存储(精确);FP8 是假设重量化。F32 的 norms 两种情况下都不动。
  nonExpertFormats: [
    { id: "bf16", label: "BF16(实际)", bf16Bytes: 2 },
    { id: "fp8",  label: "FP8",        bf16Bytes: 1 },
  ],

  // ---- 候选实例:业务判断,与用户逐个确认,不从 instance 目录自动推导(ADR-0005 §2)。
  // 2026-09-01 确认只保留真正可行的 4 个 —— 都能装下 1453.7 GiB 权重且有 NVSwitch。
  // 排除理由:p5.48xl/p4de 勉强装下但无 FP4 kernel 生态;p4d 四台连权重都装不下;
  //          G 系(g6e/g7/g7e)无 NVLink,域=1,需 2–6 台且 TP 只能为 1。
  // 顺序与 instances.js 一致:P 系优先,越新越前。
  candidateInstances: [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5e.48xlarge",
  ],
  defaultInstance: "p5en.48xlarge",
  // 打开页面的起始切分。1453.7 GiB 权重装不进一台,所以 4 台起。
  defaultParallel: { n: 4, tp: 8, dp: 4, pp: 1, ep: 32 },

  // 每个预设都绑定机型 —— 点它会连机型一起切过去,所以 UI 上必须把机型显示出来。
  presets: [
    { name: "方案 1  TP32/EP32",   inst: "p5en.48xlarge",    n: 4, tp: 32, dp: 1,  pp: 1, ep: 32 },
    { name: "方案 2  TP8/PP4",     inst: "p5en.48xlarge",    n: 4, tp: 8,  dp: 1,  pp: 4, ep: 8  },
    { name: "推荐  TP8×DP4/EP32",  inst: "p5en.48xlarge",    n: 4, tp: 8,  dp: 4,  pp: 1, ep: 32 },
    { name: "反例  纯 DP32",       inst: "p5en.48xlarge",    n: 4, tp: 1,  dp: 32, pp: 1, ep: 32 },
    { name: "b300 两台胜四台 p5en", inst: "p6-b300.48xlarge", n: 2, tp: 8,  dp: 2,  pp: 1, ep: 16 },
    // 台数要让 world 落在 896 的约数上:3 台 = 24 卡,而 24 ∤ 896 → EP 被夹到 8 →
    // expert bank 复制 3 份 → 直接装不下。故用 4 台(32 卡,32 | 896,EP32 干净),
    // 同时构成与「推荐」同台数的对照:同样 4 台,b200 的 108 路 vs p5en 的 68 路
    // (默认 BF16 KV;切成 FP8 KV 则是 212 vs 136 —— 差一倍,这就是 kvDtypes 必须可切的原因)。
    { name: "b200 四台(同台数胜 p5en)", inst: "p6-b200.48xlarge", n: 4, tp: 8, dp: 4, pp: 1, ep: 32 },
  ],

  // ---- 本模型专属的口径说明,渲染进「计算口径与假设」里 ----
  // 通用条目(单位、util、KV、state、overhead、PP)由引擎给;这里只放 K3 自己的一手件溯源。
  notes: [
    `<b>非 expert 参数量 = 57.19B,由权威一手件推导</b>(不再是残差估算)。来源:HF `
    + `<code>/api/models/moonshotai/Kimi-K3</code> 的 <code>safetensors.parameters</code> 给出 `
    + `<code>BF16 57,179,884,544</code> + <code>F32 11,122,432</code>,字节数 = `
    + `<code>BF16×2 + F32×4 = 114,404,258,816 B = 106.5 GiB</code>。<br>`
    + `它含 24 MLA + 69 KDA 的投影、2 个 shared expert、MoE latent 投影、dense 层 0、embedding + LM head。`
    + `<code>quantization_config</code> 明确排除 attention / shared experts / MLP projections,所以这部分不是 MXFP4。<br>`
    + `<b>此前用「2.8T 减 expert 部分」估为 77.3B,高估了 35%。</b>2.8T 本身是向上取整的标称值,真实总参数为 2.7799T。`,

    `<b>expert 权重 = 1347.0 GiB,同样为推导值</b>。<code>model.safetensors.index.json</code> 的 `
    + `<code>metadata.total_size = 1,560,860,324,864 B</code> 减去非 expert 部分即得。除以逻辑参数量 `
    + `<code>2,722,740,830,208</code> 恰好等于 <b>0.53125</b>,与「4 bit + 每 32 元素一个 8 bit scale」的`
    + `理论值完全吻合 —— 三条独立来源互相印证。<br>`
    + `取该文件必须用 <code>resolve</code> 端点;<code>raw</code> 端点只返回 git-lfs 指针。`,

    `MXFP4 在 Hopper(H200)上<b>没有原生张量核支持</b>,引擎走 dequant 到 BF16/FP8 的路径:`
    + `显存节省保留,算力加速拿不到。这张图只算显存,不算这部分性能损失。`,
  ],
};
