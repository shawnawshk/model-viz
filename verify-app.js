// 两个页面的脚本执行验证。CLAUDE.local.md 硬约束 #2:改完 app.html 必须在 DOM stub 下 eval 一遍,
// 并断言各容器非空 —— render() 中途抛异常时页面上半部分看着完全正常。
//
// 每个模型至少一个用例,且必须有一个 PP>1 的用例 —— PP 分支只在那时才渲染,
// 曾因为探针从不设 PP 而漏掉通信表里的一个 NaN。
//
// index.html 的渲染检查见 runIndex():它没有用例表,只断言两个 innerHTML 非空、
// 机型表 th/td 列数自洽、卡片数与挂载的模型数一致。
//
// roofline(ADR-0010)的用例见 CASES 里的 roof 字段:期望值来自独立手算,不是引擎回填。
const fs = require("fs"), path = require("path");
const ROOT = process.argv[2] || __dirname;

const html = fs.readFileSync(path.join(ROOT, "app.html"), "utf8");
const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const srcs = [...html.matchAll(/<script[^>]*\bsrc="([^"]+)"/g)].map(m => m[1]);
const modelSrcs = srcs.filter(s => s.startsWith("data/models/"));
if (inline.length !== 2 || srcs[0] !== "data/instances.js" || modelSrcs.length !== srcs.length - 1)
  throw new Error(`script 标签结构变了:inline=${inline.length} srcs=${srcs.join(",")}`);

// index.html 必须挂同一批 data 文件,否则索引页会漏掉模型
const idxSrcs = [...fs.readFileSync(path.join(ROOT, "index.html"), "utf8")
  .matchAll(/<script[^>]*\bsrc="([^"]+)"/g)].map(m => m[1]);
if (idxSrcs.join(",") !== srcs.join(","))
  throw new Error(`index.html 的 data 文件与 app.html 不一致:\n  app:   ${srcs.join(", ")}\n  index: ${idxSrcs.join(", ")}`);

// ---- 用例:两档 KV dtype 下的 B/token 与最大并发。反事实必须互为对方。----
// 默认口径 util 0.90、ctxIdx 7(128K)、并发滑块在 1 路;用例可用 util / ctxIdx 覆盖。
// expectPresets:预设按钮第三行现算出来的路数(BF16 KV)。它必须随 util / ctxIdx 变 ——
// 最后两条用例就是同一个模型的两个口径,用来证明这一点,别把它们合并掉。
//
// roof(ADR-0010):decode roofline 的期望值。conc 是探针用的并发(没有 expectLoad 时才生效);
// stepMs 取 3 位有效数字;ridge / kvCross 是 "never" / "always" 或一个数(±1%)。
// 期望值全部由独立于引擎的手算脚本按 ADR-0010 §2 的式子算出(输入抄自 data/ 与 datasheet),
// 再与引擎对账 —— 不是先跑引擎再抄回来。
const CASES = [
  { model: "kimi-k3", label: "推荐 4×p5en TP8/DP4/EP32",
    state: { instId: "p5en.48xlarge", n: 4, tp: 8, dp: 4, pp: 1, ep: 32, neFmt: "bf16", expFmt: "mxfp4" },
    expect: { bf16: [27648, 68], fp8: [13824, 136] },
    // 第 3 个(推荐)的 68 与第 6 个(b200 四台)的 108 与 ADR-0007 / 数据文件注释里
    // 独立记下的数一致;第 4 个(反例 纯 DP32)本来就该是 0 —— 单卡 160.6 GiB 超 126.9 预算
    expectPresets: [20, 81, 68, 0, 76, 108],
    // 64 路 × 128K:每步读 97.7 GiB(MLA 全量 KV 59.7 GiB 是大头)→ 21.9 ms,HBM 受限;
    // 权重读取与 KV 读取相等的临界上下文约 98K。4096 路内到不了算力受限。
    roof: { conc: 64, stepMs: "21.9", regime: "mem", ridge: "never", kvCross: 98332 } },
  { model: "glm-5.3-flash", label: "1×p5en TP8/DP1/EP8(KV ×8)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 53], fp8: [5984, 102] },
    expectPresets: [53, 320, 188, 364, 832, 912] },
  { model: "glm-5.3-flash", label: "1×p5en TP1/DP8/EP8(KV ×1)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 320], fp8: [5984, 584] },
    // 1 路:整读非 expert 15.5 GiB + 命中 1 个专家/层 + 一路的 KDA state 读写 → 3.75 ms。
    // KDA state 每 token 读写 285 MB(FP32,34 层)随并发线性涨,斜率比算力项大,所以永远到不了拐点。
    roof: { conc: 1, stepMs: "3.75", regime: "mem", ridge: "never", kvCross: "never" } },
  { model: "glm-5.3-flash", label: "PP2 1×p5en TP1/DP4/PP2/EP4",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 4, pp: 2, ep: 4, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 364], fp8: [5984, 656] },
    // PP 路径:16 路 → 每段 micro-batch 8 个 token、最忙 rank 2 个;ITL = 2 × step。
    roof: { conc: 16, stepMs: "3.40", regime: "mem", ridge: "never", kvCross: "never" } },
  { model: "glm-5.3-flash", label: "两台 2×p5en TP1/DP16/EP16",
    state: { instId: "p5en.48xlarge", n: 2, tp: 1, dp: 16, pp: 1, ep: 16, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 832], fp8: [5984, 1504] } },
  { model: "glm-5.3-flash", label: "1×p6-b300 TP1/DP8/EP8",
    state: { instId: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 912], fp8: [5984, 1648] } },
  // 截图口径:1 路请求只能落在一个 DP rank,不能均摊成每卡 1/8 路。
  { model: "glm-5.3-flash", label: "1×p6-b300 TP1/DP8/EP8 · 1M/util 0.91",
    ctxIdx: 10, util: 91,
    state: { instId: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 120], fp8: [5984, 240] },
    expectLoad: {
      conc: 1,
      requestsByDp: [1, 0, 0, 0, 0, 0, 0, 0],
      perDpCapacity: 15,
      peakUsedGiB: 75.2,
      minUsedGiB: 63.7,
    } },
  // 换口径:1M 上下文 + util 1.00。预设路数必须整组跟着变(与上面那条 128K/0.90 对照)
  { model: "glm-5.3-flash", label: "换口径 1M/util 1.00", ctxIdx: 10, util: 100,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 7], fp8: [5984, 15] },
    // 与上面 128K/0.90 的 [53,320,188,364,832,912] 对照:同一组切分,口径一换整组缩到约 1/6。
    // 这就是预设路数不能写死在数据文件里的原因。
    expectPresets: [7, 48, 28, 56, 128, 136] },

  // ---- DeepSeek-V4.1-Flash(ADR-0009)----
  // 这个模型的 kvDtypes 只有一档(FP4,架构固定),所以探针那两轮 bf16/fp8 会落到同一档 ——
  // 期望值故意写成两个相同的数,再由 kvFixed 显式断言「切 dtype 确实无效」。
  // 890 B/token 是独立外部核对点:它等于 model card 的头号数字「890 bytes per token」。
  { model: "deepseek-v4.1-flash", label: "官方 MP=8:1×p5en TP8/DP1/EP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 488], fp8: [890, 488] }, kvFixed: true,
    // 第 3、4 个预设(TP2/DP4 与 TP1×DP8)在 p5en 上是 0 —— engram 只能按 TP 切,
    // TP<4 时单卡就背不动它。这两个 0 是本模型最该被钉住的东西,别当成写错了删掉。
    expectPresets: [488, 538, 0, 0, 3364, 1680],
    // CSA2 的 decode 读取走 indexer 全扫 + top-512 latent,不是全量 KV:32 路 × 128K 一共只读 15.5 GiB。
    roof: { conc: 32, stepMs: "3.46", regime: "mem", ridge: "never", kvCross: 2850132 } },
  // p5en 上 TP4/DP2 反超官方的 TP8/DP1:engram ÷4 比 ÷8 贵,但 KV 少复制一半。
  { model: "deepseek-v4.1-flash", label: "1×p5en TP4/DP2/EP8(engram ÷4)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 2, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 538], fp8: [890, 538] }, kvFixed: true },
  // 短上下文用例。ADR-0009 §5:滑窗环形 buffer 是 5.000 MiB/请求且与上下文无关,
  // 1K 下它占单请求的 85% —— 漏乘它,1K 的期望值会差 6.75 倍,而 128K 只差 4.5%。
  // **只有这条用例能抓住那个漏乘**,别把它删了。
  { model: "deepseek-v4.1-flash", label: "1K 上下文:滑窗环 buffer 是主导项", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 9682], fp8: [890, 9682] }, kvFixed: true,
    expectPresets: [9682, 10658, 0, 0, 66640, 33340],
    roof: { conc: 256, stepMs: "8.00", regime: "mem", ridge: "never", kvCross: 812625 } },
  // roofline 专用(ADR-0010):三个模型在 p5en / b300 上的所有常规切分,4096 路之内都到不了算力受限 ——
  // 扫描下来只有这一个配置的拐点落在滑块范围内(约 4,022 路),所以用它钉住「算力受限」regime 与拐点二分。
  // 显存上它装不下(engram ÷2 = 94.6 GiB),roofline 不管装不装得下,两个结论并排出现是刻意的。
  { model: "deepseek-v4.1-flash", label: "roofline:TP2/DP4/EP8 · 1K · 4096 路 → 算力受限", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 0], fp8: [890, 0] }, kvFixed: true,
    roof: { conc: 4096, stepMs: "10.1", regime: "comp", ridge: 4022, kvCross: 205264 } },
  // PP>1(每个模型必须有一条)。这里同时钉住 ADR-0009 §8:engram 不按 PP 切,
  // 所以 lookup 在 PP=2 下与 PP=1 相同(都是 189.13 ÷ TP 4 = 47.28 GiB)。
  { model: "deepseek-v4.1-flash", label: "PP2 1×p5en TP4/DP1/PP2/EP4",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 1, pp: 2, ep: 4, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 561], fp8: [890, 561] }, kvFixed: true,
    expectLookupGiB: 47.28 },
  // 装不下的用例。前两个模型都没有「权重本身就超预算」的用例走过这条路径,
  // 而这正是本模型最容易出现的状态(TP 太小 → engram 背不动)。
  { model: "deepseek-v4.1-flash", label: "TP1×DP8:engram 不切 → 装不下",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 0], fp8: [890, 0] }, kvFixed: true,
    expectLookupGiB: 189.13 },

  // ---- Qwen3.8-27B(ADR-0012:第一个 dense、第一个 GQA)----
  // 65,536 B/token = 16 层 × 2 × 4 kv head × 256 × 2 B。KV 复制因子是 max(1, TP/4),不是 TP ——
  // 所以 TP8 与 TP4 的单卡 KV 份额相同(都是 ÷4),TP8 比 TP4 少一半 DP,并发几乎减半。
  // 期望值全部来自独立手算脚本(权重 55,562,855,904 B、state 150,994,944 B/请求、每卡 12 GiB overhead)。
  { model: "qwen3.8-27b", label: "默认 1×p5en TP8/DP1(TP > n_kv_heads,KV ×2)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 53], fp8: [32768, 106] },
    // 六个预设 @128K/0.90:TP4×DP2 最高(100),TP8(53)与 TP1×DP8(56)打平 —— 甜点在 n_kv_heads 上
    expectPresets: [53, 100, 84, 56, 46, 16],
    // 32 路 × 128K:权重 6.47 GiB + 32 路 × 2 GiB KV(÷4 份额,不是 ÷8)+ state 读写 → 71.6 GiB → 16.0 ms
    roof: { conc: 32, stepMs: "16.0", regime: "mem", ridge: "never", kvCross: 10943 } },
  { model: "qwen3.8-27b", label: "1×p5en TP4/DP2(TP = n_kv_heads,KV ×1)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 2, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 100], fp8: [32768, 196] },
    roof: { conc: 64, stepMs: "17.7", regime: "mem", ridge: "never", kvCross: 21886 } },
  // TP=1:每卡背满 51.75 GiB 权重;1 路请求只落在一个 DP rank(8 GiB KV + 144 MiB state)
  { model: "qwen3.8-27b", label: "1×p5en TP1/DP8(KV ×1,权重 ×8)· 1 路",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 56], fp8: [32768, 120] },
    expectLoad: { conc: 1, requestsByDp: [1, 0, 0, 0, 0, 0, 0, 0], perDpCapacity: 7, peakUsedGiB: 71.9, minUsedGiB: 63.7 },
    roof: { conc: 1, stepMs: "13.4", regime: "mem", ridge: "never", kvCross: 843214 } },
  // PP>1(每个模型必须有一条):KV 与 state 都 ÷PP,权重 ÷TP÷PP
  { model: "qwen3.8-27b", label: "PP2 1×p5en TP4/DP1/PP2",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 1, pp: 2, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 106], fp8: [32768, 209] },
    roof: { conc: 16, stepMs: "3.30", regime: "mem", ridge: "never", kvCross: 101370 } },
  // 1K 上下文:GDN state(144 MiB/请求,FP32)是主导项 —— 单卡 16 MiB KV + 18 MiB state
  { model: "qwen3.8-27b", label: "1K 上下文:GDN state 是主导项", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 3265], fp8: [32768, 4270] },
    expectPresets: [3265, 4014, 3504, 2480, 1852, 888],
    // 1024 路:TP8 的 all-reduce 5.2 ms 已经与算力 7.2 ms 同量级,但 state 读写让 HBM 线仍是 13.1 ms
    roof: { conc: 1024, stepMs: "13.1", regime: "mem", ridge: "never", kvCross: "always" } },
  // 官方 FP8 checkpoint(实测 28.75 GiB):权重从 6.47 降到 3.59 GiB/卡,只多出 2 路 —— KV 才是大头
  { model: "qwen3.8-27b", label: "1×p5en TP8/DP1 · 官方 FP8 权重",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 1, neFmt: "fp8", expFmt: "bf16" },
    expect: { bf16: [65536, 55], fp8: [32768, 109] },
    roof: { conc: 32, stepMs: "15.4", regime: "mem", ridge: "never", kvCross: 5055 } },
  // 无 NVLink 机型(ADR-0001 第一次实际起作用):域 = 1,TP 只能是 1,每卡背满权重
  { model: "qwen3.8-27b", label: "1×g7e TP1/DP8(无 NVLink,TP 上限 1)",
    state: { instId: "g7e.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 16], fp8: [32768, 40] },
    roof: { conc: 8, stepMs: "40.4", regime: "mem", ridge: "never", kvCross: 843214 } },
  // g6e:BF16 权重 51.75 + 12 GiB 超出 40.2 GiB 预算 → 装不下;TP=2 显存够但要跨 PCIe
  { model: "qwen3.8-27b", label: "1×g6e TP1/DP8 BF16 → 装不下",
    state: { instId: "g6e.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 0], fp8: [32768, 0] } },
  // g6e + 官方 FP8 + util 0.95 + 1K:装下了,1.7 GiB 余量 → 每卡 8 路。util 0.90 差 0.5 GiB 仍装不下 ——
  // 12 GiB overhead 那个猜测在这台机器上直接决定生死
  { model: "qwen3.8-27b", label: "1×g6e TP1/DP8 · FP8 权重 · 1K/util 0.95", ctxIdx: 0, util: 95,
    state: { instId: "g6e.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1, neFmt: "fp8", expFmt: "bf16" },
    expect: { bf16: [65536, 64], fp8: [32768, 80] } },
  // 3 台 = 24 卡:stage 的约数里有 3 / 6 / 12 / 24,都不整除 4 个 kv head 或 16 个 GDN k head。
  // PR #4 review 抓到建议器曾把 TP6/DP4 当成候选(告警说起不来,卡片却让你「应用到页面」)。
  // 这条用例钉住:建议里出现的 TP 必须都在合法集合 {1, 2, 4, 8} 里(检查在下面 qwen 专属块)。
  { model: "qwen3.8-27b", label: "3×p5en TP4/DP6:建议器不得推荐 TP3/6/12/24",
    state: { instId: "p5en.48xlarge", n: 3, tp: 4, dp: 6, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 300], fp8: [32768, 588] },
    roof: { conc: 48, stepMs: "6.60", regime: "mem", ridge: "never", kvCross: 101370 } },
];

const REQUIRED = ["h1", "sub", "scope-params", "banners", "tiles", "legend", "nodes",
                  "commhead", "commtbl", "tbl", "assumplist", "presets", "eq",
                  "expFmtLab", "precLab", "kvDtLab",
                  "roofhead", "roofverdict", "roofsum", "roofwarn", "rooftbl"];   // ADR-0010
// 这几个容器里出现 undefined / NaN 就是有字段没接上 —— 页面上看起来只是少了个数字
const CLEAN = ["tiles", "tbl", "commtbl", "assumplist", "banners", "roofverdict", "roofsum", "rooftbl"];

// ---- DOM stub ----
const mkEl = id => ({
  id, innerHTML: "", textContent: "", value: "", title: "", max: "",
  style: {}, dataset: {}, classList: { add() {}, remove() {} },
  setAttribute() {}, getAttribute: () => null, addEventListener() {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
  closest: () => null, querySelectorAll: () => [],
});

function run(c) {
  const els = new Map();
  global.document = {
    getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
    querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
    documentElement: { dataset: {} },
  };
  global.location = { search: `?model=${c.model}` };
  global.localStorage = { getItem: () => null, setItem() {} };
  global.matchMedia = () => ({ matches: false });
  global.innerWidth = 1600;
  global.window = global;

  // 数值探针与主脚本一起 eval —— S / compute / render 都是主脚本内部的 const,外面拿不到
  const PROBE = `
globalThis.__probe = [];
for (const kv of ["bf16", "fp8"]) {
  Object.assign(S, ${JSON.stringify(c.state)}, { concIdx: ${c.expectLoad ? `CONC_STEPS.indexOf(${c.expectLoad.conc})` : c.roof ? `CONC_STEPS.indexOf(${c.roof.conc})` : "0"}, ctxIdx: ${c.ctxIdx ?? 7}, util: ${c.util ?? 90}, kvDt: kv });
  syncOptions("probe");
  const C = compute();
  const gpuTotal = Array.from({ length: C.W }, (_, g) => gpuMemory(C, ranks(g)).used)
    .reduce((a, b) => a + b, 0);
  globalThis.__probe.push({ kv, bytesPerToken: C.kvBytesPerToken, maxConc: Math.floor(C.maxConc),
                            altConc: Math.floor(C.maxConcAlt),
                            requestsByDp: C.requestsByDp,
                            perDpCapacity: C.perDpCapacity,
                            totalDeltaBytes: gpuTotal - C.totalUsed,
                            peakUsedGiB: C.used / GIB, minUsedGiB: C.minUsed / GIB,
                            lookupGiB: C.lookup / GIB,
                            tp: S.tp, dp: S.dp, pp: S.pp, ep: S.ep });
  render();
  // 决策层(延迟目标)与通信表都要在 render() 之后读:此前 insightHtml 在 render() 之前取,
  // 拿到的是上一次渲染的内容(首轮是页面默认状态),延迟目标那一栏的断言其实没对着用例状态查
  globalThis.__probe[globalThis.__probe.length - 1].insightHtml = document.getElementById("roofactions").innerHTML;
  globalThis.__probe[globalThis.__probe.length - 1].insightDisplay = document.getElementById("roofactions").style.display;
  globalThis.__probe[globalThis.__probe.length - 1].commHtml =
    document.getElementById("commtbl").innerHTML;
  // roofline(ADR-0010):最忙那张卡的每步下界、regime、拐点、临界上下文
  { const R = roofTerms(C, C.conc, C.ctx), rg = roofRidge(C, C.ctx), kc = roofKvCross(C, C.conc);
    globalThis.__probe[globalThis.__probe.length - 1].roof =
      { stepMs: R.step * 1e3, regime: R.regime, memGiB: R.memBytes / GIB,
        ridge: rg.kind === "at" ? rg.at : rg.kind, kvCross: kc.kind === "at" ? kc.at : kc.kind }; }
}
globalThis.__packExamples = [1, 8, 9].map(n => distributeRequests(n, 8));
// 预设按钮第三行的路数走 presetCompute(),必须跟当前口径(util / ctxIdx / kvDt)一起变。
// 这里在 kvDt=fp8 的那一轮结束后再切回 bf16,才好与 expectPresets 对账。
Object.assign(S, { kvDt: "bf16" }); syncOptions("probe"); render();
globalThis.__presets = PRESETS.map(p => Math.floor(presetCompute(p).maxConc));
markPreset();
// 三个模型都必须能针对当前机型 / 台数给出容量方向的同机型切分反事实。
// 具体候选仍由各 family 的 compute() 约束过滤,不能绕过 KV 复制或 engram 显存规则。
{
  const keep = { ...S }, keepGoal = roofGoal;
  roofGoal = "capacity"; render();
  globalThis.__capacityInsight = document.getElementById("roofactions").innerHTML;
  Object.assign(S, keep); roofGoal = keepGoal; syncOptions("probe"); render();
}
// Kimi 决策层的「建议已采用完」状态:B300 + FP8 KV + 最短上下文。
// 此时降低延迟目标下不应拿同一条「实测校准」重复补满三栏;校准动作只留在「下一次验证」。
if (${JSON.stringify(c.model)} === "kimi-k3") {
  const keep = { ...S };
  Object.assign(S, { instId: "p6-b300.48xlarge", n: 4, tp: 8, dp: 4, pp: 1, ep: 32,
                     concIdx: 0, ctxIdx: 0, kvDt: "fp8", neFmt: "bf16", expFmt: "mxfp4" });
  syncOptions("probe"); render();
  globalThis.__exhaustedInsight = document.getElementById("roofactions").innerHTML;
  Object.assign(S, keep); roofGoal = "latency"; syncOptions("probe"); render();
}
`;
  // ---- 按 html 里的顺序执行:inline[0] → data 文件 → inline[1](主脚本)----
  eval(inline[0]);
  for (const s of srcs) eval(fs.readFileSync(path.join(ROOT, s), "utf8"));
  eval(inline[1] + "\n" + PROBE);

  const fail = [];
  const empty = REQUIRED.filter(id => {
    const e = els.get(id);
    return !e || (!String(e.innerHTML).trim() && !String(e.textContent).trim());
  });
  if (empty.length) fail.push(`空容器:${empty.join(", ")}`);
  for (const id of CLEAN) {
    const s = String(els.get(id)?.innerHTML ?? "") + String(els.get(id)?.textContent ?? "");
    for (const bad of ["undefined", "NaN"]) if (s.includes(bad)) fail.push(`${id} 里出现了 ${bad}`);
  }

  const [b, f] = globalThis.__probe;
  // syncOptions 可能会夹取 TP/DP/PP/EP —— 若被改动说明用例本身写了个非法组合
  for (const p of [b, f]) for (const k of ["tp", "dp", "pp", "ep"])
    if (p[k] !== c.state[k]) fail.push(`用例的 ${k}=${c.state[k]} 被夹到 ${p[k]},组合非法`);
  for (const [kv, [bytes, conc]] of Object.entries(c.expect)) {
    const p = globalThis.__probe.find(x => x.kv === kv);
    if (p.bytesPerToken !== bytes) fail.push(`${kv}: B/token 应为 ${bytes},实为 ${p.bytesPerToken}`);
    if (p.maxConc !== conc) fail.push(`${kv}: 最大并发应为 ${conc} 路,实为 ${p.maxConc} 路`);
  }
  if (b.altConc !== f.maxConc || f.altConc !== b.maxConc)
    fail.push(`反事实不对称:bf16.alt=${b.altConc} vs fp8.max=${f.maxConc};fp8.alt=${f.altConc} vs bf16.max=${b.maxConc}`);
  // kvDtypeFixed 的模型(ADR-0009 §3):dtype 写死在架构里,切它必须完全无效。
  // 若哪天有人给这种模型加了第二档 dtype,这条会当场喊出来。
  if (c.kvFixed && (b.bytesPerToken !== f.bytesPerToken || b.maxConc !== f.maxConc))
    fail.push(`kvFixed 的模型切 dtype 竟然有效:${b.bytesPerToken}B/${b.maxConc}路 vs ${f.bytesPerToken}B/${f.maxConc}路`);
  if (c.expectLookupGiB !== undefined && b.lookupGiB.toFixed(2) !== c.expectLookupGiB.toFixed(2))
    fail.push(`按 TP 切的查表单卡应为 ${c.expectLookupGiB.toFixed(2)} GiB,实为 ${b.lookupGiB.toFixed(2)} GiB`);
  for (const p of [b, f])
    if (Math.abs(p.totalDeltaBytes) > 1)
      fail.push(`${p.kv}:逐卡显存求和与集群合计相差 ${p.totalDeltaBytes} B`);
  if (c.expectLoad) {
    if (String(b.requestsByDp) !== String(c.expectLoad.requestsByDp))
      fail.push(`DP 请求分配应为 [${c.expectLoad.requestsByDp}],实为 [${b.requestsByDp}]`);
    if (b.perDpCapacity !== c.expectLoad.perDpCapacity)
      fail.push(`每个 DP rank 应容纳 ${c.expectLoad.perDpCapacity} 路,实为 ${b.perDpCapacity} 路`);
    if (b.peakUsedGiB.toFixed(1) !== c.expectLoad.peakUsedGiB.toFixed(1))
      fail.push(`单卡峰值应为 ${c.expectLoad.peakUsedGiB.toFixed(1)} GiB,实为 ${b.peakUsedGiB.toFixed(1)} GiB`);
    if (b.minUsedGiB.toFixed(1) !== c.expectLoad.minUsedGiB.toFixed(1))
      fail.push(`单卡最低占用应为 ${c.expectLoad.minUsedGiB.toFixed(1)} GiB,实为 ${b.minUsedGiB.toFixed(1)} GiB`);
    const packs = [[1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1], [2,1,1,1,1,1,1,1]];
    if (String(globalThis.__packExamples) !== String(packs))
      fail.push(`DP 整数分配 1/8/9 路不正确:实为 ${JSON.stringify(globalThis.__packExamples)}`);
    if (!b.commHtml.includes("<td>attention all-reduce</td><td>0</td><td>TP 1</td>"))
      fail.push("TP=1 时 communication table 应显示 0 次 attention all-reduce");
  }

  // 预设按钮上的路数(BF16 KV,该用例的 util / ctxIdx 口径下)
  if (c.expectPresets && String(globalThis.__presets) !== String(c.expectPresets))
    fail.push(`预设路数应为 [${c.expectPresets}],实为 [${globalThis.__presets}]`);

  // roofline(ADR-0010)。每个用例都要:step 有限且 > 0、regime 合法 —— 这两条抓的是 NaN 与字段没接上。
  for (const p of [b, f]) {
    if (!(p.roof.stepMs > 0 && Number.isFinite(p.roof.stepMs))) fail.push(`${p.kv}:roofline step 不是正数:${p.roof.stepMs}`);
    if (!["mem", "comp", "comm"].includes(p.roof.regime)) fail.push(`${p.kv}:roofline regime 非法:${p.roof.regime}`);
  }
  // 有手算期望的用例:3 位有效数字对账;拐点 / 临界上下文允许 ±1%(两边都是二分,收敛点会差最后一位)
  if (c.roof) {
    const r = b.roof, near = (got, want) => typeof want === "string" ? got === want
      : (typeof got === "number" && Math.abs(got - want) <= Math.max(1, 0.01 * want));
    if (Number(r.stepMs).toPrecision(3) !== c.roof.stepMs)
      fail.push(`roofline step 应为 ${c.roof.stepMs} ms,实为 ${Number(r.stepMs).toPrecision(3)} ms`);
    if (r.regime !== c.roof.regime) fail.push(`roofline regime 应为 ${c.roof.regime},实为 ${r.regime}`);
    if (!near(r.ridge, c.roof.ridge)) fail.push(`拐点应为 ${c.roof.ridge},实为 ${typeof r.ridge === "number" ? r.ridge.toFixed(1) : r.ridge}`);
    if (!near(r.kvCross, c.roof.kvCross)) fail.push(`临界上下文应为 ${c.roof.kvCross},实为 ${r.kvCross}`);
  }

  // 所有模型页面都必须把 roofline 诊断翻成目标、反事实动作和「先别做」。
  // 候选仍受各自 compute() 的 family 约束,并固定当前 PP、不新引入跨 NVLink 域 TP。
  const insight = b.insightHtml;
  for (const text of ["下一步怎么做", "降低延迟", "提高吞吐", "增加容量", "同机型切分可尝试", "先别做"])
    if (!insight.includes(text)) fail.push(`${c.model} 决策层缺少「${text}」`);
  if (b.insightDisplay === "none") fail.push(`${c.model} 决策层不应隐藏`);
  if (insight.includes('data-tp="16"') || insight.includes('data-tp="32"'))
    fail.push(`${c.model} 同机型建议不应新引入跨 NVLink 域的 TP`);

  const capacityInsight = String(globalThis.__capacityInsight || "");
  if (capacityInsight.includes("data-roof-parallel")) {
    if (!capacityInsight.includes(`data-pp="${c.state.pp}"`))
      fail.push(`${c.model} 同机型建议 v1 应固定当前 PP=${c.state.pp}`);
    if (capacityInsight.includes(`data-tp="${c.state.tp}" data-dp="${c.state.dp}" data-pp="${c.state.pp}" data-ep="${c.state.ep}"`))
      fail.push(`${c.model} 同机型切分候选不应重复当前切分`);
  }

  if (c.model === "kimi-k3") {
    if (!capacityInsight.includes("data-roof-parallel"))
      fail.push("Kimi 容量目标下没有可应用的同机型并行切分");
    if (!insight.includes("没有可信的同机型切分改善"))
      fail.push("Kimi 默认延迟目标应诚实说明没有可信的同机型切分改善");
    const exhausted = String(globalThis.__exhaustedInsight || "");
    const exhaustedCards = (exhausted.match(/class="ra-card"/g) || []).length;
    if (exhaustedCards !== 0)
      fail.push(`建议采用完后动作卡应为空,实为 ${exhaustedCards} 张`);
    if (exhausted.includes("先做实测校准"))
      fail.push("「实测校准」不应作为重复动作卡,只应留在下一次验证");
  }
  if (c.model === "glm-5.3-flash" && c.label.startsWith("1×p5en TP8/DP1")) {
    if (!capacityInsight.includes("data-roof-parallel"))
      fail.push("GLM 容量目标下没有可应用的同机型并行切分");
    if (!capacityInsight.includes("TP1 / DP8 / PP1 / EP8"))
      fail.push("GLM 容量建议应识别 TP1/DP8 可减少 KV 复制");
    if (!capacityInsight.includes("每路 KV 的跨卡副本从 ×8 降到 ×1"))
      fail.push("GLM 容量建议应解释 TP1/DP8 的收益来自减少 KV 副本");
  }
  if (c.model === "deepseek-v4.1-flash" && c.label.startsWith("官方 MP=8")) {
    if (!capacityInsight.includes("data-roof-parallel"))
      fail.push("DeepSeek 容量目标下没有可应用的同机型并行切分");
    if (!capacityInsight.includes("TP4 / DP2 / PP1 / EP8"))
      fail.push("DeepSeek 容量建议应保留装得下 engram 的 TP4/DP2");
    if (capacityInsight.includes("TP1 / DP8") || capacityInsight.includes("TP2 / DP4"))
      fail.push("DeepSeek 容量建议不应包含装不下 engram 的 TP1/TP2");
    if (!capacityInsight.includes("每卡engram 查表 23.6 → 47.3 GiB"))
      fail.push("DeepSeek 容量建议应显式写出降低 TP 会增加单卡 engram 的代价");
    if (capacityInsight.includes('data-roof-set="kvDt"'))
      fail.push("DeepSeek 的 KV dtype 架构固定,不应建议切换 KV dtype");
  }

  if (c.model === "qwen3.8-27b") {
    // dense(ADR-0012):页面上不能残留任何 MoE 专属的行、徽章、图例;EP 恒为 1
    const legend = String(els.get("legend")?.innerHTML ?? ""), tiles = String(els.get("tiles")?.innerHTML ?? "");
    const bannersHtml = String(els.get("banners")?.innerHTML ?? ""), nodes = String(els.get("nodes")?.innerHTML ?? "");
    const assump = String(els.get("assumplist")?.innerHTML ?? ""), roof = String(els.get("rooftbl")?.innerHTML ?? "");
    if (legend.includes("routed expert")) fail.push("dense 模型的图例里不应有 routed expert");
    if (tiles.includes("expert")) fail.push("dense 模型的 tiles 里不应出现 expert");
    if (bannersHtml.includes("expert bank")) fail.push("dense 模型不应出现 expert bank 复制告警");
    if (nodes.includes("专家 #")) fail.push("dense 模型的 GPU 徽章里不应有专家编号");
    if (assump.includes("routed expert 逻辑参数量")) fail.push("dense 模型的假设区不应有 routed expert 参数量那条");
    if (roof.includes("命中的 routed expert") || roof.includes("命中专家的矩阵乘")) fail.push("dense 模型的 roofline 算式里不应有 expert 行");
    if (!roof.includes("dense 模型,没有 MoE 层")) fail.push("dense 模型的 roofline 通信行应写明没有 MoE 层");
    if (!b.commHtml.includes("dense 模型,没有 MoE 层")) fail.push("dense 模型的通信表应写明没有 MoE 层");
    if (b.ep !== 1 || f.ep !== 1) fail.push(`dense 模型的 EP 应恒为 1,实为 ${b.ep}`);
    if (/data-ep="(?!1")/.test(capacityInsight)) fail.push("dense 模型的同机型建议不应出现 EP≠1 的候选");
    // 建议器只能给出引擎起得来的 TP:24 q head / 4 kv head / 16 GDN k head → 合法 TP 只有 1 / 2 / 4 / 8
    for (const [name, html] of [["延迟目标", insight], ["容量目标", capacityInsight]])
      for (const m of html.matchAll(/data-tp="(\d+)"/g))
        if (![1, 2, 4, 8].includes(Number(m[1]))) fail.push(`${name}的同机型建议推荐了不合法的 TP${m[1]}(不整除 head 数,引擎会拒绝启动)`);
    if (c.state.n === 3 && !capacityInsight.includes("data-roof-parallel") && !insight.includes("data-roof-parallel"))
      fail.push("3×p5en 上过滤掉非法 TP 后,合法候选(如 TP8/DP3、TP2/DP12)仍应存在,不应一个建议都没有");
    // GQA:复制因子随 TP 变,两个方向的告警都要在;KV 单卡份额 = ÷ min(TP, 4)
    if (c.state.tp > 4 && !bannersHtml.includes("超过 n_kv_heads=4")) fail.push("TP>4 时应有「KV 复制」告警");
    if (c.state.tp > 1 && c.state.tp <= 4 && !bannersHtml.includes("KV 不复制")) fail.push("1<TP≤4 时应有「KV 不复制」的 good banner");
    if (c.state.tp === 8 && !tiles.includes("复制 <b>2</b> 份")) fail.push("TP8 的 hero 应写 KV 复制 2 份(不是 8 份)");
    if (c.state.tp === 4 && !tiles.includes("不复制")) fail.push("TP4 的 hero 应写 KV 不复制");
    if (c.label.startsWith("默认 1×p5en TP8/DP1")) {
      if (!capacityInsight.includes("TP4 / DP2 / PP1 / EP1")) fail.push("Qwen 容量建议应识别 TP4/DP2(KV 不再复制,DP 翻倍)");
      if (!capacityInsight.includes("每路 KV 的跨卡副本从 ×2 降到 ×1")) fail.push("Qwen 容量建议应把副本数写成 ×2 → ×1(GQA),不是 ×8 → ×4");
    }
    // 无 NVLink 机型上 TP=1:不该有跨域告警;g6e BF16 装不下时要点出「TP 上限是 1,权重切不开」
    if (c.state.instId === "g7e.48xlarge" && bannersHtml.includes("超出 NVLink 域")) fail.push("g7e TP=1 不应有跨域告警");
    if (c.state.instId === "g6e.48xlarge" && c.state.neFmt === "bf16" && !bannersHtml.includes("权重切不开")) fail.push("g6e BF16 装不下时应点出 TP 上限 1、权重切不开");
    // 「非 expert 权重」在 dense 页上应叫「权重」
    if (String(els.get("precName")?.textContent ?? "") !== "权重") fail.push("dense 模型的权重格式控件应叫「权重」");
    if (String(els.get("epCtl")?.style?.display) !== "none" || String(els.get("expFmtCtl")?.style?.display) !== "none")
      fail.push("dense 模型应隐藏 EP 与 expert 权重两个控件");
  }

  return { fail, probe: globalThis.__probe, presets: globalThis.__presets };
}

// ---- index.html ----
// 上面的用例只覆盖 app.html。index.html 那两个 innerHTML 一直没人管,而它同样会
// 半途出错而不露相:机型表的 <th> 与 <td> 数一旦对不上,整张表往后错一列,页面其余
// 部分照样正常。行数与卡片数分别与 data 对账,顺带钉住 split("<tr>") 的切法。
function runIndex() {
  const idxHtml = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
  const idxInline = [...idxHtml.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (idxInline.length !== 2)
    throw new Error(`index.html 的 inline script 结构变了:${idxInline.length} 个,期望 2 个`);

  const els = new Map();
  global.document = {
    getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
    querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
    documentElement: { dataset: {} },
  };
  global.location = { search: "" };
  global.localStorage = { getItem: () => null, setItem() {} };
  global.matchMedia = () => ({ matches: false });
  global.window = global;

  // 与 app.html 同一个顺序:inline[0] 建 REG → data 文件填充 → inline[1] 渲染
  eval(idxInline[0]);
  for (const s of idxSrcs) eval(fs.readFileSync(path.join(ROOT, s), "utf8"));
  eval(idxInline[1]);

  const fail = [];
  const txt = id => String(els.get(id)?.innerHTML ?? "") + String(els.get(id)?.textContent ?? "");
  const IDX_REQUIRED = ["models", "insts", "foot"];
  for (const id of IDX_REQUIRED) {
    if (!txt(id).trim()) fail.push(`空容器:${id}`);
    for (const b of ["undefined", "NaN"]) if (txt(id).includes(b)) fail.push(`${id} 里出现了 ${b}`);
  }

  const tbl = String(els.get("insts")?.innerHTML ?? "");
  // 叶子列数 = 不带 colspan 的 th 数(表头一行或多行都成立);每个数据行的 td 数必须与之相等。
  // 分隔行(tr.grp,有 NVLink 与没有的机型之间那一行)不算数据行,但它的 colspan 必须横跨全部叶子列。
  const nTh = (tbl.match(/<th(?![^>]*\bcolspan=)[\s>]/g) || []).length;
  const bodyRows = (tbl.split("<tbody>")[1] || "").split("<tr").slice(1);
  const rows = bodyRows.filter(r => !r.startsWith(' class="grp"'));
  const grps = bodyRows.filter(r => r.startsWith(' class="grp"'));
  const tdCounts = [...new Set(rows.map(r => (r.match(/<td[\s>]/g) || []).length))];
  if (tdCounts.length !== 1) fail.push(`机型表各行 td 数不一致:${tdCounts.join(", ")}`);
  else if (tdCounts[0] !== nTh) fail.push(`机型表叶子 th=${nTh} 与 td=${tdCounts[0]} 不匹配,表格会错一列`);
  for (const g of grps) {
    const m = g.match(/colspan="(\d+)"/);
    if (!m || Number(m[1]) !== nTh) fail.push(`分隔行 colspan=${m ? m[1] : "无"},应为叶子列数 ${nTh}`);
  }
  if (rows.length !== Object.keys(REG.instances).length)
    fail.push(`机型表 ${rows.length} 行,但 data 里有 ${Object.keys(REG.instances).length} 个机型`);

  const nCards = (String(els.get("models")?.innerHTML ?? "").match(/class="card"/g) || []).length;
  if (nCards !== modelSrcs.length)
    fail.push(`模型卡 ${nCards} 张,但挂了 ${modelSrcs.length} 个模型`);

  return { fail, nCards, nRows: rows.length, nTh };
}

let bad = 0;
for (const c of CASES) {
  const { fail, probe, presets } = run(c);
  const tag = `${c.model} · ${c.label}`;
  if (fail.length) {
    bad++;
    console.error(`✗ ${tag}`);
    for (const m of fail) console.error(`    ${m}`);
    console.table(probe);
  } else {
    console.log(`✓ ${tag} — BF16 ${probe[0].maxConc} 路 / FP8 ${probe[1].maxConc} 路,`
      + `${probe[0].bytesPerToken} / ${probe[1].bytesPerToken} B/token,反事实对称`
      + (c.expectPresets ? `,预设 [${presets}]` : "")
      + (c.roof ? ` · roofline ${Number(probe[0].roof.stepMs).toPrecision(3)} ms ${probe[0].roof.regime}` : ""));
  }
}
// 放在用例之后:runIndex 会重建 global.document,此时已经没人用了
const idx = runIndex();
if (idx.fail.length) {
  console.error("✗ index.html");
  for (const m of idx.fail) console.error(`    ${m}`);
} else {
  console.log(`✓ index.html — ${idx.nCards} 张模型卡 · 机型表 ${idx.nRows} 行 × ${idx.nTh} 列,th/td 自洽`);
}

if (bad || idx.fail.length) {
  if (bad) console.error(`\n✗ ${bad}/${CASES.length} 个用例失败`);
  if (idx.fail.length) console.error("✗ index.html 渲染检查失败");
  process.exit(1);
}
console.log(`\n✓ ${CASES.length} 个用例全过 · ${REQUIRED.length} 个容器非空 · index.html 渲染自洽 · `
  + `${modelSrcs.length} 个模型(${modelSrcs.map(s => path.basename(s, ".js")).join(", ")})`);
