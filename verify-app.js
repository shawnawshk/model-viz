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
if (inline.length !== 2 || srcs[0] !== "i18n.js" || srcs[1] !== "data/instances.js"
    || modelSrcs.length !== srcs.length - 3 || srcs[srcs.length - 1] !== "sidebar.js")
  throw new Error(`script 标签结构变了:inline=${inline.length} srcs=${srcs.join(",")}`);

// 点预设时并发必须走 concIdxForDp(p.dp)。这条只能静态查:DOM stub 里 addEventListener 是空操作,
// 派发不了点击事件,而下面 __presetLoads 是在探针里**重算**同一个公式,改坏 handler 它照样通过。
// 钉死的是**参数**而不只是函数名 —— 第一版传的是总卡数,那会让每个 DP rank 拿 TP×PP 路
// (K3 TP32/DP1 本来 1 路就均匀,却被设成 32 路、白白撞红)。
{
  const h = html.match(/getElementById\("presets"\)\.addEventListener\([\s\S]*?\n\}\);/);
  if (!h) throw new Error("找不到 presets 的 click handler —— 这条静态检查失效了,先修它");
  if (!h[0].includes("concIdxForDp(p.dp)"))
    throw new Error("presets click handler 必须用 concIdxForDp(p.dp) 设并发(不是总卡数派生量)");
  if (/concIdx:\s*0\b/.test(h[0]))
    throw new Error("presets click handler 又把并发写回 1 路了");
}

// 两个页面里静态引用的 assets/ 文件必须存在 —— 图片路径写错,页面上只是一个空框,不会报错
for (const page of ["app.html", "index.html"])
  for (const [, ref] of fs.readFileSync(path.join(ROOT, page), "utf8").matchAll(/(?:src|href)="(assets\/[^"]+)"/g))
    if (!fs.existsSync(path.join(ROOT, ref))) throw new Error(`${page} 引用的 ${ref} 不存在`);

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
    // 第一个是新置顶的「b300 一台 TP4×DP2/EP8」:一台 b300 装得下 K3,一台 p5en 装不下。
    // 18 路独立手算(128K / util 0.90 / BF16 KV / MXFP4 expert):
    //   预算 275040 MiB ÷1024 ×0.90 = 241.73 GiB;非 expert 114,404,258,816 B ÷TP4 = 26.64;
    //   expert 1347.0 ÷EP8 = 168.38;+12 overhead → 固定 207.01 → 每卡余量 34.72 GiB。
    //   每路 = MLA KV 131,072 × 27,648 B(latent 不除 TP)= 3.3750 + KDA state 0.4043÷TP4 = 0.1011
    //        = 3.4761 GiB。floor(34.72 / 3.4761) = 9 路/DP rank × DP2 = **18**。
    expectPresets: [18, 20, 81, 68, 0, 76, 108],
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

  // ---- DeepSeek-V4.1-Flash(ADR-0009 / 0013)----
  // 这个模型的 kvDtypes 只有一档(FP4,架构固定),所以探针那两轮 bf16/fp8 会落到同一档 ——
  // 期望值故意写成两个相同的数,再由 kvFixed 显式断言「切 dtype 确实无效」。
  // 890 B/token 是独立外部核对点:它等于 model card 的头号数字「890 bytes per token」。
  //
  // 2026-10-30(ADR-0013):engram 放在哪提成了输入,默认 = vLLM 默认的 host offload。
  // 前三条是新默认口径;后面带 lookupPlace: "hbm" 的是原来那组「整张常驻 HBM」用例,数字逐字未改 ——
  // 常驻那条代码路径(÷TP 不 ÷PP、TP 下限、装不下的红字、建议器里的 engram 代价)仍要有人钉住。
  { model: "deepseek-v4.1-flash", label: "默认(engram 放 host):官方 MP=8 1×p5en TP8/DP1/EP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 697], fp8: [890, 697] }, kvFixed: true, expectLookupGiB: 0,
    // 放 host 时 TP 越小越高:TP1×DP8 是 MP=8 的 7 倍;常驻时那两个预设是 0(见下面 hbm 那条)
    expectPresets: [697, 1370, 2648, 4928, 6696, 3348],
    // roofline 不看 engram 放哪(它每 token 只查几十行):与常驻那条的 3.46 ms 相同
    roof: { conc: 32, stepMs: "3.46", regime: "mem", ridge: "never", kvCross: 2850132 } },
  { model: "deepseek-v4.1-flash", label: "engram 放 host:TP1×DP8 → 常驻时装不下的那一档",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 4928], fp8: [890, 4928] }, kvFixed: true, expectLookupGiB: 0,
    // TP1:非 expert 10.45 GiB 整读 + 命中的 expert,64 路 × 128K 的 CSA2 读取压不过权重,临界上下文超出 2^24
    roof: { conc: 64, stepMs: "7.16", regime: "mem", ridge: "never", kvCross: "never" } },
  { model: "deepseek-v4.1-flash", label: "1K · engram 放 host", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4" },
    expect: { bf16: [890, 13806], fp8: [890, 13806] }, kvFixed: true,
    expectPresets: [13806, 27158, 52492, 97696, 132632, 66340] },
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · 官方 MP=8:1×p5en TP8/DP1/EP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 488], fp8: [890, 488] }, kvFixed: true,
    // 第 3、4 个预设(TP2/DP4 与 TP1×DP8)在常驻时是 0 —— engram 只能按 TP 切,
    // TP<4 时单卡就背不动它。这两个 0 是常驻那一档最该被钉住的东西,别当成写错了删掉。
    expectPresets: [488, 538, 0, 0, 3364, 1680],
    // CSA2 的 decode 读取走 indexer 全扫 + top-512 latent,不是全量 KV:32 路 × 128K 一共只读 15.5 GiB。
    roof: { conc: 32, stepMs: "3.46", regime: "mem", ridge: "never", kvCross: 2850132 } },
  // p5en 上 TP4/DP2 反超官方的 TP8/DP1:engram ÷4 比 ÷8 贵,但 KV 少复制一半。
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · 1×p5en TP4/DP2/EP8(engram ÷4)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 2, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 538], fp8: [890, 538] }, kvFixed: true },
  // 短上下文用例。ADR-0009 §5:滑窗环形 buffer 是 5.000 MiB/请求且与上下文无关,
  // 1K 下它占单请求的 85% —— 漏乘它,1K 的期望值会差 6.75 倍,而 128K 只差 4.5%。
  // **只有这条与上面的「1K · engram 放 host」能抓住那个漏乘**,别把它们删了。
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · 1K 上下文:滑窗环 buffer 是主导项", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 9682], fp8: [890, 9682] }, kvFixed: true,
    expectPresets: [9682, 10658, 0, 0, 66640, 33340],
    roof: { conc: 256, stepMs: "8.00", regime: "mem", ridge: "never", kvCross: 812625 } },
  // roofline 专用(ADR-0010):各模型在 p5en / b300 上的常规切分,4096 路之内都到不了算力受限 ——
  // 扫描下来只有这一个配置的拐点落在滑块范围内(约 4,022 路),所以用它钉住「算力受限」regime 与拐点二分。
  // 显存上它在常驻时装不下(engram ÷2 = 94.6 GiB),roofline 不管装不装得下,两个结论并排出现是刻意的。
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · roofline:TP2/DP4/EP8 · 1K · 4096 路 → 算力受限", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 0], fp8: [890, 0] }, kvFixed: true,
    roof: { conc: 4096, stepMs: "10.1", regime: "comp", ridge: 4022, kvCross: 205264 } },
  // PP>1(每个模型必须有一条)。这里同时钉住 ADR-0009 §8:engram 不按 PP 切,
  // 所以常驻时 lookup 在 PP=2 下与 PP=1 相同(都是 189.13 ÷ TP 4 = 47.28 GiB)。
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · PP2 1×p5en TP4/DP1/PP2/EP4",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 1, pp: 2, ep: 4, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 561], fp8: [890, 561] }, kvFixed: true,
    expectLookupGiB: 47.28 },
  // 装不下的用例:「权重本身就超预算」这条路径。常驻时这是本模型最容易出现的状态(TP 太小 → engram 背不动)。
  { model: "deepseek-v4.1-flash", label: "常驻 HBM · TP1×DP8:engram 不切 → 装不下",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "mxfp4", lookupPlace: "hbm" },
    expect: { bf16: [890, 0], fp8: [890, 0] }, kvFixed: true,
    expectLookupGiB: 189.13 },

  // ---- GLM-5.3(78 层全 DSA + IndexCache;expert bank 703.7 GiB)----
  // 92,628 B/token = 78 × 576 × 2(latent,BF16)+ 21 × 132(indexer key,只在 full 层存,走 vLLM 的 FP8 + fp32 scale 布局)。
  // FP8 档 47,700 = 78 × 576 + 2,772 —— indexer 那一项不随 --kv-cache-dtype 变。期望值全部来自独立手算。
  { model: "glm-5.3", label: "默认 2×p5en TP8/DP2/EP16(KV ×8)",
    state: { instId: "p5en.48xlarge", n: 2, tp: 8, dp: 2, pp: 1, ep: 16, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 12], fp8: [47700, 22] },
    // 六个预设 @128K/0.90:一台 TP8(recipe)只有 2 路,一台 TP1×DP8 一路 128K 的 KV 都放不进;两台 PP2 最高
    expectPresets: [12, 2, 64, 88, 0, 88],
    // 8 路:非 expert 2.44 GiB + 命中的 expert 整读;57 个 shared 层不扫 indexer,所以 KV 读取不随上下文线性涨到压过权重
    roof: { conc: 8, stepMs: "3.12", regime: "mem", ridge: "never", kvCross: 1086908 } },
  // vLLM recipe 的 8×H200 标准配置:权重装得下,128K 下只剩 2 路(FP8 KV 4 路)
  { model: "glm-5.3", label: "vLLM recipe:1×p5en TP8/DP1/EP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 2], fp8: [47700, 4] },
    roof: { conc: 8, stepMs: "5.69", regime: "mem", ridge: "never", kvCross: 968546 } },
  // 与 GLM-5.3-Flash 方向相反:一台上 TP1×DP8 权重装得下(117.1 / 126.9 GiB),但一路 128K 的 KV(11.31 GiB)都放不进
  { model: "glm-5.3", label: "1×p5en TP1/DP8/EP8:权重装得下,128K 一路 KV 都放不进",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 0], fp8: [47700, 8] } },
  // PP>1(每个模型必须有一条)
  { model: "glm-5.3", label: "PP2 2×p5en TP1/DP8/PP2/EP8",
    state: { instId: "p5en.48xlarge", n: 2, tp: 1, dp: 8, pp: 2, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 88], fp8: [47700, 168] },
    roof: { conc: 16, stepMs: "4.36", regime: "mem", ridge: "never", kvCross: 14841368 } },
  { model: "glm-5.3", label: "1×p6-b300 TP1/DP8/EP8",
    state: { instId: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 88], fp8: [47700, 168] },
    roof: { conc: 64, stepMs: "13.5", regime: "mem", ridge: "never", kvCross: 4431214 } },
  // recipe:「要跑满 1M 需要 8×B200 + FP8 KV」—— 本页口径下 b200 TP8 1M 正好 FP8 1 路、BF16 0 路
  { model: "glm-5.3", label: "1×p6-b200 TP8 · 1M:FP8 KV 才放得下一路", ctxIdx: 10,
    state: { instId: "p6-b200.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [92628, 0], fp8: [47700, 1] },
    expectPresets: [0, 0, 0, 8, 0, 8] },
  // 官方 BF16 checkpoint(逐张量实测 1,368 GiB expert):两台 b300 才宽裕。钉住 measured 那条文案路径
  { model: "glm-5.3", label: "2×p6-b300 TP8/DP2/EP16 · 官方 BF16 expert",
    state: { instId: "p6-b300.48xlarge", n: 2, tp: 8, dp: 2, pp: 1, ep: 16, neFmt: "native", expFmt: "bf16" },
    expect: { bf16: [92628, 24], fp8: [47700, 48] } },

  // ---- Qwen3.8-Flash-Next(ADR-0013 / 0014:QSA + n-gram 表;MoE + GQA + GDN 第一次同时出现)----
  // 25,344 B/token = 12 × 2 × 2 kv head × 256 × 2(主 KV,BF16)+ 12 × 128 × 2 ÷ 4(indexer 压缩 key,BF16,不随 --kv-cache-dtype 变)。
  // 单卡:主 KV ÷ min(TP, 2),indexer key 不除。n-gram 表默认放 host。期望值全部来自独立手算。
  { model: "qwen3.8-flash-next", label: "默认 TEP8:1×p5en TP8/DP1/EP8(KV ×4)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 52], fp8: [13056, 99] }, expectLookupGiB: 0,
    // 七个预设 @128K/0.90:TP2×DP4 最高(196),官方 TEP8 只有它的约 1/4
    expectPresets: [52, 102, 196, 184, 18, 60, 472],
    roof: { conc: 8, stepMs: "1.46", regime: "mem", ridge: "never", kvCross: 871144 } },
  { model: "qwen3.8-flash-next", label: "1×p5en TP2/DP4/EP8(TP = n_kv_heads,KV ×1)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 196], fp8: [13056, 360] },
    roof: { conc: 32, stepMs: "4.49", regime: "mem", ridge: "never", kvCross: 3013378 } },
  // 常驻那一档:n-gram 表 95.37 GiB ÷ TP
  { model: "qwen3.8-flash-next", label: "常驻 HBM · 1×p5en TP4/DP2/EP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 2, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16", lookupPlace: "hbm" },
    expect: { bf16: [25344, 72], fp8: [13056, 136] }, expectLookupGiB: 23.84,
    expectPresets: [45, 72, 80, 0, 11, 0, 360] },
  { model: "qwen3.8-flash-next", label: "常驻 HBM · 1×p5en TP1/DP8:n-gram 表不切 → 装不下",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16", lookupPlace: "hbm" },
    expect: { bf16: [25344, 0], fp8: [13056, 0] }, expectLookupGiB: 95.37 },
  { model: "qwen3.8-flash-next", label: "p5 H100 TEP8(官方 H100 配置)",
    state: { instId: "p5.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 18], fp8: [13056, 35] },
    roof: { conc: 8, stepMs: "2.09", regime: "mem", ridge: "never", kvCross: 871144 } },
  // PP>1(每个模型必须有一条):数字照算,但页面必须出「vLLM 起不来」的红字(检查在下面 qwen-next 专属块)
  { model: "qwen3.8-flash-next", label: "PP2 1×p5en TP4/DP1/PP2/EP4(vLLM 不支持)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 4, dp: 1, pp: 2, ep: 4, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 104], fp8: [13056, 195] },
    roof: { conc: 16, stepMs: "1.36", regime: "mem", ridge: "never", kvCross: 1775056 } },
  // 1K:GDN state(108 MiB/请求,FP32)÷8 后仍占单请求的 51%
  { model: "qwen3.8-flash-next", label: "1K 上下文 · TEP8", ctxIdx: 0,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 3312], fp8: [13056, 4293] },
    expectPresets: [3312, 4308, 4972, 4688, 1170, 1604, 12020],
    roof: { conc: 1024, stepMs: "15.5", regime: "mem", ridge: "never", kvCross: 193 } },
  // 官方 FP8 checkpoint:expert 与 n-gram 表一起换。放 host 时只省 expert;常驻时 n-gram 表也跟着减半(47.68 ÷ 8 = 5.96)
  { model: "qwen3.8-flash-next", label: "官方 FP8 checkpoint · TEP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "bf16", expFmt: "fp8" },
    expect: { bf16: [25344, 61], fp8: [13056, 115] } },
  { model: "qwen3.8-flash-next", label: "常驻 HBM · 官方 FP8 checkpoint · TEP8",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "bf16", expFmt: "fp8", lookupPlace: "hbm" },
    expect: { bf16: [25344, 58], fp8: [13056, 108] }, expectLookupGiB: 5.96 },
  { model: "qwen3.8-flash-next", label: "1×p6-b300 TP2/DP4/EP8",
    state: { instId: "p6-b300.48xlarge", n: 1, tp: 2, dp: 4, pp: 1, ep: 8, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [25344, 472], fp8: [13056, 872] },
    roof: { conc: 64, stepMs: "4.02", regime: "mem", ridge: "never", kvCross: 2028466 } },

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
  // 2026-09-17:g6e 不再是候选(它的生死线由未校准的 12 GiB overhead 单独决定,见
  // qwen3.8-27b.js 的 candidateInstances 注释),原先那两条 g6e 用例删除。本模型的「装不下」
  // 用例改由 1M 上下文 + g7e TP1×DP8 承担 —— 权重装得下但**一路 KV 都放不进**,
  // 与 DSv4.1 那条「权重本身就超预算」是不同的代码路径,两条都要留。
  // 手算:g7e 预算 96 × 0.90 = 86.4;权重 55,562,855,904 B ÷ TP1 = 51.7465,+12 overhead = 63.7465
  //   → 余量 22.6535 GiB。1M 单路 BF16 = 64.0 KV + 0.1406 state = 64.1406 GiB > 22.65 → 0 路;
  //   FP8 = 32.0 + 0.1406 = 32.1406 GiB,仍 > 22.65 → 也是 0 路(切 dtype 救不回来)。
  { model: "qwen3.8-27b", label: "1×g7e TP1/DP8 · 1M → 一路 KV 都放不进", ctxIdx: 10,
    state: { instId: "g7e.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 0], fp8: [32768, 0] } },
  // 3 台 = 24 卡:stage 的约数里有 3 / 6 / 12 / 24,都不整除 4 个 kv head 或 16 个 GDN k head。
  // PR #4 review 抓到建议器曾把 TP6/DP4 当成候选(告警说起不来,卡片却让你「应用到页面」)。
  // 这条用例钉住:建议里出现的 TP 必须都在合法集合 {1, 2, 4, 8} 里(检查在下面 qwen 专属块)。
  { model: "qwen3.8-27b", label: "3×p5en TP4/DP6:建议器不得推荐 TP3/6/12/24",
    state: { instId: "p5en.48xlarge", n: 3, tp: 4, dp: 6, pp: 1, ep: 1, neFmt: "bf16", expFmt: "bf16" },
    expect: { bf16: [65536, 300], fp8: [32768, 588] },
    // 探针停在 1 路:此时延迟目标下 TP8/DP3(权重 ÷8)是真改善,候选必然存在,断言才不是空的;
    // 48 路时 TP4/DP6 自己就是延迟与容量两个目标的最优,没有候选是对的,不是 bug
    roof: { conc: 1, stepMs: "3.36", regime: "mem", ridge: "never", kvCross: 843214 } },
];

const REQUIRED = ["h1", "sub", "scope-params", "banners", "tiles", "legend", "nodes", "model-nav",
                  "commhead", "commtbl", "tbl", "assumplist", "presets", "eq",
                  "expFmtLab", "precLab", "kvDtLab",
                  "roofhead", "roofverdict", "roofsum", "roofwarn", "rooftbl"];   // ADR-0010
// 这几个容器里出现 undefined / NaN 就是有字段没接上 —— 页面上看起来只是少了个数字
const CLEAN = ["tiles", "tbl", "commtbl", "assumplist", "banners", "roofverdict", "roofsum", "rooftbl", "model-nav"];

function checkNavigation(els, currentModel) {
  const markup = String(els.get("model-nav")?.innerHTML ?? "");
  const fail = [];
  const links = [...markup.matchAll(/<a\b[^>]*href="app\.html\?model=([^"]+)"[^>]*>/g)];
  const ids = links.map(link => link[1]).sort();
  if (ids.join(",") !== Object.keys(REG.models).sort().join(","))
    fail.push("provider 导航必须包含每个已注册模型,且每个模型只出现一次");
  const selected = links.filter(link => link[0].includes('aria-current="page"'));
  if (currentModel && (selected.length !== 1 || selected[0][1] !== currentModel))
    fail.push(`provider 导航应选中 ${currentModel}`);
  for (const [, source] of markup.matchAll(/<img\b[^>]*src="([^"]+)"/g))
    if (!fs.existsSync(path.join(ROOT, source))) fail.push(`provider logo 文件不存在:${source}`);
  return fail;
}

// ---- DOM stub ----
const mkEl = id => ({
  id, innerHTML: "", textContent: "", value: "", title: "", max: "",
  style: {}, dataset: {}, classList: { add() {}, remove() {} },
  setAttribute() {}, getAttribute: () => null, addEventListener() {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
  closest: () => null, querySelectorAll: () => [],
});

// i18n.js 在繁体时用 document.write 同步插入 OpenCC;stub 里直接在全局作用域 eval 那个文件。
// 必须用间接 eval:在本模块的函数里直接 eval,UMD 会看到 CommonJS 的 exports,把 OpenCC 挂到 module.exports 上
const docWrite = s => { const m = String(s).match(/src="([^"]+)"/); if (m) (0, eval)(fs.readFileSync(path.join(ROOT, m[1]), "utf8")); };

// 原有用例一律固定用简体跑(?lang=zh-Hans),断言里的中文字符串才有意义;另外两种语言见下面的 runLang()
function run(c, lang = "zh-Hans") {
  const els = new Map();
  global.document = {
    getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
    querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
    documentElement: { dataset: {} }, title: "", write: docWrite,
  };
  global.location = { search: `?model=${c.model}&lang=${lang}` };
  global.localStorage = { getItem: () => null, setItem() {} };
  global.matchMedia = () => ({ matches: false });
  global.innerWidth = 1600;
  global.window = global;

  // 数值探针与主脚本一起 eval —— S / compute / render 都是主脚本内部的 const,外面拿不到
  const PROBE = `
// 页面刚加载、还没被探针改状态时的负载分配。并发默认「每个 DP rank 1 路」= DP,
// 目的是让 DP>1 时每个 rank 拿到相同的整数路数(1 路会是 [1,0,0,…],图上只有第一张卡有 KV),
// 同时不把本来均匀的 DP=1 推成红字。必须在下面的 Object.assign 之前取,否则拿到的是探针的 concIdx。
globalThis.__defaultLoad = (() => {
  const C = compute();
  return { conc: C.conc, world: C.W, dp: S.dp, requestsByDp: C.requestsByDp,
           floor: CONC_FLOOR, steps: CONC_STEPS.slice() };
})();
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
// 点预设时并发被设成该预设的 DP(app.html 的 presets click)。这里复算同一个式子,
// 断言每个预设都落在 DP 上 —— 固定 8 会让 DP16 / DP32 剩余数,总卡数会让每 rank 拿 TP×PP 路。
globalThis.__presetLoads = PRESETS.map(p => {
  const w = p.n * REG.instances[p.inst].gpusPerInstance;
  const conc = CONC_STEPS[concIdxForDp(p.dp)];
  return { name: p.name, world: w, dp: p.dp, conc };
});
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
  // 页面默认状态(1 路)下的延迟目标:此前这条断言读的是探针改状态前的首次渲染,恰好就是 1 路;
  // 探针改为 render() 之后读以后,拿到的是 roof 用例的 64 路 —— 那时 TP4/DP8/EP32 是真改善,所以这里单独渲染 1 路
  Object.assign(S, { concIdx: 0 }); roofGoal = "latency"; syncOptions("probe"); render();
  globalThis.__defaultInsight = document.getElementById("roofactions").innerHTML;
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
  fail.push(...checkNavigation(els, c.model));
  // 每个模型页都要有 vLLM recipe 外链(数据文件的 vllmRecipe),而且必须指向 recipes.vllm.ai —— 页面里引用的 recipe 配置都出自那里
  if (!String(els.get("model-recipe")?.href ?? "").startsWith("https://recipes.vllm.ai/"))
    fail.push(`${c.model} 缺 vLLM recipe 链接(data 里的 vllmRecipe)`);
  // SGLang cookbook 外链同理(data 里的 sglangCookbook),必须指向 docs.sglang.io —— 不用 lmsysorg.mintlify.app,那是它的托管地址,页面 canonical 是前者
  if (!String(els.get("model-sglang")?.href ?? "").startsWith("https://docs.sglang.io/cookbook/"))
    fail.push(`${c.model} 缺 SGLang cookbook 链接(data 里的 sglangCookbook)`);
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

  // ---- 并发默认值:能被 DP 整除、且 ≥ CONC_FLOOR 的最小一档。四条性质各抓一种错法:
  //   均匀        —— 原本的显示 bug:DP>1 时 1 路只有第一张卡有 KV(分配 [1,0,0,…]);
  //   ≥ 下限      —— 「所有模型页面都不许默认 1 路」这条要求;DP=1 的三个模型光靠整除会停在 1;
  //   能被 DP 整除 —— 有余数就不均匀;
  //   最小        —— 第一版的过头修法:conc=总卡数 让每个 rank 拿 TP×PP 路(K3 默认变每 rank
  //                  8 路;TP32/DP1 本来 1 路就均匀,却被设成 32 路而上限只有 20、白白撞红)。
  const dl = globalThis.__defaultLoad;
  // **下限的期望值必须写死在这里**,不能只用 app 自报的 dl.floor —— 否则把 CONC_FLOOR 改成 1
  // 断言会跟着变松、静默放过(第一版就是这样,漏掉了这条错法)。8 是扫过全部 28 个切分定下来的:
  // 装得下的配置里上限最低的是 Qwen g7e TP1×DP8 的 16 路。要动这个数,得连带重扫一遍会不会撞红。
  const WANT_FLOOR = 8;
  if (dl.floor !== WANT_FLOOR)
    fail.push(`CONC_FLOOR 应为 ${WANT_FLOOR},实为 ${dl.floor} —— 改它要重扫各切分会不会撞红`);
  const check = (what, conc, dp, byDp) => {
    if (byDp && new Set(byDp).size !== 1)
      fail.push(`${what}各 DP rank 的路数应完全相同,实为 [${byDp}](DP ${dp})`);
    if (conc < dl.floor)
      fail.push(`${what}并发 ${conc} 低于下限 ${dl.floor} —— 所有模型页面都不许默认 1 路`);
    if (conc % dp !== 0)
      fail.push(`${what}并发 ${conc} 不能被 DP${dp} 整除,各卡会不均`);
    const want = dl.steps.find(c => c % dp === 0 && c >= dl.floor);
    if (want === undefined)
      fail.push(`${what}DP${dp} 在 CONC_STEPS 里找不到 ≥ ${dl.floor} 的可整除档`);
    else if (conc !== want)
      fail.push(`${what}并发应取满足条件的最小一档 ${want}(DP${dp}),实为 ${conc}`);
  };
  check("默认口径下", dl.conc, dl.dp, dl.requestsByDp);
  for (const pl of globalThis.__presetLoads) check(`预设「${pl.name}」点开后`, pl.conc, pl.dp, null);

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
    if (!String(globalThis.__defaultInsight || "").includes("没有可信的同机型切分改善"))
      fail.push("Kimi 默认状态(1 路)的延迟目标应诚实说明没有可信的同机型切分改善");
    // 64 路下 TP4/DP8/EP32 让每个 DP rank 少一半 token,ITL 下界确实更低 —— ADR-0011 对 K3 验证的正是「降 TP 升 DP 减 MLA KV 副本」
    if (!insight.includes("TP4 / DP8 / PP1 / EP32"))
      fail.push("Kimi 64 路的延迟目标应给出 TP4/DP8/EP32(每 rank 少一半 token)");
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
  // 查表放置(ADR-0013):控件只在带查表的模型上出现;两档各有自己的 banner,反事实路数与另一档的用例互相对得上
  {
    const hasLookup = !!REG.models[c.model].weights.tpShardedBytes;
    const bannersHtml = String(els.get("banners")?.innerHTML ?? "");
    if (String(els.get("lookupCtl")?.style?.display) !== (hasLookup ? "" : "none"))
      fail.push(`${c.model} 的「查表放置」控件应${hasLookup ? "显示" : "隐藏"}`);
    if (hasLookup && c.state.lookupPlace === "hbm" && !bannersHtml.includes("整张常驻 HBM(不是 vLLM 默认)"))
      fail.push("常驻 HBM 时应有「不是 vLLM 默认」的 warning");
    if (hasLookup && c.state.lookupPlace !== "hbm" && !bannersHtml.includes("放在 host 内存(vLLM 默认)"))
      fail.push("默认口径下应有「放在 host 内存(vLLM 默认)」的 banner");
    if (hasLookup && c.state.lookupPlace === "hbm" && !capacityInsight.includes('data-roof-set="lookupPlace"'))
      fail.push("常驻 HBM 时,容量目标的首选动作应是把查表放回 host");
  }

  if (c.model === "deepseek-v4.1-flash" && c.label.startsWith("默认(engram 放 host)")) {
    // 放 host 时 engram 不在 HBM,建议器里不该再出现「降 TP 会增加单卡 engram」这类代价
    if (capacityInsight.includes("engram 查表"))
      fail.push("engram 放 host 时,容量建议不应再写单卡 engram 的代价");
  }
  if (c.model === "deepseek-v4.1-flash" && c.label.startsWith("常驻 HBM · 官方 MP=8")) {
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

  if (c.model === "glm-5.3") {
    const roof = String(els.get("rooftbl")?.innerHTML ?? ""), bannersHtml = String(els.get("banners")?.innerHTML ?? "");
    // IndexCache:57 个 shared 层的 decode 读法必须单独写出来(不扫 indexer)
    if (!roof.includes("复用上一个 full 层的 top-k")) fail.push("GLM-5.3 的 roofline 应写出 shared 层复用 top-k、不扫 indexer");
    // 第一个没有 recurrent state、也没有滑窗环的模型:图例 / 明细表里不该留一个 0 的 state 行
    for (const id of ["legend", "tbl"]) if (String(els.get(id)?.innerHTML ?? "").includes("recurrent state"))
      fail.push(`GLM-5.3 没有 recurrent state,${id} 里不该出现这一项`);
    if (c.label.startsWith("默认 2×p5en")) {
      if (!capacityInsight.includes("TP1 / DP16 / PP1 / EP16")) fail.push("GLM-5.3 两台 TP8/DP2 的容量建议应识别 TP1/DP16");
      if (!capacityInsight.includes("每路 KV 的跨卡副本从 ×8 降到 ×1")) fail.push("GLM-5.3 容量建议应解释收益来自 KV 副本 ×8 → ×1");
    }
    if (c.state.expFmt === "bf16" && !bannersHtml.includes("官方 checkpoint 逐张量实测"))
      fail.push("GLM-5.3 的 BF16 expert 是官方 checkpoint 实测档,banner 不应写成按参数量推算");
  }

  if (c.model === "qwen3.8-flash-next") {
    const bannersHtml = String(els.get("banners")?.innerHTML ?? ""), tiles = String(els.get("tiles")?.innerHTML ?? "");
    // 合法 TP:24 q head / 2 kv head(可被 TP 整除或整除 TP)/ 48 v 与 16 k 个 GDN head → 1 / 2 / 4 / 8
    for (const [name, html] of [["延迟目标", insight], ["容量目标", capacityInsight]])
      for (const m of html.matchAll(/data-tp="(\d+)"/g))
        if (![1, 2, 4, 8].includes(Number(m[1]))) fail.push(`${name}的同机型建议推荐了不合法的 TP${m[1]}`);
    // QSA 的主 KV 复制因子 max(1, TP/2);indexer key 不算进「复制几份」里(否则 TP2 会被误判成复制 1.03 份)
    if (c.state.tp > 2 && !bannersHtml.includes("超过 n_kv_heads=2")) fail.push("TP>2 时应有「KV 复制」告警");
    if (c.state.tp === 2 && !bannersHtml.includes("KV 不复制")) fail.push("TP=2 时应有「KV 不复制」的 good banner");
    if (c.state.tp === 8 && !tiles.includes("复制 <b>4</b> 份")) fail.push("TP8 的 hero 应写 KV 复制 4 份");
    if (c.state.tp > 1 && !bannersHtml.includes("indexer 的压缩 key")) fail.push("QSA 的 KV banner 应交代 indexer key 每卡一份");
    if (c.state.pp > 1 && !bannersHtml.includes("起不来")) fail.push("PP>1 时应有「vLLM 起不来」的红字(n-gram 表不支持 PP)");
    if (c.label.startsWith("默认 TEP8")) {
      if (!capacityInsight.includes("TP2 / DP4 / PP1 / EP8")) fail.push("Qwen-Next 容量建议应识别 TP2/DP4(TP = n_kv_heads)");
      if (!capacityInsight.includes("每路 KV 的跨卡副本从 ×4 降到 ×1")) fail.push("Qwen-Next 容量建议应写 KV 副本 ×4 → ×1");
    }
    if (c.state.expFmt === "fp8") {
      if (!bannersHtml.includes("官方 checkpoint 逐张量实测")) fail.push("官方 FP8 checkpoint 那一档应标为逐张量实测");
      if (!bannersHtml.includes("降到")) fail.push("BF16 → FP8 是变小,banner 不应写「涨到」");
    }
    if (c.state.lookupPlace === "hbm" && !bannersHtml.includes("这一档的路数是上限"))
      fail.push("Qwen-Next 常驻 HBM 时应显示 recipe 里那条「常驻会 OOM」的反证");
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
    // 无 NVLink 机型上 TP=1:不该有跨域告警
    if (c.state.instId === "g7e.48xlarge" && bannersHtml.includes("超出 NVLink 域")) fail.push("g7e TP=1 不应有跨域告警");
    // 已知覆盖缺口(2026-09-17):「TP 上限 1,权重切不开」那条 banner 现在**没有用例覆盖**。
    // 它要求「无 NVLink 且权重本身超预算」,而 g6e 撤出候选后,四个模型的候选里再没有这种机型
    // (g7e 96 GiB 上 Qwen BF16 权重 63.75 < 86.4 预算,装得下)。这条路径因此在 UI 上不可达。
    // 不用后门 state 去测一个用户选不到的机型 —— 等 overhead 提为输入、g6e 可能加回时再补。
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
function runIndex(lang = "zh-Hans") {
  const idxHtml = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
  const idxInline = [...idxHtml.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (idxInline.length !== 2)
    throw new Error(`index.html 的 inline script 结构变了:${idxInline.length} 个,期望 2 个`);

  const els = new Map();
  global.document = {
    getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
    querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
    documentElement: { dataset: {} }, title: "", write: docWrite,
  };
  global.location = { search: `?lang=${lang}` };
  global.localStorage = { getItem: () => null, setItem() {} };
  global.matchMedia = () => ({ matches: false });
  global.window = global;

  // 与 app.html 同一个顺序:inline[0] 建 REG → data 文件填充 → inline[1] 渲染
  eval(idxInline[0]);
  for (const s of idxSrcs) eval(fs.readFileSync(path.join(ROOT, s), "utf8"));
  eval(idxInline[1]);

  const fail = [];
  const txt = id => String(els.get(id)?.innerHTML ?? "") + String(els.get(id)?.textContent ?? "");
  const IDX_REQUIRED = ["models", "insts", "foot", "model-nav"];
  for (const id of IDX_REQUIRED) {
    if (!txt(id).trim()) fail.push(`空容器:${id}`);
    for (const b of ["undefined", "NaN"]) if (txt(id).includes(b)) fail.push(`${id} 里出现了 ${b}`);
  }
  fail.push(...checkNavigation(els));

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

  return { fail, nCards, nRows: rows.length, nTh, els };
}

// ---- 界面语言(ADR-0015)----
// 四组检查,各抓一种错法:
//   (1) 静态标记:页面里每一段可见的中文都要有英文(data-en / data-en-title / data-en-aria-label),否则英文页面上会露出中文;
//   (2) 数据文件:每个要显示的字段都有 xxxEn,而且中英两版的数字、<code> 片段、链接逐个一致 —— 翻译最容易错的就是数字;
//   (3) 每个模型、首页在英文 / 繁体下各渲染一次:不许有 undefined / NaN,英文里不许剩汉字,繁体里不许剩简体字;
//   (4) 语言判定顺序:?lang= > 上次的选择 > 浏览器语言。
const HAN = /[㐀-鿿豈-﫿　-〿＀-￯]/;
function checkStaticMarkup(file) {
  const src = fs.readFileSync(path.join(ROOT, file), "utf8");
  const fail = [];
  const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);
  const tok = /<!--[\s\S]*?-->|<script\b[\s\S]*?<\/script>|<style\b[\s\S]*?<\/style>|<(\/?)([a-zA-Z][\w-]*)((?:\s+[^\s=>\/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>|([^<]+)/g;
  const attrsOf = a => Object.fromEntries([...a.matchAll(/([^\s=\/]+)(?:\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+)))?/g)].map(m => [m[1], m[3] ?? m[4] ?? m[5] ?? ""]));
  const stack = [];
  let htmlAttrs = {};
  for (const m of src.matchAll(tok)) {
    if (m[2]) {
      const tag = m[2].toLowerCase();
      if (m[1]) { const i = stack.map(x => x.tag).lastIndexOf(tag); if (i >= 0) stack.length = i; continue; }
      const at = attrsOf(m[3] || "");
      if (tag === "html") htmlAttrs = at;
      for (const [k, v] of Object.entries(at)) {
        if (k.startsWith("data-en") && HAN.test(v)) fail.push(`${file}: ${k} 里有汉字:${v.slice(0, 40)}`);
        if ((k === "title" || k === "aria-label") && HAN.test(v) && !(`data-en-${k}` in at))
          fail.push(`${file}: <${tag} ${k}="${v.slice(0, 30)}"> 缺 data-en-${k}`);
      }
      if (!VOID.has(tag) && !m[4]) stack.push({ tag, en: "data-en" in at });
    } else if (m[5] && HAN.test(m[5]) && !/^(<!--|<script|<style)/.test(m[0])) {
      const inTitle = stack.some(x => x.tag === "title");
      if (inTitle ? !("data-en-doc-title" in htmlAttrs) : !stack.some(x => x.en))
        fail.push(`${file}: 这段文字没有英文(缺 data-en):${m[5].trim().slice(0, 40)}`);
    }
  }
  return fail;
}
function checkDataTranslations() {
  const fail = [];
  const SCALARS = ["blurb", "indexerAsm", "tpShardedLabel", "tpShardedShort", "tpShardedNote", "stateNote",
                   "lookupHbmCaveat", "ppUnsupported", "expertLayerDesc", "nonExpertFixedDesc"];
  // 数字:逗号后紧跟 3 位才算千分位;「X 亿」= X×100 M、「X 万」= X/100 M,对应英文的「…M」
  const nums = t => (t.replace(/<[^>]+>/g, " ")
      .replace(/(\d+(?:\.\d+)?)\s*亿/g, (_, x) => " " + Math.round(parseFloat(x) * 100) + " ")
      .replace(/(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*万/g, (_, x) => " " + +(parseFloat(x.replace(/,/g, "")) / 100).toFixed(2) + " ")
      .match(/\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?/g) || []).map(x => x.replace(/,/g, "")).sort().join(" ");
  // <code> 不看顺序逐个配对(英文语序会调换「X 的 Y」):逐字节相同;只有中文那一段本身带汉字(如「16 层」)时,
  // 这一段放宽到只比数字与符号。每个中文片段都要配上一个英文片段,英文也不许多出来
  const relax = c => c.replace(/[A-Za-z㐀-鿿]+/g, "").replace(/\s+/g, " ").trim();
  const codeList = t => [...t.matchAll(/<code>([\s\S]*?)<\/code>/g)].map(x => x[1]);
  const sameCodes = (zh, en) => { const e = codeList(en);
    for (const c of codeList(zh)) {
      const i = HAN.test(c) ? e.findIndex(x => relax(x) === relax(c)) : e.indexOf(c);
      if (i < 0) return false;
      e.splice(i, 1);
    }
    return e.length === 0; };
  const hrefs = t => [...t.matchAll(/href="([^"]+)"/g)].map(x => x[1]).sort().join(" ");
  const pair = (where, zh, en) => {
    if (zh == null) return;
    if (en == null) { fail.push(`${where} 缺英文(${where.split(".").pop()}En)`); return; }
    if (HAN.test(en)) fail.push(`${where} 的英文里有汉字`);
    if (nums(zh) !== nums(en)) fail.push(`${where} 中英数字不一致:[${nums(zh)}] vs [${nums(en)}]`);
    if (!sameCodes(zh, en)) fail.push(`${where} 中英 <code> 片段不一致`);
    if (hrefs(zh) !== hrefs(en)) fail.push(`${where} 中英链接不一致`);
  };
  global.REG = { instances: {}, models: {} };
  for (const s of modelSrcs) (0, eval)(fs.readFileSync(path.join(ROOT, s), "utf8"));
  for (const [id, m] of Object.entries(REG.models)) {
    for (const k of SCALARS) pair(`${id}.${k}`, m[k], m[k + "En"]);
    if ((m.notesEn || []).length !== (m.notes || []).length) fail.push(`${id}.notesEn 与 notes 条数不同`);
    (m.notes || []).forEach((n, i) => pair(`${id}.notes[${i}]`, n, (m.notesEn || [])[i]));
    for (const [arr, key] of [["kvDtypes", "label"], ["expertFormats", "label"], ["nonExpertFormats", "label"], ["presets", "name"], ["layers", "label"]])
      (m[arr] || []).forEach((o, i) => pair(`${id}.${arr}[${i}].${key}`, o[key], o[key + "En"]));
  }
  return fail;
}
// 英文 / 繁体下渲染一次页面默认状态。英文:可见输出里不许有汉字(语言下拉框本身除外);
// 繁体:逐字检查,cn→tw 的字形转换还会改动的字就是漏转的简体字
function langLeftovers(els, lang) {
  const out = new Set();
  if (lang === "zh-Hant" && typeof OpenCC === "undefined") return ["繁体没有加载 OpenCC(vendor/opencc-js/cn2t.js)"];
  const ch = lang === "zh-Hant" ? OpenCC.Converter({ from: "cn", to: "tw" }) : null;
  for (const [id, e] of els) {
    if (id === "lang") continue;
    const t = String(e.innerHTML ?? "") + String(e.textContent ?? "") + String(e.title ?? "");
    if (lang === "en") { for (const m of t.matchAll(/.{0,20}[㐀-鿿豈-﫿　-〿＀-￯].{0,20}/g)) out.add(`${id}: …${m[0]}…`); }
    else for (const m of t.matchAll(/[㐀-鿿]/g)) if (ch(m[0]) !== m[0]) out.add(`${id}: 「${m[0]}」应为「${ch(m[0])}」`);
  }
  return [...out];
}
function runLang(model, lang) {
  const els = new Map();
  global.document = {
    getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
    querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
    documentElement: { dataset: {} }, title: "", write: docWrite,
  };
  global.location = { search: `?model=${model}&lang=${lang}` };
  global.localStorage = { getItem: () => null, setItem() {} };
  global.matchMedia = () => ({ matches: false });
  global.innerWidth = 1600;
  global.window = global;
  eval(inline[0]);
  for (const s of srcs) eval(fs.readFileSync(path.join(ROOT, s), "utf8"));
  eval(inline[1]);
  const fail = [];
  const empty = REQUIRED.filter(id => { const e = els.get(id); return !e || (!String(e.innerHTML).trim() && !String(e.textContent).trim()); });
  if (empty.length) fail.push(`空容器:${empty.join(", ")}`);
  for (const id of CLEAN) for (const b of ["undefined", "NaN"])
    if ((String(els.get(id)?.innerHTML ?? "") + String(els.get(id)?.textContent ?? "")).includes(b)) fail.push(`${id} 里出现了 ${b}`);
  fail.push(...langLeftovers(els, lang).slice(0, 8));
  if (document.documentElement.lang !== { en: "en", "zh-Hant": "zh-TW" }[lang]) fail.push(`<html lang> 应为 ${{ en: "en", "zh-Hant": "zh-TW" }[lang]},实为 ${document.documentElement.lang}`);
  return fail;
}
function checkLangDetection() {
  const fail = [];
  const pick = (search, saved, nav) => {
    global.location = { search }; global.localStorage = { getItem: () => saved, setItem() {} };
    global.document = { documentElement: { dataset: {} }, write() {} };
    const keep = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    Object.defineProperty(globalThis, "navigator", { value: { languages: [nav], language: nav }, configurable: true, writable: true });
    try { (0, eval)(fs.readFileSync(path.join(ROOT, "i18n.js"), "utf8")); return I18N.LANG; }
    finally { if (keep) Object.defineProperty(globalThis, "navigator", keep); else delete globalThis.navigator; }
  };
  const want = [[["?lang=en", "zh-Hant", "zh-CN"], "en"], [["", "zh-Hant", "en-US"], "zh-Hant"], [["", null, "zh-TW"], "zh-Hant"],
                [["", null, "zh-HK"], "zh-Hant"], [["", null, "zh-Hant-TW"], "zh-Hant"], [["", null, "zh-CN"], "zh-Hans"],
                [["", null, "zh"], "zh-Hans"], [["", null, "en-US"], "en"], [["", null, "ja-JP"], "en"], [["?lang=xx", null, "zh-CN"], "zh-Hans"]];
  for (const [[q, sv, nv], w] of want) { const got = pick(q, sv, nv); if (got !== w) fail.push(`?lang=${q || "-"} / 上次=${sv} / 浏览器=${nv} 应判为 ${w},实为 ${got}`); }
  return fail;
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

// 界面语言(ADR-0015)
const i18nFail = [];
for (const f of ["app.html", "index.html"]) i18nFail.push(...checkStaticMarkup(f));
i18nFail.push(...checkDataTranslations());
for (const lang of ["en", "zh-Hant"]) {
  for (const id of Object.keys(REG.models)) i18nFail.push(...runLang(id, lang).map(m => `${lang} · ${id}:${m}`));
  const ix = runIndex(lang);
  i18nFail.push(...ix.fail.map(m => `${lang} · index.html:${m}`), ...langLeftovers(ix.els, lang).slice(0, 8).map(m => `${lang} · index.html:${m}`));
}
i18nFail.push(...checkLangDetection());
if (i18nFail.length) {
  console.error("✗ 界面语言");
  for (const m of i18nFail.slice(0, 60)) console.error(`    ${m}`);
} else {
  console.log(`✓ 界面语言 — 静态标记全部有英文 · 数据文件中英数字逐个一致 · ${Object.keys(REG.models).length} 个模型 + 首页在 EN / 繁体下渲染干净 · 语言判定 10 种情形`);
}

if (bad || idx.fail.length || i18nFail.length) {
  if (bad) console.error(`\n✗ ${bad}/${CASES.length} 个用例失败`);
  if (idx.fail.length) console.error("✗ index.html 渲染检查失败");
  if (i18nFail.length) console.error(`✗ 界面语言检查失败(${i18nFail.length} 处)`);
  process.exit(1);
}
console.log(`\n✓ ${CASES.length} 个用例全过 · ${REQUIRED.length} 个容器非空 · index.html 渲染自洽 · `
  + `${modelSrcs.length} 个模型(${modelSrcs.map(s => path.basename(s, ".js")).join(", ")})`);
