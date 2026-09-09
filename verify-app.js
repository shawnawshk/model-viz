// app.html 的脚本执行验证。CLAUDE.md 硬约束 #2:改完 app.html 必须在 DOM stub 下 eval 一遍,
// 并断言各容器非空 —— render() 中途抛异常时页面上半部分看着完全正常。
//
// 每个模型至少一个用例,且必须有一个 PP>1 的用例 —— PP 分支只在那时才渲染,
// 曾因为探针从不设 PP 而漏掉通信表里的一个 NaN。
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
const CASES = [
  { model: "kimi-k3", label: "推荐 4×p5en TP8/DP4/EP32",
    state: { instId: "p5en.48xlarge", n: 4, tp: 8, dp: 4, pp: 1, ep: 32, neFmt: "bf16", expFmt: "mxfp4" },
    expect: { bf16: [27648, 69], fp8: [13824, 136] },
    // 第 3 个(推荐)的 69 与第 6 个(b200 四台)的 109 与 ADR-0007 / 数据文件注释里
    // 独立记下的数一致;第 4 个(反例 纯 DP32)本来就该是 0 —— 单卡 160.6 GiB 超 126.9 预算
    expectPresets: [20, 81, 69, 0, 77, 109] },
  { model: "glm-5.3-flash", label: "1×p5en TP8/DP1/EP8(KV ×8)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 53], fp8: [5984, 102] },
    expectPresets: [53, 325, 191, 365, 838, 918] },
  { model: "glm-5.3-flash", label: "1×p5en TP1/DP8/EP8(KV ×1)",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 325], fp8: [5984, 585] } },
  { model: "glm-5.3-flash", label: "PP2 1×p5en TP1/DP4/PP2/EP4",
    state: { instId: "p5en.48xlarge", n: 1, tp: 1, dp: 4, pp: 2, ep: 4, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 365], fp8: [5984, 656] } },
  { model: "glm-5.3-flash", label: "两台 2×p5en TP1/DP16/EP16",
    state: { instId: "p5en.48xlarge", n: 2, tp: 1, dp: 16, pp: 1, ep: 16, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 838], fp8: [5984, 1506] } },
  { model: "glm-5.3-flash", label: "1×p6-b300 TP1/DP8/EP8",
    state: { instId: "p6-b300.48xlarge", n: 1, tp: 1, dp: 8, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 918], fp8: [5984, 1649] } },
  // 换口径:1M 上下文 + util 1.00。预设路数必须整组跟着变(与上面那条 128K/0.90 对照)
  { model: "glm-5.3-flash", label: "换口径 1M/util 1.00", ctxIdx: 10, util: 100,
    state: { instId: "p5en.48xlarge", n: 1, tp: 8, dp: 1, pp: 1, ep: 8, neFmt: "native", expFmt: "fp8" },
    expect: { bf16: [11616, 7], fp8: [5984, 15] },
    // 与上面 128K/0.90 的 [53,325,191,365,838,918] 对照:同一组切分,口径一换整组缩到约 1/6。
    // 这就是预设路数不能写死在数据文件里的原因。
    expectPresets: [7, 53, 29, 59, 132, 142] },
];

const REQUIRED = ["h1", "sub", "scope-params", "banners", "tiles", "legend", "nodes",
                  "commhead", "commtbl", "tbl", "assumplist", "presets", "eq",
                  "expFmtLab", "precLab", "kvDtLab"];
// 这几个容器里出现 undefined / NaN 就是有字段没接上 —— 页面上看起来只是少了个数字
const CLEAN = ["tiles", "tbl", "commtbl", "assumplist", "banners"];

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
  Object.assign(S, ${JSON.stringify(c.state)}, { concIdx: 0, ctxIdx: ${c.ctxIdx ?? 7}, util: ${c.util ?? 90}, kvDt: kv });
  syncOptions("probe");
  const C = compute();
  globalThis.__probe.push({ kv, bytesPerToken: C.kvBytesPerToken, maxConc: Math.floor(C.maxConc),
                            altConc: Math.floor(C.maxConcAlt),
                            tp: S.tp, dp: S.dp, pp: S.pp, ep: S.ep });
  render();
}
// 预设按钮第三行的路数走 presetCompute(),必须跟当前口径(util / ctxIdx / kvDt)一起变。
// 这里在 kvDt=fp8 的那一轮结束后再切回 bf16,才好与 expectPresets 对账。
Object.assign(S, { kvDt: "bf16" }); syncOptions("probe"); render();
globalThis.__presets = PRESETS.map(p => Math.floor(presetCompute(p).maxConc));
markPreset();
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

  // 预设按钮上的路数(BF16 KV,该用例的 util / ctxIdx 口径下)
  if (c.expectPresets && String(globalThis.__presets) !== String(c.expectPresets))
    fail.push(`预设路数应为 [${c.expectPresets}],实为 [${globalThis.__presets}]`);

  return { fail, probe: globalThis.__probe, presets: globalThis.__presets };
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
      + (c.expectPresets ? `,预设 [${presets}]` : ""));
  }
}
if (bad) { console.error(`\n✗ ${bad}/${CASES.length} 个用例失败`); process.exit(1); }
console.log(`\n✓ ${CASES.length} 个用例全过 · ${REQUIRED.length} 个容器非空 · `
  + `${modelSrcs.length} 个模型(${modelSrcs.map(s => path.basename(s, ".js")).join(", ")})`);
