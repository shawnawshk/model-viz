// app.html 的脚本执行验证。CLAUDE.md 硬约束 #2:改完 app.html 必须在 DOM stub 下 eval 一遍,
// 并断言各容器非空 —— render() 中途抛异常时页面上半部分看着完全正常。
const fs = require("fs"), path = require("path");
const ROOT = process.argv[2] || __dirname;

const html = fs.readFileSync(path.join(ROOT, "app.html"), "utf8");
const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const srcs = [...html.matchAll(/<script[^>]*\bsrc="([^"]+)"/g)].map(m => m[1]);
if (inline.length !== 2 || srcs.length !== 2) throw new Error(`script 标签数变了:inline=${inline.length} src=${srcs.length}`);

// ---- DOM stub ----
const els = new Map();
const mkEl = id => ({
  id, innerHTML: "", textContent: "", value: "", title: "", max: "",
  style: {}, dataset: {}, classList: { add() {}, remove() {} },
  setAttribute() {}, getAttribute: () => null, addEventListener() {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
  closest: () => null, querySelectorAll: () => [],
});
const doc = {
  getElementById(id) { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); },
  querySelectorAll: () => [], addEventListener() {}, body: mkEl("body"),
  documentElement: { dataset: {} },
};
global.document = doc;
global.location = { search: "?model=kimi-k3" };
global.localStorage = { getItem: () => null, setItem() {} };
global.matchMedia = () => ({ matches: false });
global.innerWidth = 1600;
global.window = global;

// ---- 按 html 里的顺序执行:inline[0] → data 文件 → inline[1](主脚本)----
// 数值探针与主脚本一起 eval —— S / compute / render 都是主脚本内部的 const,外面拿不到
const PROBE = `
globalThis.__probe = [];
for (const kv of ["bf16", "fp8"]) {
  Object.assign(S, { instId: "p5en.48xlarge", n: 4, tp: 8, dp: 4, pp: 1, ep: 32,
                     concIdx: 0, ctxIdx: 7, util: 90, neFmt: "bf16", expFmt: "mxfp4", kvDt: kv });
  syncOptions("probe");
  const C = compute();
  globalThis.__probe.push({ kv, bytesPerToken: C.kvBytesPerToken, maxConc: Math.floor(C.maxConc),
                            altConc: Math.floor(C.maxConcAlt) });
  render();
}
`;

eval(inline[0]);
for (const s of srcs) eval(fs.readFileSync(path.join(ROOT, s), "utf8"));
eval(inline[1] + "\n" + PROBE);

// ---- 断言:每个容器都必须被 render() 真正填过 ----
const REQUIRED = ["h1", "sub", "scope-params", "banners", "tiles", "legend", "nodes",
                  "commtbl", "tbl", "assumplist", "presets", "eq", "expFmtLab", "precLab", "kvDtLab"];
const empty = REQUIRED.filter(id => {
  const e = els.get(id);
  return !e || (!String(e.innerHTML).trim() && !String(e.textContent).trim());
});
if (empty.length) { console.error("✗ 空容器:", empty.join(", ")); process.exit(1); }
if (String(els.get("tbl").innerHTML).includes("undefined")
 || String(els.get("tiles").innerHTML).includes("NaN")) {
  console.error("✗ tiles/tbl 里出现了 undefined / NaN"); process.exit(1);
}

// ---- 数值断言:两档 KV dtype 都要能算,反事实互为对方 ----
const out = globalThis.__probe;
const [b, f] = out;
console.table(out);
const ok = b.bytesPerToken === 27648 && f.bytesPerToken === 13824
        && b.maxConc === 69 && f.maxConc === 136
        && b.altConc === f.maxConc && f.altConc === b.maxConc;
if (!ok) { console.error("✗ 数值不符合预期(应为 BF16 69 路 / FP8 136 路,且反事实互为对方)"); process.exit(1); }
console.log(`✓ 全部 ${REQUIRED.length} 个容器非空,两档 KV dtype 均可计算,反事实互相对称`);
