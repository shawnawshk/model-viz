// 界面语言(ADR-0015):简体 / 繁体 / English。classic script,file:// 下照样能用。
//
// 判定顺序:?lang= > 上次的选择(localStorage;带 ?lang= 打开也会记住)> 浏览器语言
//   (zh-TW / zh-HK / zh-MO / zh-Hant → 繁体,其余 zh → 简体,其余 → English)。
// 文案只维护简体与英文两份:页面里写 L(简, 英),数据文件里写 key 与 keyEn;
// 繁体由 OpenCC(台湾用语,s2twp)从简体现转,再套一张术语表(HANT_TERMS)。
// OpenCC 的词典有 1.1 MB,只在选了繁体时才加载 —— document.write 在解析阶段同步插入,file:// 下也能用。
(() => {
  "use strict";
  const LANGS = [
    { id: "zh-Hans", label: "简体中文", html: "zh-CN" },
    { id: "zh-Hant", label: "繁體中文", html: "zh-TW" },
    { id: "en",      label: "English",  html: "en" },
  ];
  const ids = LANGS.map(l => l.id);
  const qs = new URLSearchParams(location.search).get("lang");
  let saved = null; try { saved = localStorage.getItem("viz-lang"); } catch (e) { /* file:// 下可能被禁 */ }
  const nav = (typeof navigator !== "undefined" && ((navigator.languages && navigator.languages[0]) || navigator.language)) || "";
  const fromBrowser = /^zh[-_](tw|hk|mo|hant)/i.test(nav) ? "zh-Hant" : /^zh/i.test(nav) ? "zh-Hans" : "en";
  const LANG = ids.includes(qs) ? qs : ids.includes(saved) ? saved : fromBrowser;
  if (ids.includes(qs)) { try { localStorage.setItem("viz-lang", qs); } catch (e) {} }
  document.documentElement.lang = LANGS.find(l => l.id === LANG).html;
  if (LANG === "zh-Hant") document.write('<script src="vendor/opencc-js/cn2t.js"><\/script>');

  // OpenCC 的台湾用语在本工具的语境里有几处是错的。转换前先把这些简体词换成私用区占位符,转完再换成右边的繁体。
  const HANT_TERMS = [
    ["显存", "顯存"],        // OpenCC 给「視訊記憶體」;姊妹项目 kimi-k3-reference 的 zh-TW 版用「顯存」
    ["参数", "參數"],        // OpenCC 给「引數」,那是编程里的 argument
    ["实例", "執行個體"],    // OpenCC 给「例項」;AWS 台湾版文档把 instance 叫「執行個體」
    ["声明", "聲明"],        // OpenCC 给「宣告」,那是编程里的 declaration
    ["复制", "複製"],        // OpenCC 把「被复制了」切成「被複」+「制」
    ["激活", "激活"],        // OpenCC 给「啟用」;「激活峰值」说的是 activation,不是 enable
  ];
  let conv = null;
  const cache = new Map();
  function toHant(s) {
    if (typeof s !== "string" || !s) return s;
    if (!conv) {
      if (!globalThis.OpenCC) return s;   // 词典没加载上:宁可显示简体,也不要整页白屏
      conv = globalThis.OpenCC.Converter({ from: "cn", to: "twp" });
    }
    let r = cache.get(s);
    if (r !== undefined) return r;
    r = s;
    HANT_TERMS.forEach(([a], i) => { r = r.split(a).join(String.fromCharCode(0xE000 + i)); });
    r = conv(r);
    HANT_TERMS.forEach(([, b], i) => { r = r.split(String.fromCharCode(0xE000 + i)).join(b); });
    cache.set(s, r);
    return r;
  }

  // L(简, 英):所有可见文案都过这一道。英文缺了就回落到中文 —— verify 会查英文页面里不许有汉字。
  const L = (zh, en) => LANG === "en" ? (en ?? zh) : LANG === "zh-Hant" ? toHant(zh) : zh;
  // 数据文件里的字段:obj[key] 是简体,obj[key + "En"] 是英文;数组(notes)同理
  const LM = (obj, key) => {
    const v = obj[key], e = obj[key + "En"];
    if (LANG === "en" && e != null) return e;
    if (LANG === "zh-Hant") return Array.isArray(v) ? v.map(toHant) : toHant(v);
    return v;
  };

  // 静态标记:带 data-en 的元素按语言换 innerHTML;data-en-title / data-en-aria-label 换对应属性;
  // <html data-en-doc-title> 是文档标题的英文(不叫 data-en-title,免得和上一条撞名)。
  // 繁体时把这些元素里的简体现转。必须在页面脚本第一次渲染之前调用 —— 之后的内容都由 L() 生成。
  function applyStatic(root = document) {
    document.title = L(document.title, document.documentElement.dataset.enDocTitle);
    if (LANG === "zh-Hans") return;
    for (const el of root.querySelectorAll("[data-en]")) el.innerHTML = L(el.innerHTML, el.dataset.en);
    for (const attr of ["title", "aria-label"])
      for (const el of root.querySelectorAll(`[data-en-${attr}]`))
        el.setAttribute(attr, L(el.getAttribute(attr), el.getAttribute(`data-en-${attr}`)));
  }

  // 语言切换:记住选择,带上 ?lang= 重新载入(这样链接也能直接分享某个语言)
  function bindSwitcher(select) {
    if (!select) return;
    select.innerHTML = LANGS.map(l => `<option value="${l.id}"${l.id === LANG ? " selected" : ""}>${l.label}</option>`).join("");
    select.addEventListener("change", () => {
      try { localStorage.setItem("viz-lang", select.value); } catch (e) {}
      const u = new URL(location.href);
      u.searchParams.set("lang", select.value);
      location.href = u.toString();
    });
  }

  window.I18N = { LANG, LANGS, L, LM, toHant, applyStatic, bindSwitcher };
  window.L = L;
  window.LM = LM;
})();
