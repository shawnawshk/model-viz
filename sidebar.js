// Shared provider navigation and sidebar preference. Classic script keeps file:// support.
(() => {
  "use strict";
  const providers = [
    { id: "moonshot", namespace: "moonshotai", name: "Moonshot AI" },
    { id: "zai", namespace: "zai-org", name: "Z.ai" },
    { id: "deepseek", namespace: "deepseek-ai", name: "DeepSeek" },
    { id: "qwen", namespace: "Qwen", name: "Qwen" },
  ];
  const currentModel = new URLSearchParams(location.search).get("model")
    || (document.body.className === "analysis" ? "kimi-k3" : null);
  const modelLogo = document.getElementById("model-logo");
  if (modelLogo && currentModel) {
    const namespace = REG.models[currentModel].hf.split("/")[0];
    const provider = providers.find(provider => provider.namespace === namespace);
    modelLogo.src = `assets/providers/${provider.id}.svg`;
  }
  const navigation = document.getElementById("model-nav");
  navigation.innerHTML = providers.map(provider => {
    const models = Object.entries(REG.models).filter(([, model]) =>
      model.hf.split("/")[0] === provider.namespace);
    if (!models.length) return "";
    const current = models.some(([id]) => id === currentModel);
    return `<details class="provider-group" name="model-provider" data-provider="${provider.id}" data-current="${current}"${current ? " open" : ""}>
      <summary class="provider-trigger" aria-label="${provider.name}" title="${provider.name}">
        <span class="provider-logo"><img src="assets/providers/${provider.id}.svg" alt="" width="18" height="18"></span>
        <span class="nav-text">${provider.name}</span>
        <svg class="provider-chevron" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m7 4 6 6-6 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </summary>
      <div class="provider-models">${models.map(([id, model]) =>
        `<a class="nav-link" href="app.html?model=${id}" aria-label="${model.name}" title="${model.name}"${id === currentModel ? ' aria-current="page"' : ""}>
          <svg class="model-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m10 3 7 4-7 4-7-4 7-4zM3 11l7 4 7-4M3 15l7 4 7-4" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>
          <span>${model.name}</span>
        </a>`).join("")}</div>
    </details>`;
  }).join("");

  const toggle = document.getElementById("sidebar-toggle");
  function setSidebar(collapsed) {
    document.documentElement.dataset.sidebar = collapsed ? "collapsed" : "expanded";
    const label = collapsed ? "展开侧边栏" : "收起侧边栏";
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
  }
  let saved = null;
  try { saved = localStorage.getItem("viz-sidebar"); } catch (e) {}
  setSidebar(saved === "collapsed");
  function saveSidebar(collapsed) {
    setSidebar(collapsed);
    try { localStorage.setItem("viz-sidebar", collapsed ? "collapsed" : "expanded"); } catch (e) {}
  }
  toggle.addEventListener("click", () => {
    saveSidebar(document.documentElement.dataset.sidebar !== "collapsed");
  });
  navigation.addEventListener("click", event => {
    const summary = event.target.closest(".provider-trigger");
    if (!summary || document.documentElement.dataset.sidebar !== "collapsed") return;
    event.preventDefault();
    saveSidebar(false);
    summary.parentElement.open = true;
  });
})();
