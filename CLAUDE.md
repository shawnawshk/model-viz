# LLM 并行切分与显存分布可视化

> **从本目录启动会话**:`cd model-viz && claude`
> 目录 2026-09-02 由 `kimi-k3-parallelism-viz` 改名为 `model-viz` —— 引擎本就是模型无关的,单模型的命名是历史遗留。
> 记忆 2026-09-01 graduate 过一次,但落在**旧目录名**的 project key(`-…-workdir-kimi-k3-parallelism-viz`)上,尚未跟着改名搬过来,所以从本目录启动暂时召不回它。待办。

## 这是什么

交互式工具,回答:**某模型在 N 台某机型上,按给定 TP/DP/PP/EP 切分,显存装不装得下、能收多少并发。**

```
index.html                     模型索引(双击打开这个)
app.html?model=kimi-k3         唯一引擎,公式只有一份实现
data/instances.js              10 个机型目录
data/models/kimi-k3.js         模型定义
data/models/glm-5.3-flash.js   第二个模型(2026-09-09)
```

**数据文件是 `.js` 不是 `.json`** —— `file://` 下 `fetch()` 会被 CORS 拦,classic script 标签不受限制。这样零构建、零服务器、双击即用,也能整个目录打包发给别人。原因写在 ADR-0002 的实施记录里。

加一个模型 = 加一个 `data/models/<id>.js` + 在两个 html 里各加一行 `<script src>` + 在 `verify-app.js` 的用例表里加一条。契约见 `glm-5.3-flash.js`(比 K3 那份新,字段最全):`weights` 是两桶实测字节、`notes` 放该模型专属的一手件溯源、`defaultParallel` 是起始切分、`hidden` 给 PP 的激活量。引擎加载时会硬断言 `weights` 的两条闭合关系与 `TP×DP×PP = 卡数`。

**预设名字里不要写绝对路数,也不要写「推荐 / 最优」** —— 路数由 `markPreset()` 按当前口径现算,名字只描述结构或相对关系。理由见 ADR-0008 §9。

## 读之前先看 docs/

设计已收敛,**决策都在 `docs/` 里,不要重新讨论**:

| 文件 | 内容 |
|---|---|
| `docs/glossary.md` | 领域词汇。NVLink 域 ≠ 实例、推理 DP ≠ 训练 DP、KV 复制因子按 attention family 而异、单位口径、provenance 五级 |
| `docs/instance-specs.md` | **规格总目录**:G5–G7 / P4d–P6 共 57 个实例,以 `describe-instance-types` 的 MiB 为真值。任何实例规格从这里查,**不要重新拉 API** |
| `docs/adr/0001` | TP 上限由 NVLink 域决定(不是「节点」);两字段建模 |
| `docs/adr/0002` | 导航分页 + 单引擎 + data 契约 |
| `docs/adr/0003` | **只算显存**,不引入成本/吞吐/FLOPS;instance 轴先于 model 轴 |
| `docs/adr/0004` | provenance 一等公民 + 敏感度标记 |
| `docs/adr/0005` | **校验器不是求解器**(不做配置搜索);instance 候选是 per-model 配置 |
| `docs/adr/0006` | 单位一律 GiB,真值取 MiB |
| `docs/adr/0007` | **KV cache 的 dtype 是引擎启动参数,不是模型属性** —— 提为界面输入,默认取引擎默认 BF16;敏感度是 overhead 猜测的 5 倍 |
| `docs/adr/0008` | **权重改为「两桶实测字节」**(非 expert 可以是混合 dtype);参数量排除 `*_scale_inv` 使「每参数字节数」成为格式指纹;新增 `dsa` family(**稀疏不省显存**);K3 字面量抽成 `notes` / `defaultParallel` / `hidden` |

## 三条容易踩的硬约束

1. **单位是 GiB,不是十进制 GB。** 真值取 `data/instances.js` 里的 `gpuMemMiB`(p5en = `144384 MiB` = 141 GiB)。曾把 141 GiB 当成 `141e9` 字节,低估 7.4%。厂商标称的「GB」在不同 GPU 上时而是 GiB 时而是十进制,**永远不要拿标称数字直接当某一种单位用**。
2. **改完 `app.html` 必须执行 `node verify-app.js`,不能只截图。** 它把 `<script>` 连同 data 文件在 Node DOM stub 里 eval 一遍,断言容器非空 —— `render()` 中途抛异常时,页面上半部分看着完全正常。**用例表里每个模型至少一条,且必须保留那条 `PP > 1` 的**:PP 分支只在 PP>1 时渲染,曾因探针从不设 PP 而让通信表里的一个 `NaN` 活了两个月。加模型就加用例,期望值先手算再与引擎对账。
3. **页面可能给客户看。** 两个页面顶部的边界声明块(不回答吞吐/价格、是校验器不是求解器、头号数字建立在猜测的 12 GiB overhead 与假设的 KV dtype 上)必须保留。**声明块里不要写「仅某一项是假设」这类封闭断言** —— 曾因此漏标了敏感度最高的那项(ADR-0007)。
4. **`TP` 的上限是 `nvlinkDomainGpus`,不是每实例卡数。** G 系 8 卡机型没有 NVLink,域 = 1 —— 显存装得下不代表能用。

## 当前状态与未完成项

- ✅ 引擎/数据分层完成,10 个机型可切换,attention family 走语义表
- ✅ 单位 GiB、provenance 标记、边界声明块、dtype 不匹配告警、TP 超 NVLink 域告警
- ✅ **KV dtype 提为界面输入 + 敏感度标记**(2026-09-01,ADR-0007)。默认改为引擎默认的 BF16,推荐配置的头号数字因此从 136 路变 69 路 —— 不是 bug,是把原先隐含的 FP8 假设显式化。反事实(换另一档是多少路)在 banner、边界声明、假设区三处与结论同时出现。
- ✅ **第二个模型 GLM-5.3-Flash**(2026-09-09,ADR-0008)。它逼出了四件事:权重改成两桶实测字节(它的非 expert 是 BF16/FP8 混合,旧的残差口径会把 27.4 亿 FP8 参数误并进 expert 桶);新增 `dsa` family;引擎里剩下的 K3 字面量抽成 `notes`/`defaultParallel`/`hidden`;边界声明块不再声明假设的**数量**。K3 的数字一个没变。
  - **两个模型的切分权衡方向相反**,这是页面最值得讲的一点:K3 非 expert 有 106.5 GiB,DP 复制它很贵;GLM 只有 15.5 GiB,复制几乎免费,而 TP 会把 KV latent 复制 TP 份 → 同一台 p5en 上 **128K 时** TP1×DP8 是 325 路、TP8/DP1 只有 53 路。但这个 6 倍随口径缩水到 1.11×(1K 上下文),那时 TP2×DP4 还会反超两者。
- ✅ **预设路数改为按当前口径现算**(2026-09-09,ADR-0008 §9)。**任何路数都不许脱离口径出现** —— 曾把「325 路」写进预设名字,而点预设时 util / 上下文 / dtype 一律不重置,于是按钮标 325、页面显示 44,两个数同屏打脸。现在第三行由 `markPreset()` 现算并标 `@128K/0.90`,拖滑块时六个预设整组联动。**也因此撤掉了「推荐 / 折中」这类标签**:排序本身随口径变,标它就是越界(ADR-0005)。
- ⬜ **待做**:把同样的处理套到 `overhead` 上 —— 每卡 12 GiB 仍是不可调的常数,应提为输入并同样显示反事实。
- ⬜ `gqa` family(届时 KV 复制因子是 `max(1, TP/n_kv_heads)`,是并行配置的函数,`perTokenElems(g)` 的签名不够用;注意这与 KV dtype、与 ADR-0008 新增的 `perTokenFixedBytes` 都是不同的事,不要混)
- **剩下的软数字**:(1) 每卡 12 GiB 的「激活 + 通信 buffer」,纯猜测,±12 GiB 使并发变动约 20%;(2) KV dtype,现已可切但**本质仍是假设**,BF16↔FP8 使并发变动约 100%;(3) 仅 GLM:DSA indexer key cache 的池化方式与 dtype,按 pooled 32 元素/FP8 计,不池化则是 4 倍,使并发变动约 10%。三者的消除办法是同一个,见 `adr/0004` / `adr/0007` / `adr/0008` —— 在目标机型上起一次服务,读引擎自报的 KV block 数反解。**不要在声明块里写「只有 N 项是假设」** —— 这个清单每加一个模型就可能变长。

## 姊妹目录

`../kimi-k3-reference/` 是 2026-08-03 写的 Kimi K3 on AWS 部署指南(中/英/繁三版),同主题的另一面,有自己的 project key。两者记忆未合并。
