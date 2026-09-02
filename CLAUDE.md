# LLM 并行切分与显存分布可视化

> **从本目录启动会话**:`cd model-viz && claude`
> 目录 2026-09-02 由 `kimi-k3-parallelism-viz` 改名为 `model-viz` —— 引擎本就是模型无关的,单模型的命名是历史遗留。
> 记忆 2026-09-01 graduate 过一次,但落在**旧目录名**的 project key(`-…-workdir-kimi-k3-parallelism-viz`)上,尚未跟着改名搬过来,所以从本目录启动暂时召不回它。待办。

## 这是什么

交互式工具,回答:**某模型在 N 台某机型上,按给定 TP/DP/PP/EP 切分,显存装不装得下、能收多少并发。**

```
index.html                  模型索引(双击打开这个)
app.html?model=kimi-k3      唯一引擎,公式只有一份实现
data/instances.js           10 个机型目录
data/models/kimi-k3.js      模型定义
```

**数据文件是 `.js` 不是 `.json`** —— `file://` 下 `fetch()` 会被 CORS 拦,classic script 标签不受限制。这样零构建、零服务器、双击即用,也能整个目录打包发给别人。原因写在 ADR-0002 的实施记录里。

加一个模型 = 加一个 `data/models/<id>.js` + 在两个 html 里各加一行 `<script src>`。

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

## 三条容易踩的硬约束

1. **单位是 GiB,不是十进制 GB。** 真值取 `data/instances.js` 里的 `gpuMemMiB`(p5en = `144384 MiB` = 141 GiB)。曾把 141 GiB 当成 `141e9` 字节,低估 7.4%。厂商标称的「GB」在不同 GPU 上时而是 GiB 时而是十进制,**永远不要拿标称数字直接当某一种单位用**。
2. **改完 `app.html` 必须执行验证,不能只截图。** 抽出 `<script>`,连同两个 data 文件在 Node DOM stub 里 eval 一遍,并断言 `tiles`/`nodes`/`tbl` 等容器非空 —— `render()` 中途抛异常时,页面上半部分看着完全正常。
3. **页面可能给客户看。** 两个页面顶部的边界声明块(不回答吞吐/价格、是校验器不是求解器、头号数字建立在猜测的 12 GiB overhead 与假设的 KV dtype 上)必须保留。**声明块里不要写「仅某一项是假设」这类封闭断言** —— 曾因此漏标了敏感度最高的那项(ADR-0007)。
4. **`TP` 的上限是 `nvlinkDomainGpus`,不是每实例卡数。** G 系 8 卡机型没有 NVLink,域 = 1 —— 显存装得下不代表能用。

## 当前状态与未完成项

- ✅ 引擎/数据分层完成,10 个机型可切换,attention family 走语义表
- ✅ 单位 GiB、provenance 标记、边界声明块、dtype 不匹配告警、TP 超 NVLink 域告警
- ✅ **KV dtype 提为界面输入 + 敏感度标记**(2026-09-01,ADR-0007)。默认改为引擎默认的 BF16,推荐配置的头号数字因此从 136 路变 69 路 —— 不是 bug,是把原先隐含的 FP8 假设显式化。反事实(换另一档是多少路)在 banner、边界声明、假设区三处与结论同时出现。
- ⬜ **待做**:把同样的处理套到 `overhead` 上 —— 每卡 12 GiB 仍是不可调的常数,应提为输入并同样显示反事实。
- ⬜ 第二个模型(届时 `FAMILY` 表要加 `gqa`,其 KV 复制因子是 `max(1, TP/n_kv_heads)`,是并行配置的函数;注意这与 KV dtype 是两件事,不要混)
- **剩下的两个软数字**:(1) 每卡 12 GiB 的「激活 + 通信 buffer」,纯猜测,±12 GiB 使并发变动约 20%;(2) KV dtype,现已可切但**本质仍是假设**,BF16↔FP8 使并发变动约 100%。两者的消除办法是同一个,见 `adr/0004` / `adr/0007` —— 在目标机型上起一次服务,读引擎自报的 KV block 数反解。

## 姊妹目录

`../kimi-k3-reference/` 是 2026-08-03 写的 Kimi K3 on AWS 部署指南(中/英/繁三版),同主题的另一面,有自己的 project key。两者记忆未合并。
