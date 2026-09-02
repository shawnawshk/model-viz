# ADR-0007：KV cache 的 dtype 是输入，不是模型属性

- 状态：**Accepted → 已在 `app.html` + `data/models/kimi-k3.js` 实施**（2026-09-01）
- 影响：头号数字（最大并发）、顶部边界声明块、data 契约新增 `kvDtypes` 字段
- 相关：[ADR-0004](0004-provenance-first-class.md)（敏感度标记）、[ADR-0005](0005-validator-not-solver-and-instance-list.md)（同类的 `expertFormats` 设计）

## 触发原因：一个没有出处、却能把头号数字翻倍的常数

`data/models/kimi-k3.js` 里原本写着 `kvDtypeBytes: 1`，即假设 MLA 的 latent KV cache 按 **FP8** 存。

这个数字**无出处**。核对过一手件：

- `config.json` 里没有任何 KV cache dtype 字段（`text_config` 的 `dtype: "bfloat16"` 是模型 dtype，不是 KV cache 的）
- checkpoint 里也没有 —— KV cache 是运行时分配的，不在权重里
- 它由**引擎启动参数**决定：SGLang / vLLM 的 `--kv-cache-dtype`，默认 `auto` = 模型 dtype = BF16；FP8 要显式传 `fp8_e4m3`

也就是说：这项信息在模型侧根本不存在，工具无论怎么查 config 都推不出来。把它写成数据文件里的常数，等于把一个部署决策伪装成模型事实。

同时，页面顶部的边界声明块当时写的是：

> 模型参数已全部落实为推导值……**仅** recurrent state 的 dtype 仍是假设

这句话有两重错：**「仅」是事实错误**；而且它标错了对象 —— 被标出来的那项（state dtype）几乎不敏感，没被标的那项（KV dtype）能把结论翻倍。这正是 ADR-0004 定义的最坏失效模式：**给出一个干净的数字，而它建立在一个未声明的假设上。**

## 量级：为什么这条必须是输入而不是常数

配置：4 × p5en.48xlarge = 32 卡，`TP8/DP4/PP1/EP32`，util 0.90，128K 上下文，非 expert BF16，expert MXFP4（原生）。

| KV dtype | 每 token/份 | 每路请求/卡 | 128K 最大并发 |
|---|---|---|---|
| **BF16**（引擎默认，现为本工具默认） | 27,648 B | 919.5 MB | **69 路** |
| **FP8**（需显式开启，原先的隐含假设） | 13,824 B | 466.6 MB | **136 路** |

与本工具其它两个软数字的敏感度对比（同一基准，单项变动）：

| 变动 | 最大并发 | 相对基准 |
|---|---|---|
| KV dtype BF16 → FP8 | 69 → **136** | **+97%** |
| overhead 12 → 24 GiB/卡 | 69 → 55 | −20% |
| state dtype FP32 → BF16 | 69 → 69 | 0%（<2%，被取整吃掉） |

**KV dtype 的敏感度是那个「唯一 guessed 项」的约 5 倍。** 一个能把选型结论翻倍的量，不能藏在数据文件里。

## 决议

1. **KV dtype 从数据文件常数提为界面输入。** 新增顶层 `kvDtypes: [{ id, label, bytes }]`，与 `expertFormats` / `nonExpertFormats` 同一模式。
2. **默认取引擎默认（BF16），不取乐观档。** 不加任何启动参数就是这一档；工具的用途是配置自查，默认值不该讨好使用者。想看 FP8 的账，一次点击。
3. **反事实必须与结论同时出现**（ADR-0004 的敏感度标记，落到三处）：
   - 一条常驻 banner，两个方向都说：当前档 + 换另一档的并发是多少 + 差几倍
   - 顶部边界声明块改为明确列出两个 dtype 假设，并给出 KV 档切换后的并发数
   - 假设区的 KV 条目拆成「元素数是推导值 / dtype 不是」两句
4. **provenance 从 `推导` 降为 `估算`。** 元素数（`kv_lora_rank + qk_rope_head_dim`）确实是 config 推出的，但显示出来的**字节数**取决于一个假设，所以整项按 `估算` 标记，表格的「依据」列 tooltip 写明当前 dtype。
5. **`FAMILY` 表返回元素数而非字节。** `perTokenBytes(g)` → `perTokenElems(g)`，dtype 在 `compute()` 里统一乘上。这样加 GQA 等新 family 时不必各自处理 dtype。
6. **data 契约：每个模型必须提供 `kvDtypes`，数组第一项即默认值。**

## 为什么不是「保持 FP8 默认，只加个开关」

那样能保住页面上现有的数字（136 路而不是 69 路），代价是把一个需要显式开启、且有精度影响的部署选择继续当作基线。本 ADR 的整个动机就是「不要把假设当事实」，默认值本身若还是那个乐观假设，改动就只剩装饰意义。

## 实施后果

- **推荐配置的头号数字从 136 路变为 69 路。** 这不是修 bug，是把此前隐含的 FP8 假设显式化后回到默认口径。
- `kimi-k3.js` 的预设注释已更新（b200 四台 vs p5en 四台：**109 vs 69**；FP8 KV 下是 215 vs 136）。
- **[ADR-0005](0005-validator-not-solver-and-instance-list.md) 的「K3 的实际差距」表里那个 `136`，是在 FP8 KV 的隐含假设下算的。** ADR 是决策时点的记录，按惯例不回改；在此标注，读那张表时把它当作「FP8 KV 口径」。该表要比较的是 expert 格式之间的差距，结论方向不受影响。
- `quantExcludes` 顺带补上了 `lm_head`（与 `config.json` 的 `quantization_config.ignore` 逐条对齐；此前漏列。只影响描述文本，字节数一直含在 57.19B 里）。

## 仍未解决

- **`overhead` 还是常数。** 每卡 12 GiB 仍是 `guessed`，且仍不可调 —— 按同样的理由它也该提成输入，见 CLAUDE.md 的待做项。
- **真正消除这两项的办法只有一个**：在目标机型上起一次服务，读引擎自报的 KV block 数反解真实 overhead 与真实 KV 单价，把 provenance 提到 `measured`（ADR-0004）。届时 `kvDtypes` 的选择会退化为「记录当时用了哪档」。
- **GQA family 落地时**这条同样适用，但那时 KV 复制因子还额外依赖 `n_kv_heads` 与 TP 的关系（见 `glossary.md`），是并行配置的函数，两者不要混在一起处理。
