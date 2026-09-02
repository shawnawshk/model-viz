# ADR-0004：数据可信度（provenance）是一等公民

- 状态：**Accepted**（作为 instance 轴扩展的前置条件）
- 日期：2026-08-31

## 背景

当前单配置实现里有两个软数字：

| 数字 | 值 | 级别 | 误差来源 |
|---|---|---|---|
| 每卡激活 + 通信 buffer | 12 GB | `guessed` | 无依据；且实际随 batch、chunked-prefill token 预算、DeepEP buffer 大小变化 |
| K3 非 expert 参数量 | 77.26e9 | `estimated` | `2.8T − expert 部分` 的残差，两个大数相减，误差可能达几十 B |

单配置下这两项的影响可以口头说明。**一旦扩成 instance × config 的矩阵，每一格都携带同样的未验证估算，而没有任何机制告诉使用者哪一格可信。** 一个自信输出 20 个数字的工具，比一个输出 1 个带误差棒的数字的工具更危险——尤其当输出被用于支撑采购台数。

## 决议

1. 所有进入计算的量必须携带 `confidence` 字段，取值见 `glossary.md` 的五级分级：`spec` / `derived` / `estimated` / `guessed` / `measured`。
2. UI 上可见：至少在表格视图里每行一个级别标记，`guessed` 与 `estimated` 用非颜色通道（图标 + 文字）标注——不能只靠颜色，参见既有的可访问性约束。
3. **敏感度标记（sensitivity flag）**：当一个 `guessed` 量在合理区间内变动会**翻转结论**时，必须显式警告。

## 敏感度标记的具体规则

overhead 的合理区间取 `12–24 GB`（下界是当前猜测，上界考虑大 batch + EP32 的 DeepEP RDMA buffer）。对当前配置：

- 若 `used(overhead=12)` ≤ 预算 且 `used(overhead=24)` > 预算 → 标记为 **「结论依赖猜测值」**，不得显示为干净的「装得下」。
- 「最少实例数」同理：若 N 台在 overhead=12 时可行、在 24 时不可行，则输出应为 `N（不稳）/ N+1（稳）` 而非单一数字。

这条规则的意义在于：它把「我不知道」从脚注提升为**一等输出**。使用者据此决定要不要去实测，而不是拿着一个看起来精确的数字去申请容量。

## overhead 随 instance 变化的问题

instance 轴引入后，`12 GiB` 这个常数变得更可疑：b300 每卡 268.6 GiB、系统内存 4 TB，实际会跑更大的 batch，buffer 也更大。把同一个 12 GiB 套到 32 GiB 的 g7 卡、141 GiB 的 H200 卡和 268.6 GiB 的 B300 卡上，等于假设 overhead 与卡容量无关——这个假设未经验证，且跨度已达 8 倍。

**决议**：暂时保留常数（不发明一个同样没依据的公式），但：
- 该常数标为 `guessed` 且**可在 UI 中调整**，使用者可代入自己实测值；
- 敏感度标记按上述区间始终生效；
- 一旦拿到任一 instance 上的实测值，该 instance 的 overhead 升级为 `measured` 并在 UI 中区别显示。

## 获得 `measured` 的路径（写下来以免遗忘）

在目标实例上起一次服务，读引擎自报的 KV cache 块数，反解真实可用于 KV 的字节数：

```
overhead_real = 物理显存 × util − 权重 − (KV block 数 × 每 block 字节)
```

这一个数能把 `guessed` 变 `measured`，并且顺带校准 `nonExpertParams` 的残差估算。**这是整个模型里投入产出比最高的一次测量。**
