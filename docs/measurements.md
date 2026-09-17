# 实测记录

本项目至今**唯一**的 `measured` 一级数据。每条记录必须写全:机型、切分、**引擎版本(带 commit)与
模型 revision**、引擎实际生效的关键配置、完整的 serve 参数、引擎原话、以及**这条实测能推广到哪、
不能推广到哪**。凡是当时没记下、事后无法恢复的标识符,**明写「已丢」,不许含糊过去**。

> 教训(M-001):按可变 tag 拉镜像、`hf download` 不带 `--revision`,事后就只能靠引擎自报的
> 版本号补救。**以后做实测一律 pin 住镜像 digest 与模型 revision。**

口径约定见 [glossary.md](glossary.md) 的 provenance 五级;为什么必须靠实测消除假设见
`adr/0004`、`adr/0007`、`adr/0008`、`adr/0009`。

---

## M-001 · 2026-09-16 · GLM-5.3-Flash / p5en.48xlarge / TP8 —— 每卡非权重非 KV 开销

**动机**:`app.html` 里 `HW.overheadPerGpu` 是一个硬写的 `12 GiB`,四个模型、十个机型、任何 TP
共用同一个常数,而它直接决定最大并发。ADR-0004 / 0007 里反复写的消除办法是「在目标机型上起一次
服务,读引擎自报的数反解」—— 这是第一次真的做。

### 版本与来源(定死,便于复现)

| 项 | 值 | 可复现性 |
|---|---|---|
| 硬件 | AWS `p5en.48xlarge`,8×H200 | 定死 |
| vLLM | **`0.28.1rc1.dev580+g385dce36b`**(git commit `385dce36b`) | **定死** —— 引擎自报,见 `core.py:123` |
| 容器镜像 | tag `vllm/vllm-openai:glm53-flash` | **digest 已丢**,见下 |
| 模型权重 | `zai-org/GLM-5.3-Flash`,`hf download` 未带 `--revision`(引擎自报 `revision=None`) | **未 pin**,见下 |

**镜像 digest 不可恢复。** 当时按 tag 拉取,节点已销毁,而这个 tag 是可变的 —— 现在去查
registry 拿到的是「现在」的 digest,不能证明是 2026-09-16 拉到的那一个。所以镜像层面
**只能靠上面那个 vLLM commit 定位**;它才是决定 allocator / CUDA-graph / 显存记账行为的东西。

**模型 revision 未 pin,但几乎可以确定。** 下载时没写 `--revision`,取的是当时的 `main`。
HF 上该仓库的 `lastModified` 是 **2026-09-07T12:13:47Z**,早于本次实测(09-16),
而当前 `main` 的 sha 是 `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` —— 时间上它就是当时那一份,
**但本次运行没有把它记下来,所以这是推断不是证据**。以后做实测一律带 `--revision`。

### 引擎实际生效的配置

下面这几项直接决定这份显存账,复现时必须一致(全部摘自引擎自报的 `core.py:123` 那行):

| 项 | 值 | 为什么重要 |
|---|---|---|
| `max_num_batched_tokens` | **8192** | 决定 4.04 GiB 的激活峰值 |
| `cudagraph_mode` | `FULL_AND_PIECEWISE`,`max_cudagraph_capture_size = 512` | 决定 1.2 GiB 的 CUDA graph 那一项 |
| `enable_prefix_caching` | `True` | 它使 Mamba cache 走 `'align'` 模式 → attention page 撑到 640 tokens、mamba page padding 20.75% |
| `enable_chunked_prefill` | `True` | 与上面的 token 预算配套 |
| `compilation mode` | `NONE`(`enforce_eager=False`) | 未开 inductor 编译;开了会改激活峰值 |
| `quantization` / `dtype` | `fp8` / `torch.bfloat16` | 权重 38.24 GiB 的前提 |
| `seed` | `0` | — |

### `serve` 参数

单台 p5en.48xlarge(8×H200):

```
/root/.cache/huggingface/GLM-5.3-Flash
--trust-remote-code
--tensor-parallel-size=8
--gpu-memory-utilization=0.90
--max-model-len=131072
--kv-cache-dtype=auto        # = 模型 dtype = BF16
--max-num-seqs=512
--no-enable-flashinfer-autotune
```

### 引擎原话

```
weight_utils.py:895   Checkpoint size: 305.79 GiB
model_runner.py:422   Model loading took 38.24 GiB memory and 226.512052 seconds
gpu_worker.py:637     Available KV cache memory: 81.79 GiB
kv_cache_utils.py:2315  GPU KV cache size: 7,228,559 tokens,
                        Maximum concurrency for 131,072 tokens per request: 55.15x
gpu_worker.py:876     Free memory on device (137.64/139.8 GiB) on startup.
                      Desired GPU memory utilization is (0.9, 125.82 GiB).
                      Actual usage is 39.98 GiB for consumed memory (weights + non-torch),
                        4.04 GiB for peak activation, and 1.2 GiB for CUDAGraph memory.
                      Current kv cache memory in use is 81.79 GiB.
core.py:379           init engine (profile, create kv cache, warmup model) took 467.68 s
```

混合池的分页规则也在日志里:`Mamba cache mode is set to 'align'`、
`Setting attention block size to 640 tokens to ensure that attention page size is >= mamba page size`、
`Padding mamba page size by 20.75%`。

### 读数

| 量 | 实测 | 本页当时 | 差 |
|---|---|---|---|
| checkpoint 总量 | 305.79 GiB | 305.78 | −0.0% |
| 每卡总显存(vLLM 可见) | **139.8 GiB** | 141(API 的 `144384 MiB`) | +0.9% |
| `util 0.90` 预算 | 125.82 GiB | 126.9 | +0.9% |
| 权重/卡 | 38.24 GiB | 38.222 | **−0.05%** |
| 非权重非 KV | **6.98 GiB**(non-torch 1.74 + 激活峰值 4.04 + CUDA graph 1.20) | 12 | **+71.9%** |
| KV 池 | 81.79 GiB | 76.14 | −6.9% |
| 128K 满长并发 | **55.15 路** | 53 | −3.9% |

### 这条实测证明了什么

- **权重侧口径是对的**:38.24 对 38.222,差 0.05%。三桶字节 + 按 TP/PP/EP 切的规则不用动。
- **`12 GiB` 在这一个配置上偏高 72%**,该配置的路数因此偏低 3.9%。把总显存与 overhead 两处
  都换成实测值,引擎算出 KV 池 80.62 GiB(实测 81.79,−1.4%)、56 路(实测 55.15,+1.9%)。

### 这条实测**不能**推广到什么

- **不能把 6.98 GiB 当成别的模型 / 别的 TP 的值。** 它由三项组成,三项的缩放规律都不同:
  non-torch 随 TP 规模与拓扑(NCCL buffer;MoE 还有 DeepEP buffer)、激活峰值随
  `max_num_batched_tokens` × hidden × dtype、CUDA graph 随 capture 形状数与层数。
- **不能推出「本页一律偏保守 N%」。** 同样把 12 换成 6.98,GLM 在 p5en/TP8 上是 53 → 56(5%),
  而 Qwen3.8-27B 在 g7e/TP1 上是 16 → 24(**50%**)—— 小模型上 overhead 占固定开销的比重大得多。
- **不能推出「方向单一、不会高估」。** TP=1 没有跨卡通信 buffer,但激活不分片可能更大;
  DSv4.1 的 `hc_mult = 4` 意味着残差流 4 份副本、激活峰值本该更大。**别处的真实开销可能超过
  12 GiB,那时页面会高估容量。**
- **不能用它反解 KV 每 token 字节数。** `81.79 GiB ÷ 7,228,559 = 12,149 B`,但这是统一 page
  摊平后的混合数(含 KDA recurrent state 与 20.75% 的 padding),不是纯 MLA latent,
  所以既不能证实也不能否证本页的 11,616 B。要拆开得换 `--max-num-seqs` 再跑一次两点法:
  两次 `Available KV cache memory` 的差 ÷ `max-num-seqs` 的差 = 每路 state 的真实字节数,
  截距才是真正的激活 + 通信。

### 由这条实测产生的动作

- `app.html` 的图例与假设区不再声称 overhead「无实测依据」,并把 72% / 5% 的结论**限定在本配置**。
- Qwen3.8-27B 的候选机型撤掉 `g6e.48xlarge`:它的余量(预算 40.23 − FP8 权重 28.75 = 11.5 GiB)
  小于 12 与 6.98 之间的差,红字方向由这个未校准常量单独决定 —— 详见
  `data/models/qwen3.8-27b.js` 的 `candidateInstances` 注释。
- **仍未做**:把 `overhead` 提为界面输入并显示反事实。只有一个实测点,做的时候默认值要按模型给
  (本模型标 `measured`,其余仍标 `guessed`),不能拿这一档去覆盖另外三个。
- **仍未解释**:vLLM 只看到 139.8 GiB,而 `describe-instance-types` 给 `144384 MiB` = 141 GiB、
  `nvidia-smi` 报 `143771 MiB` = 140.401 GiB。三个数都不一样。ADR-0006 取 API 为真值这一条
  在这里第一次被实测打到,差 1.2 GiB;改口径会动所有模型的所有断言,本次未改。

### 原始件

完整启动日志(5,816 行)当时抓在工作机的 `verify-vllm/glm53-p5en-util090.log`,**未入库**
(一次性实验产物)。上面「引擎原话」一节是逐字摘录。

**复现条件,按可靠性排序**:在 p5en 上用 **vLLM commit `385dce36b`**、同一组 serve 参数、
以及上面「引擎实际生效的配置」那张表里的各项,起一次服务,读 `gpu_worker.py:876` 与
`kv_cache_utils.py:2315` 两行即可。**若换了 vLLM 版本,这份账不保证可复现** —— allocator、
CUDA-graph 记账、混合池分页规则都在版本间变过。镜像 digest 与模型 revision 当时没记(见上),
所以严格意义上这条实测**只能近似复现,不能逐字复现**。
