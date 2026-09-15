// AWS GPU 实例目录 —— 显存真值全部取 describe-instance-types 的 MemoryInfo.SizeInMiB。
// 完整 57 个实例的规格见 docs/instance-specs.md;此处只收录被某个模型列为候选的那些。
//
// nvlinkDomainGpus 是 TP 的上限,与 gpusPerInstance 不同:
//   P 系 8 卡机型 → 域 = 8(NVSwitch 全连)
//   G 系 8 卡机型 → 域 = 1(一张 NVLink 都没有,卡间只有 PCIe)
// 详见 docs/adr/0001。
//
// 用 .js 而非 .json:file:// 下 fetch 会被 CORS 拦,classic script 标签不受限制。
//
// hbmGBs / denseTflops(ADR-0010,roofline 的两条线)全部取厂商 datasheet,十进制单位,**只取 dense 行**。
// NVIDIA 的印法不统一:有的 `a | b` 标 sparse | dense,有的只印 sparse 而脚注说 dense 取一半,
// 有的(RTX PRO)什么都不标 —— 后者标 tflopsSparsityUnstated,页面会提示。每条的出处与原文在各机型注释里。
// 注意 datasheet 印的显存(B300 270 GB、B200 180 GB)与 describe-instance-types 的 MiB 对不上,
// 显存真值仍按 ADR-0006 取 API,这里只记录差异,不解释。

REG.instances = {
  // 顺序即下拉框顺序:P 系优先,同系内越新越前;G 系在后。
  "p6-b300.48xlarge": {
    gpu: "B300", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 275040,
    p2p: { type: "nvswitch", gbs: 1800 }, interDomainGbps: 6400, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
    // HGX B300 datasheet(dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf),HGX B300 那一列,2026-09-15 抓取:
    //   "GPU Memory | Bandwidth  270 GB HBM3E | 7.7 TB/s";"FP4 Tensor Core¹  18 PFLOPS | 14 PFLOPS"(脚注 1:Sparse | Dense,
    //   注意 dense 不是一半);"FP8/FP6 Tensor Core²  9 PFLOPS"、"FP16/BF16 Tensor Core²  4.5 PLFOPS"[sic](脚注 2:只印 sparse,dense 取 ½)。
    //   datasheet 没区分 NVFP4 / MXFP4,只写 FP4。
    hbmGBs: 7700, denseTflops: { bf16: 2250, fp8: 4500, fp4: 14000 },
  },
  "p6-b200.48xlarge": {
    gpu: "B200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 183359,
    p2p: { type: "nvswitch", gbs: 1800 }, interDomainGbps: 3200, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],   // Blackwell:原生 FP4 + microscaling
    // HGX B200 datasheet(dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf),HGX B200 那一列,2026-09-15 抓取:
    //   "GPU Memory | Bandwidth  180 GB HBM3E | 7.7 TB/s";"FP4 Tensor Core²  18 PFLOPS"、"FP8/FP6 Tensor Core²  9 PFLOPS"、
    //   "FP16/BF16 Tensor Core²  4.5 PFLOPS"(脚注 2:只印 sparse,dense 取 ½)。FP4 dense 与板级行
    //   "Total NVFP4 Tensor Core  144 | 72 PFLOPS"(sparse | dense)÷ 8 卡 = 9 PFLOPS 互相印证。
    hbmGBs: 7700, denseTflops: { bf16: 2250, fp8: 4500, fp4: 9000 },
  },
  "p5en.48xlarge": {
    gpu: "H200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 144384,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv3",
    nativeDtypes: ["bf16", "fp8"],
    // nvidia.com/en-us/data-center/h200/(H200 SXM 列,页面注明 preliminary),2026-09-15 抓取:
    //   "GPU Memory Bandwidth 4.8TB/s";"BFLOAT16 Tensor Core 1,979 TFLOPS"、"FP8 Tensor Core 3,958 TFLOPS" —— 页面只印 sparse,
    //   dense 取 ½(与 H100 页面的 "* With sparsity" 同一口径)。表里没有 FP4 行。
    hbmGBs: 4800, denseTflops: { bf16: 989.5, fp8: 1979 },
  },
  "p5e.48xlarge": {
    gpu: "H200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 144384,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv2",
    nativeDtypes: ["bf16", "fp8"],
    hbmGBs: 4800, denseTflops: { bf16: 989.5, fp8: 1979 },   // 同 p5en,同一块 H200 SXM,出处见上
  },
  "p5.48xlarge": {
    gpu: "H100", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 81920,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv2",
    nativeDtypes: ["bf16", "fp8"],    // Hopper:有 FP8,无原生 FP4/microscaling
    // nvidia.com/en-us/data-center/h100/(H100 SXM 列),2026-09-15 抓取:
    //   "GPU Memory Bandwidth 3.35TB/s";"BFLOAT16 Tensor Core* 1,979 teraFLOPS"、"FP8 Tensor Core* 3,958 teraFLOPS",
    //   脚注 "* With sparsity",dense 取 ½。没有 FP4 行。
    hbmGBs: 3350, denseTflops: { bf16: 989.5, fp8: 1979 },
  },
  "p4de.24xlarge": {
    gpu: "A100 80GB", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 81920,
    p2p: { type: "nvswitch", gbs: 600 }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp16"],
    // A100 80GB datasheet(nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf),
    //   SXM 列,2026-09-15 抓取:"GPU Memory Bandwidth 2,039 GB/s";"Peak FP16 Tensor Core 312 TF | 624 TF*"(* With sparsity)。
    //   Ampere 没有 FP8 / FP4 行。
    hbmGBs: 2039, denseTflops: { bf16: 312 },
  },
  "p4d.24xlarge": {
    gpu: "A100 40GB", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 40960,
    p2p: { type: "nvswitch", gbs: 600 }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp16"],   // Ampere:无 FP8,无 FP4
    // 同一份 A100 datasheet,40GB 列:"GPU Memory Bandwidth 1,555 GB/s";算力行两档共用(312 | 624*)。
    hbmGBs: 1555, denseTflops: { bf16: 312 },
  },
  // ---- 以下 G 系 8 卡机型没有 NVLink,域 = 1(不是 8)。----
  "g7e.48xlarge": {
    gpu: "RTX PRO 6000", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 98304,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 1600, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
    // RTX PRO 6000 Blackwell Server Edition。datasheet(dam-cdn.nvd.orangelogic.com/AssetLink/707m1632ypg4du1fj3ci1jo3h4w1k78j.pdf):
    //   "Memory bandwidth 1597 GB/s"、"Peak FP4 AI PFLOPS 4 PFLOPS";BF16 / FP8 只在产品页
    //   (nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/):"FP16 | BF16 Tensor Core: 1 PFLOP"、"FP8 Tensor Core: 2 PFLOPS"。
    //   **两处都没写 sparse 还是 dense**,按原文照录;若实为 sparse,算力一行松 2 倍。AWS g7e 页面确认是 Server Edition、96 GB。
    hbmGBs: 1597, denseTflops: { bf16: 1000, fp8: 2000, fp4: 4000 }, tflopsSparsityUnstated: true,
  },
  "g7.48xlarge": {
    gpu: "RTX PRO 4500", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 32768,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 700, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
    // RTX PRO 4500 Blackwell Server Edition datasheet(dam-cdn.nvd.orangelogic.com/AssetLink/x4x3l8i437r0s53bi55omn7y4o2bda4m.pdf),2026-09-15 抓取:
    //   "Memory Bandwidth 800 GB/s"、"FP4 Tensor Core 1.6 PFLOPS"、"FP8 Tensor Core 811 TFLOPS"、"FP16 | BF16 Tensor Core 406 TFLOPS"。
    //   **全文没有 sparse / dense 字样**,按原文照录。AWS g7 页面确认是 Server Edition、32 GB。
    hbmGBs: 800, denseTflops: { bf16: 406, fp8: 811, fp4: 1600 }, tflopsSparsityUnstated: true,
  },
  "g6e.48xlarge": {
    gpu: "L40S", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 45776,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp8"],
    // nvidia.com/en-us/data-center/l40s/,2026-09-15 抓取:"864GB/s";"FP16 Tensor Core 362.05 | 733*"、"FP8 Tensor Core 733 | 1,466*"
    //   (* With Sparsity,左边是 dense)。Ada 没有 FP4。
    hbmGBs: 864, denseTflops: { bf16: 362.05, fp8: 733 },
  },
};
