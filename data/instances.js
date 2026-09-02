// AWS GPU 实例目录 —— 显存真值全部取 describe-instance-types 的 MemoryInfo.SizeInMiB。
// 完整 57 个实例的规格见 docs/instance-specs.md;此处只收录被某个模型列为候选的那些。
//
// nvlinkDomainGpus 是 TP 的上限,与 gpusPerInstance 不同:
//   P 系 8 卡机型 → 域 = 8(NVSwitch 全连)
//   G 系 8 卡机型 → 域 = 1(一张 NVLink 都没有,卡间只有 PCIe)
// 详见 docs/adr/0001。
//
// 用 .js 而非 .json:file:// 下 fetch 会被 CORS 拦,classic script 标签不受限制。

REG.instances = {
  // 顺序即下拉框顺序:P 系优先,同系内越新越前;G 系在后。
  "p6-b300.48xlarge": {
    gpu: "B300", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 275040,
    p2p: { type: "nvswitch", gbs: 1800 }, interDomainGbps: 6400, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
  },
  "p6-b200.48xlarge": {
    gpu: "B200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 183359,
    p2p: { type: "nvswitch", gbs: 1800 }, interDomainGbps: 3200, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],   // Blackwell:原生 FP4 + microscaling
  },
  "p5en.48xlarge": {
    gpu: "H200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 144384,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv3",
    nativeDtypes: ["bf16", "fp8"],
  },
  "p5e.48xlarge": {
    gpu: "H200", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 144384,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv2",
    nativeDtypes: ["bf16", "fp8"],
  },
  "p5.48xlarge": {
    gpu: "H100", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 81920,
    p2p: { type: "nvswitch", gbs: 900 }, interDomainGbps: 3200, efaGen: "EFAv2",
    nativeDtypes: ["bf16", "fp8"],    // Hopper:有 FP8,无原生 FP4/microscaling
  },
  "p4de.24xlarge": {
    gpu: "A100 80GB", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 81920,
    p2p: { type: "nvswitch", gbs: 600 }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp16"],
  },
  "p4d.24xlarge": {
    gpu: "A100 40GB", gpusPerInstance: 8, nvlinkDomainGpus: 8, gpuMemMiB: 40960,
    p2p: { type: "nvswitch", gbs: 600 }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp16"],   // Ampere:无 FP8,无 FP4
  },
  // ---- 以下 G 系 8 卡机型没有 NVLink,域 = 1(不是 8)。----
  "g7e.48xlarge": {
    gpu: "RTX PRO 6000", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 98304,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 1600, efaGen: "EFAv4",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
  },
  "g7.48xlarge": {
    gpu: "RTX PRO 4500", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 32768,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 700, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp8", "mxfp4", "nvfp4"],
  },
  "g6e.48xlarge": {
    gpu: "L40S", gpusPerInstance: 8, nvlinkDomainGpus: 1, gpuMemMiB: 45776,
    p2p: { type: "pcie", gbs: null }, interDomainGbps: 400, efaGen: "EFA",
    nativeDtypes: ["bf16", "fp8"],
  },
};
