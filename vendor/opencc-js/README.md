# opencc-js

`cn2t.js` is the UMD build `dist/umd/cn2t.js` of [opencc-js](https://github.com/nk2028/opencc-js)
version **1.4.2** (npm), downloaded from jsDelivr on 2026-10-30. Package license: `MIT AND Apache-2.0` —
the code is MIT (see `LICENSE` in this directory); the conversion dictionaries come from
[OpenCC](https://github.com/BYVoid/OpenCC) under the Apache License 2.0.

It is used only for the Traditional Chinese view: `i18n.js` loads it when the page language is `zh-Hant`
and converts Simplified text with `OpenCC.Converter({ from: "cn", to: "twp" })` (Taiwan phrases),
then fixes six terms that the Taiwan phrase table gets wrong in this tool's context (see `HANT_TERMS` in `i18n.js`).
The file is bundled locally so the page works offline and under `file://`.
