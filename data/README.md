# Data

The experiments use a seed-42 candidate pool of **20,000 examples** (10K Math + 10K Medical) and a shared **2,000-example warmup subset**.

| Artifact | SHA-256 |
|---|---|
| `data/mixed/math10k_med10k_seed42.jsonl` (20K candidate pool; not distributed) | `3c31c43d47065b1b850568588ae17fdd06e7b9f8e7f9f819a812d459b54b1f05` |
| Warmup subset (2K; not distributed) | `ada4cb631860df9039fb4aced93ef56fba2ab92ceb723f993fb70d2ede2a0326` |
| `results/selections/k256_selected.jsonl` (2K; included) | `61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22` |
| `results/selections/k128_selected.jsonl` (2K; included) | `c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2` |

The candidate pool's original source snapshots and exact construction procedure are not provided. The frozen warmup/calibration model is also not provided. Final selected-set fine-tuning is supported by the published data and training scripts; full selection-stage reproduction requires those additional inputs.
