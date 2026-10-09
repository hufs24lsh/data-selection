# Reproducibility

## Experimental Setup

| Setting | Value |
|---|---|
| Base model | Qwen2.5-7B |
| Candidate pool | 20,000 (10K Math + 10K Medical) |
| Selected training set | 2,000 |
| Seed | 42 |
| Prefix depths | K=256 (primary), K=128 |
| Training | Full-parameter SFT, 2 × RTX 5090 |
| Utility | Macro = (Math average + Medical average) / 2 |

### Final fine-tuning

The matched T1/K256/K128 training configuration:

```text
model_max_length              2048
per_device_train_batch_size   1
global_batch_size             64
num_train_epochs              3
learning_rate                 2e-5
optimizer                     adamw_bnb_8bit
bf16                          true
gradient_checkpointing        true
lr_scheduler_type             cosine
warmup_ratio                  0.05
weight_decay                  0.01
max_grad_norm                 1.0
seed                          42
```

TF32 was disabled. The training environment was Python 3.10.21, PyTorch 2.13.0+cu132, Transformers 4.47.1 and bitsandbytes 0.50.2. The evaluation environment was Python 3.10.21, vLLM 0.26.0, PyTorch 2.11.0+cu130 and Transformers 5.17.0. Use `scripts/repro/check_environment.py` to check the recorded versions. A complete transitive dependency lock is not provided.

## Data and Results

- Selected 2K training sets: `results/selections/` (published with SHA-256 values in [Data](../data/README.md)).
- Performance summaries: `results/performance/`.
- Scoring cost and energy: `results/cost/` and `results/energy/`.
- Selection and scoring implementations: `scripts/selection/` and `scripts/scoring/`.

Original pre-registration and cost manifests are preserved unchanged in `results/manifests/archive/`. They record the experiment design at the time of registration, not current completion status.

The selected sets support final fine-tuning with the public training scripts and base model. Full candidate scoring requires the exact 20K pool and matching warmup/calibration model. The pool, model checkpoint and complete upstream source-data reconstruction inputs are not distributed; raw-data-to-results reproduction is not available from this repository alone.

## Energy Accounting

Gross GPU energy uses NVML cumulative energy counters sampled at 1 Hz, summed over calibration, candidate scoring and final 2K training. Stage aggregation is implemented in `scripts/energy/aggregate_energy.py`. Downstream evaluation, non-GPU components and datacenter PUE are outside the reported boundary. The original raw NVML counter stream is not distributed, so historical per-sample energy reconstruction is not possible with public files alone.

The Full20K energy comparison is an **estimate from a partial run**, not a completed energy measurement. Operational CO₂eq values apply the 2023 Korean electricity consumption-end factor recorded in `results/energy/carbon_factor_status.json`; they are modeled values, not direct emissions measurements. The unchanged measurement protocol records the measurement-time verification rules; original logger metadata is retained in `results/energy/archive/logger_meta.json`.

## Interpretation

The reported downstream comparison uses seed 42 only. The criterion `Macro >= 46.60` is an operational 0.50-point tolerance relative to T1 (47.10), not evidence of statistical non-inferiority. The correction-ablation table in the README measures selection overlap, not a causal downstream performance improvement.
