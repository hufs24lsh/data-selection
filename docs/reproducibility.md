# Reproducibility

## Experimental Scope

ShallowFrontier was evaluated as a modification of the candidate-scoring stage of InstructDiff.

All final comparisons use:

- Base model: Qwen2.5-7B
- Candidate pool: 20,000 examples
- Selected training set: 2,000 examples
- Random seed: 42
- Final fine-tuning: full-parameter SFT
- Prefix configurations: K=256 and K=128

K256 is the primary configuration. K128 is the predefined secondary, more aggressive configuration.

## Training Recipe

The final 2K fine-tuning recipe is matched to the InstructDiff T1 reference:

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

TF32 was not enabled.

## Training Environment

Final training environment:

```text
Python         3.10.21
PyTorch        2.13.0+cu132
Transformers   4.47.1
bitsandbytes   0.50.2
```

Hardware:

```text
2 x NVIDIA RTX 5090
~31.36 GiB VRAM per GPU
```

## Evaluation Environment

Official downstream evaluation used a separate environment:

```text
Python         3.10.21
vLLM           0.26.0
PyTorch        2.11.0+cu130
Transformers   5.17.0
```

## Evaluation Benchmarks

Math average:

- Math-OAI
- Minerva Math
- OlympiadBench
- AIME24
- AMC23

Medical average:

- MedQA
- MMLU Medical
- MedMCQA

Overall utility:

```text
Macro = (Math Average + Medical Average) / 2
```

The frozen utility acceptance criterion was:

```text
Macro >= 46.60
```

## Scoring Cost

Full-response InstructDiff T1 processed:

```text
14,510,380 scoring tokens
```

ShallowFrontier:

```text
K256   10,282,340 tokens   29.14% reduction
K128    7,702,010 tokens   46.92% reduction
```

These values refer specifically to candidate-selection scoring, not total training tokens.

## GPU Energy Accounting

Primary energy reporting uses gross GPU energy.

Matched E2E scope:

```text
Calibration
+ Candidate Scoring
+ Final 2K Fine-Tuning
```

Official downstream evaluation is excluded.

Measured gross GPU energy:

| Method | GPU Energy |
|---|---:|
| InstructDiff T1 | 0.682982 kWh |
| ShallowFrontier K256 | 0.644048 kWh |
| ShallowFrontier K128 | 0.605340 kWh |

Relative to T1:

```text
K256   5.70% reduction
K128  11.37% reduction
```

The measurement boundary is GPU-attributed operational energy. CPU, RAM, storage, networking, cooling, and datacenter PUE are not included.

## Full20K Cost

The Full20K end-to-end cost was not measured to completion.

Any Full20K energy or runtime value reported in the accompanying study is explicitly marked as an estimate derived from a partial matched run.

## Result Integrity

Exact selected-set hashes and frozen experiment manifests are provided in:

```text
results/selections/
results/manifests/
```

Final official downstream summaries are provided in:

```text
results/performance/
```

## Historical Manifests

Files under `results/manifests/` are preserved as experiment provenance.
They intentionally retain historical internal paths and experiment names
because those strings are part of the frozen hash records from the original
runs. They are not runtime paths required by the public release.
