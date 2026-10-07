# ShallowFrontier

**Energy-Efficient LLM Data Selection via Asymmetric Prefix Extrapolation**

ShallowFrontier reduces the hidden compute cost of training-data selection for large language models.

Instead of scoring every candidate over its full response, ShallowFrontier evaluates only a **Prefix-K** and applies asymmetric correction to the entropy signal used by InstructDiff.

<p align="center">
  <img src="figs/overview.png" width="92%">
</p>

## Overview

Data selection can reduce the amount of data used for final fine-tuning, but the selection process itself can be expensive when every candidate is evaluated over its full response.

ShallowFrontier targets this **selection-stage overhead**.

The final pipeline is:

1. Warmup calibration on a fixed 2K subset.
2. Score each candidate with the base and calibration models using only the first **K response tokens**.
3. Apply the original raw ΔNLL q10-q90 bi-directional filter.
4. Correct ΔH using asymmetric prefix extrapolation.
5. Rank by corrected ΔH and select 2K samples.
6. Fine-tune a fresh Qwen2.5-7B using the same full-FT recipe as the InstructDiff T1 reference.

We evaluate two operating points:

- **K=256** — primary configuration
- **K=128** — more aggressive configuration

---

## Main Results

| Method | Math | Medical | Macro |
|---|---:|---:|---:|
| Base | 12.06 | 65.19 | 38.63 |
| Full20K | 29.17 | 57.37 | 43.27 |
| Random2K | 26.66 | 62.96 | 44.81 |
| InstructDiff T1 | 28.62 | 65.59 | **47.10** |
| ShallowFrontier K256 | 27.88 | 65.50 | **46.69** |
| ShallowFrontier K128 | 28.02 | 65.19 | **46.61** |

Both ShallowFrontier configurations satisfy the frozen utility criterion:

**Macro >= 46.60**

This criterion was defined as the T1 reference Macro score
`47.10 - 0.50 = 46.60`, allowing at most a 0.50 Macro-point decrease
(about 1.06% of T1). It is an operational utility-preservation tolerance,
not a statistical-significance threshold or field-wide standard, and was
frozen before the official K256 downstream evaluation.

### Compute and Energy

| Method | Scoring-token reduction vs T1 | E2E GPU energy | Energy reduction vs T1 |
|---|---:|---:|---:|
| InstructDiff T1 | 0% | 0.682982 kWh | 0% |
| ShallowFrontier K256 | **29.14%** | 0.644048 kWh | **5.70%** |
| ShallowFrontier K128 | **46.92%** | 0.605340 kWh | **11.37%** |

E2E energy includes the matched GPU stages:

**Calibration + Candidate Scoring + Final 2K Fine-Tuning**

Official downstream evaluation energy is excluded from this comparison.

<p align="center">
  <img src="figs/performance_energy_pareto.png" width="68%">
</p>

---

## Method

For a signal measured at prefix depth K:

- If response length L <= K, use the observed value at K.
- If L > K, extrapolate from the change between K/2 and K.

For ΔH:

```text
ΔH_hat = ΔH_K                                  if L <= K
ΔH_hat = ΔH_K + (1 - K/L)(ΔH_K - ΔH_K/2)      if L > K
```

The final selector uses:

```text
raw ΔNLL
   ↓
q10-q90 filtering
   ↓
corrected ΔH ranking
   ↓
lowest 2K samples
```

The correction is applied only to ΔH.

### Correction Ablation

| Correction | K128 overlap | K256 overlap |
|---|---:|---:|
| Raw | 62.55% | 79.05% |
| NLL-only | 61.75% | 78.40% |
| **H-only** | **66.05%** | **81.40%** |
| Both | 65.10% | 80.85% |

ΔH-only correction produced the highest selected-set overlap with the full-response T1 reference at both evaluated depths.

---

## Repository Structure

```text
.
├── configs/
│   └── instructdiff_t1_reference.yaml
│
├── scripts/
│   ├── scoring/
│   │   ├── prefix128_score.py
│   │   └── prefix256_score.py
│   ├── selection/
│   │   └── select_from_scores.py
│   ├── training/
│   │   ├── train_k128_fullft.py
│   │   └── train_k256_fullft.py
│   ├── evaluation/
│   ├── energy/
│   └── plots/
│
├── src/
│   └── instdiff/
│
├── eval/
│   ├── math_evaluation/
│   ├── medeval/
│   └── run_all_eval.py
│
├── results/
│   ├── performance/
│   ├── selections/
│   ├── cost/
│   ├── energy/
│   └── manifests/
│
├── figs/
└── docs/
```

---

## Reproduction

Run all commands below from the **repository root**.

### 0. Environment Setup

Core / training environment:

```bash
python -m pip install -r requirements-train.txt
```

Evaluation environment:

```bash
python -m pip install -r requirements-eval.txt
```

The evaluation requirements install the bundled `latex2sympy2`
implementation used by the Math evaluator. For exact reproduction,
use Python 3.10 and CUDA-compatible PyTorch/vLLM builds matching the
versions documented in `docs/reproducibility.md`.

Release-integrity checks can be run without a GPU:

```bash
python -m pip install -r requirements-dev.txt
ruff check --select F821 scripts tests
python -m pytest -q tests
```

The same checks are executed by GitHub Actions on every push and pull
request.

### 1. Base Model

The experiments use:

```text
Qwen2.5-7B
```

By default, the training scripts expect:

```text
models/Qwen2.5-7B
```

A custom location can be provided with:

```bash
export SHALLOWFRONTIER_BASE_MODEL=/path/to/Qwen2.5-7B
```

### 2. Data

The final candidate pool contains 20,000 examples:

- 10,000 Math
- 10,000 Medical

Random seed:

```text
42
```

Large raw datasets and the mixed 20K candidate-pool file are not redistributed in this repository.

To rerun candidate scoring and selection, provide the exact pool at:

```text
data/mixed/math10k_med10k_seed42.jsonl
```

with SHA-256:

```text
3c31c43d47065b1b850568588ae17fdd06e7b9f8e7f9f819a812d459b54b1f05
```

The exact final K256 and K128 selected 2K training sets are included in
`results/selections/`, so final fine-tuning can be reproduced without
reconstructing the 20K pool.

Full selection-stage reproduction additionally requires the base model
and a calibration model produced with the frozen T1 warmup recipe.
The warmup subset used by the study has SHA-256:

```text
ada4cb631860df9039fb4aced93ef56fba2ab92ceb723f993fb70d2ede2a0326
```

See `data/README.md` and `docs/reproducibility.md` for the frozen hashes,
training recipe, and experiment scope.

### 3. Prefix Scoring

Exact scoring implementations are provided for:

- K=128
- K=256

under:

```text
scripts/scoring/
```

### 4. Selection

Selection is performed with:

```text
scripts/selection/select_from_scores.py
```

using raw ΔNLL filtering followed by corrected ΔH ranking.

### 5. Final Full Fine-Tuning

The public training entrypoints intentionally enforce the execution
topology used by the frozen experiment: two visible GPUs, a single
Python process, and the recorded CUDA allocator setting.

K256:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WORLD_SIZE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/training/train_k256_fullft.py
```

K128:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WORLD_SIZE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/training/train_k128_fullft.py
```

`LOCAL_RANK`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` must remain unset.
The scripts also verify the selected-set SHA, row count, TF32 state, and
resolved training arguments before marking a run complete.

The frozen training recipe is:

```text
model_max_length: 2048
per_device_train_batch_size: 1
global_batch_size: 64
epochs: 3
learning_rate: 2e-5
optimizer: adamw_bnb_8bit
bf16: true
gradient_checkpointing: true
scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 1.0
seed: 42
```

### 6. Evaluation

Math benchmarks:

- Math-OAI
- Minerva Math
- OlympiadBench
- AIME24
- AMC23

Medical benchmarks:

- MedQA
- MMLU Medical
- MedMCQA

The aggregate metric is:

```text
Macro = (Math Average + Medical Average) / 2
```

After activating the evaluation environment, run:

```bash
bash scripts/evaluation/eval_k256_official.sh
```

or:

```bash
bash scripts/evaluation/eval_k128_official.sh
```

The official wrappers verify the frozen evaluator files before launching
Math and Medical evaluation. The Math wrapper internally invokes
`python3`, so `python` and `python3` should resolve to the same activated
evaluation environment.

---

## Energy Measurement

GPU energy was measured at 1 Hz using NVML device-energy counters.

Relevant files:

```text
scripts/energy/energy_logger.py
scripts/energy/measure_idle_baseline.py
results/energy/
```

Primary reporting uses **gross GPU energy**.

Idle-adjusted values are retained as secondary measurements.

<p align="center">
  <img src="figs/stagewise_energy.png" width="68%">
</p>

---

## Sustainability Metrics

### QCCR

**Quality-Constrained Carbon Reduction (QCCR)** measures the reduction in operational carbon while satisfying the frozen utility threshold.

For this study:

```text
utility threshold = 46.60
```

Results:

- T1: 0%
- K256: 5.70%
- K128: 11.37%

### SUE

**Sustainable Utility Efficiency (SUE)** normalizes downstream utility gain by operational carbon cost relative to T1.

<p align="center">
  <img src="figs/sue.png" width="64%">
</p>

Within the evaluated configurations, K128 achieved the highest SUE.

---

## Additional Analysis

### Prefix Depth Trade-off

<p align="center">
  <img src="figs/prefix_depth_tradeoff.png" width="64%">
</p>

### CO2 Scale-out Scenario

<p align="center">
  <img src="figs/co2_scaleout_scenario.png" width="64%">
</p>

The scale-out plot is a scenario based on the measured per-run GPU-energy difference and should not be interpreted as a directly observed deployment footprint.

---

## Reproducibility Notes

- Random seed: **42**
- K256 is the primary ShallowFrontier configuration.
- K128 is the predefined secondary, more aggressive operating point.
- The downstream utility criterion **Macro >= 46.60** was frozen before the official K256 evaluation.
- Full20K compute cost reported in the accompanying study is estimated from a partial matched run and is not presented as a completed measured run.
- Reported energy and carbon values refer to **GPU-attributed operational energy**.
- CPU, RAM, storage, networking, cooling, and datacenter PUE are outside the measurement boundary.

---

## Upstream Attribution

ShallowFrontier builds on the InstructDiff framework:

**InstructDiff: Domain-Adaptive Data Selection via Differential Entropy for Efficient LLM Fine-Tuning**

Official repository:

https://github.com/zhuchichi56/Instruct-diff

The upstream project provides the original InstructDiff framework and evaluation components.

ShallowFrontier adds:

- Prefix-limited candidate scoring
- Asymmetric ΔH extrapolation
- K128/K256 operating points
- Matched selection-stage cost measurement
- GPU energy measurement and sustainability analysis

---

## Citation

If you use the upstream InstructDiff framework, please cite the original InstructDiff work.

A project-specific citation for ShallowFrontier will be added with the final paper release.
