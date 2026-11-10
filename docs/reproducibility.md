# Reproducibility

## Setup of our runs

- **Hardware:** two NVIDIA RTX 5090 (32 GB). Training uses both GPUs (per-device batch 1, gradient accumulation to the global batch);
  feature extraction, scoring and noise measurement use one.
- **Software:**
  - training: transformers 4.47.1, bitsandbytes 0.50.2 (`requirements/requirements_train.txt`);
  - evaluation: PyTorch 2.11 (CUDA 13), vLLM 0.26 (`requirements/requirements_eval.txt`).
- **Recipes** (`configs/testbeds/`): full-parameter SFT for 3 epochs with a cosine schedule, 5% warm-up, weight decay 0.01,
  clipping 1.0 and 8-bit AdamW in bf16. The mixed pools use lr 2e-5, global batch 64 and length 2048; the math pool uses
  5e-5 / 256 / 2048 and the medical pool 2e-5 / 64 / 512. Every method on a testbed uses the same recipe.
- **Seeds:** one seed drives the whole pipeline: the calibration subset, calibration training, the noise-measurement order and the
  final training. Rounds or stages never mix seeds.

## What has been checked against our runs

Each check ran the released code (`src/shallowfrontier`) on the saved artifacts of our experiments and compared the outputs byte for byte.

| Check | Result |
|---|---|
| Selections from saved scores, groups and noise (5 testbeds × seeds 42, 43, 44) | 15 / 15 identical |
| Calibration subsets of every calibration run | 17 / 17 identical |
| Mixed pools from InstructDiff's source files (`scripts/make_mixed_pools.py`) | 2 / 2 identical |
| Groups from saved prompt features (5 testbeds) | 5 / 5 identical |
| Sign convention of ΔH and ΔNLL against InstructDiff's scored.jsonl | identical (base − calibrated) |

**Not yet checked on GPU with this code:** the noise measurement and the scorer. They are ported with the same seeds, sketch indices and
row order, and their decision rules are unit-tested. The scorer computes in fp32, while InstructDiff's computes in fp16, so scores can
differ in the last digits. Our runs used InstructDiff's round-1 scores.

## Energy accounting (end to end)

- **Definition:** E2E energy = selection + final fine-tuning, as GPU energy from the NVML total-energy counters of the GPUs in use.
  Evaluation is excluded. Values are the mean over the measured runs.
- **InstructDiff:** per-round records of the official pipeline (calibration, scoring, fine-tuning), cumulative over rounds.
  - Base-model scoring is counted once; rounds 2–3 add the calibrated-model pass and that round's fine-tuning.
  - Its best round is only known after all three rounds, so that cost is the T3 row.
- **ShallowFrontier:** its own measured calibration and final fine-tuning. Scoring is priced at the same per-pass rate as InstructDiff's
  pipeline (base pass, plus the calibrated pass where a group takes the InstructDiff route). Noise measurement is its run time × the
  measured mean power of single-GPU scoring (0.60 kW).
- **Compute units:** with F = one forward pass over the pool, the final fine-tune costs 0.9F for every method. InstructDiff's three rounds
  cost 7.6F, and ShallowFrontier averaged 3.6F over the four testbeds, the same 53% reduction as the measured energy.

## Known differences from the InstructDiff paper

- **Math:** our reproduced InstructDiff scores are 1.7–3.5 points below the published ones, and full-data training is stronger than
  reported.
- **Comparison:** all methods share one recipe and one implementation here, so the comparison is internal. The published runs'
  environment is not available.
