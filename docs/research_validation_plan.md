# Research Validation Plan

This document separates completed evidence from experiments that are required
for stronger scientific claims. No unrun experiment below is presented as a
result.

## Current evidence boundary

Completed/frozen evidence is currently limited to the seed-42 Qwen2.5-7B
Math/Medical setting reported in the repository.

The correction ablation in the README reports **selection overlap with T1**.
It does not establish that asymmetric correction improves downstream utility.
Likewise, the K128 Macro score of 46.6057 is only 0.0057 points above the
operational 46.60 threshold and should not be described as robust utility
preservation without repeated training runs.

## Priority 1 — Downstream correction ablation

### Research question

Does asymmetric correction improve downstream utility relative to raw prefix
truncation under matched data, training, and evaluation conditions?

### Conditions

Run both K=128 and K=256 with:

| Condition | ΔNLL filtering | ΔH ranking |
|---|---|---|
| Raw-Prefix | raw | raw |
| NLL-Only | corrected | raw |
| H-Only | raw | corrected |
| Both | corrected | corrected |
| T1 | full response | full response |

The frozen ShallowFrontier method is H-Only: raw ΔNLL filtering followed by
corrected ΔH ranking.

### Controlled variables

Keep fixed within each matched comparison:

- exact 20K candidate pool and ordering;
- base and calibration checkpoints;
- 2K selection size;
- final full-FT recipe;
- model initialization/training seed for a paired run;
- evaluation code and evaluator hashes;
- hardware/topology when measuring energy.

Alternative ablation selections must be written to a new experiment namespace.
They must not replace the frozen K128/K256 selected artifacts.

### Required outputs

For every condition:

- selected-set SHA-256;
- overlap with T1;
- Math average;
- Medical average;
- Macro;
- scoring processed tokens;
- scoring GPU energy;
- selection wall time;
- final-FT GPU energy and wall time;
- matched E2E energy under the same accounting boundary.

### Primary comparison

For each K, compare Raw-Prefix against H-Only. NLL-Only and Both diagnose
which corrected statistic drives any change.

Selection overlap is a diagnostic, not the primary endpoint.

### Success/failure interpretation

A higher T1 overlap alone is insufficient. Evidence for the correction should
require a consistent downstream advantage or a better performance-energy
trade-off under matched runs. If downstream differences are within observed
seed variability, the correction should be described as improving selection
fidelity rather than proven downstream utility.

## Priority 2 — Multi-seed matched comparison

### Goal

Estimate whether the seed-42 result is stable and quantify uncertainty around
the performance difference relative to T1.

### Design

Use at least 3 and preferably 5 predeclared training seeds. The seed set must
be frozen before looking at new downstream results.

For every seed, run matched:

- T1;
- K256;
- K128.

Report every seed individually plus mean, standard deviation, and confidence
interval. Use paired differences because methods are evaluated under the same
seed and experimental setup.

### Non-inferiority analysis

The operational margin used in the current study is 0.50 Macro points:

```text
difference = Macro(ShallowFrontier) - Macro(T1)
non-inferiority margin = -0.50
```

A non-inferiority claim requires an appropriate confidence interval for the
paired difference whose relevant lower bound lies above -0.50. Failure to
reject a conventional difference test is **not** evidence of equivalence or
non-inferiority.

The present single-seed results cannot support such a statistical claim.

### Resource estimate

Using the measured seed-42 matched E2E values as a rough planning basis:

| Method | Wall time/run | Gross GPU energy/run |
|---|---:|---:|
| T1 | 54.24 min | 0.683 kWh |
| K256 | 52.74 min | 0.644 kWh |
| K128 | 50.67 min | 0.605 kWh |

Five complete matched seeds would therefore be approximately:

- 13.1 sequential wall-hours for the three E2E methods;
- 26.3 GPU-hours on a 2-GPU setup;
- 9.66 kWh gross GPU energy;

excluding official downstream evaluation cost.

Because seed 42 already exists, four additional matched seeds would be roughly
10.5 sequential wall-hours, 21.0 GPU-hours, and 7.73 kWh under the same rough
scaling assumption. These are planning estimates, not measured future costs.

## Priority 3 — Generalization

Run the smallest set of experiments that most strongly tests the mechanism.

### G1. Response-length stratification — low additional compute

Hypothesis: shallow extrapolation should help most when responses exceed K,
while examples with L <= K receive no correction.

Analyze, without retraining if score artifacts are available:

- response-length bins;
- fraction with L <= K and L > K;
- raw-vs-corrected ranking changes by length;
- T1 overlap by length bin.

This directly tests the mechanism and should be prioritized before broad model
sweeps.

### G2. Selection-ratio sensitivity — moderate compute

Test at least one ratio on each side of the current 10% selection ratio while
holding the pool fixed. This tests whether the method depends on selecting
exactly 2K/20K.

### G3. Model-family or model-size transfer — high compute

Choose one additional model condition that changes either model family or
scale, not many combinations at once. The goal is to test whether prefix
ranking fidelity transfers, not to maximize a benchmark table.

Record the full scoring and training energy boundary again; do not transfer
the RTX-5090 energy number to a different hardware/model setup.

## Open research questions

1. **Repeated utility preservation:** unresolved; single-seed evidence only.
2. **Correction vs raw truncation downstream:** unresolved; overlap evidence
   exists, downstream ablation does not.
3. **Overlap vs downstream performance:** unresolved; requires ablation points
   and preferably multiple seeds.
4. **Token reduction to GPU-energy conversion:** partially answered for the
   frozen seed-42 runs, but uncertainty is not quantified.
5. **Model/data generalization:** unresolved.
6. **Utility threshold meaning:** operational tolerance only, not a
   field-standard or statistical threshold.
7. **Contribution relative to InstructDiff:** ShallowFrontier modifies the
   candidate-scoring/selection-cost side through bounded prefix scoring and
   asymmetric ΔH extrapolation. InstructDiff's broader iterative data-selection
   framework and reused code must remain attributed to the upstream work.
