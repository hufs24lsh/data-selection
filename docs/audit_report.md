# ShallowFrontier Critical Audit Report

Audit target: public repository `hufs24lsh/data-selection`.

Audit base: `main` at
`a55f32fe851fc6e74176a318027b371b4a3c97b7`.

Audit work is isolated on `audit/reproducibility-hardening`. Frozen
manifests, published selected datasets, and frozen performance results were not
rewritten.

## Status vocabulary

- **Confirmed bug** — current public code can fail or silently mis-handle a
  valid/invalid input.
- **Reproducibility gap** — the public release lacks information or artifacts
  needed for independent reconstruction.
- **Research evidence gap** — the claim requires new experimental evidence.
- **Documentation problem** — the evidence exists but presentation is
  ambiguous or overstates what is supported.
- **Resolved before this audit** — a reported concern was already fixed on the
  audited `main`.

## Executive findings

The implementation-level P0 findings were real: the official evaluation
wrappers incorrectly tested a PATH command with `-x`, and the selector did
not verify row alignment or score integrity before combining base/calibration
files. Both are fixed on the audit branch with CPU-testable guards.

The larger limitations are scientific rather than cosmetic. The current
downstream evidence is single-seed; the correction ablation establishes
selection overlap rather than downstream utility; the exact raw-data-to-20K
construction is not publicly reconstructable; and historical raw NVML logs are
not public. These limitations remain open and must not be hidden by release
engineering.

## Issue-by-issue audit

### Issue 1 — Official evaluation Python executable check

**Severity:** P0

**Classification:** Confirmed bug

**Original problem:** `[ -x "$EVAL_PY" ]` rejects a valid PATH command such
as `python` because `-x` tests the literal path string.

**Root cause:** executable resolution and executable-file testing were
conflated.

**Evidence:** both K128/K256 wrappers on the audited main used
`EVAL_PY=${EVAL_PY:-python}` followed by `[ -x "$EVAL_PY" ]`.

**Files affected:**

- `scripts/evaluation/eval_k128_official.sh`
- `scripts/evaluation/eval_k256_official.sh`

**Fix implemented:**

- resolve PATH commands with `command -v`;
- retain explicit-path executable support;
- resolve the frozen Math evaluator's `python3`;
- compare `sys.prefix` and Python major/minor for `EVAL_PY` and
  `python3`;
- add `--preflight-only`, which checks the Python environment and frozen
  evaluator hashes without requiring a trained model or GPU.

**Validation:** GitHub Actions runs `bash -n` and both CPU preflight commands.
A completed audit-branch CI run reported both
`OFFICIAL_EVAL_PREFLIGHT=PASS`.

**Remaining risk:** the two wrappers intentionally retain substantial
duplication. During a fidelity audit, centralizing them into a new shared
launcher offers limited benefit and creates another execution layer. Safe
deduplication is deferred rather than mixed into this fix.

### Issue 2 — Selector input alignment and integrity

**Severity:** P0

**Classification:** Confirmed bug

**Original problem:** base/calibration rows were combined with `zip` after
only a length check. A same-length reordered file could therefore compare
scores from different examples and then use the resulting position to index
the candidate pool.

**Root cause:** no explicit scorer-output schema or row-identity contract was
enforced at selection time.

**Files affected:**

- `scripts/selection/select_from_scores.py`
- new `scripts/selection/core.py`

**Fix implemented:**

- exact row count;
- integer index requirement;
- duplicate, missing, out-of-range and out-of-order index rejection;
- base/calibration index match;
- base/calibration `response_tokens_full` match;
- positive finite integer-valued response length;
- required score fields;
- numeric finite score checks, including NaN/Inf rejection;
- deterministic ΔNLL ranking and stable ΔH ranking that preserves the previous ΔNLL order when ΔH values tie.

**Frozen-behavior note:** the original selector uses Python stable sorting
for both ΔNLL and ΔH. Explicit index tie-breaking at the second stage
changed the order of ΔH ties, so the audit selector was corrected to
preserve preceding ΔNLL order. Frozen output files were not rewritten.

**Validation:** synthetic CPU tests cover malformed alignment/schema cases.
The refactored selector was also rerun against the original K256 and
independent K128 historical scoring files on the experiment server.
Both generated selections match their published frozen files byte-for-byte:
`61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22`
(K256) and
`c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2`
(K128).

**Remaining risk:** historical score files are not public; scorer GPU forward
passes and downstream fine-tuning were not rerun during this audit.

### Issue 3 — Core algorithm semantic tests

**Severity:** P0/P1

**Classification:** Reproducibility gap, substantially improved

**Fix implemented:** `tests/test_selection_semantics.py` now exercises:

- separate K/2 and K prefix statistics on synthetic data;
- L <= K unchanged behavior;
- L > K asymmetric extrapolation closed form;
- q10-q90-style rank trimming;
- corrected ΔH ranking;
- exact target count;
- equal-score boundaries;
- deterministic ΔNLL ordering and stable ΔH tie behavior;
- malformed score schemas;
- permuted rows;
- response-length mismatch;
- correction-ablation modes.

**Validation:** Python 3.10 CPU regression tests cover stable ΔH ties,
linear interpolation, exact sample boundaries, and invalid boundary-gap
rejection. Eight pre-existing Math evaluator warnings remain.

**Remaining risk:** the test suite verifies prefix aggregation and selection
semantics without loading a language model. It does not independently
recompute the scorer's full torch softmax/entropy path from model logits.
That path remains covered by code review/syntax rather than a tiny model-free
numerical integration test.

### Issue 4 — Independent effect of asymmetric correction

**Severity:** P1 research

**Classification:** Research evidence gap

**Evidence status:** the README correction table measures selected-set overlap
with T1 only. It does not demonstrate higher downstream Math/Medical/Macro
utility.

**Implemented support:**

- `select_indices_variant` supports Raw, NLL-Only, H-Only and Both;
- `scripts/selection/select_ablation_from_scores.py` creates new,
  explicitly `EXPERIMENTAL_UNFROZEN` ablation artifacts without replacing
  frozen selections;
- `docs/research_validation_plan.md` freezes the matched comparison design
  and required measurements.

**Remaining blocker:** new 2K fine-tuning and downstream evaluations are
required for the ablation conditions. No such result is fabricated here.

### Issue 5 — Multi-seed confidence and non-inferiority

**Severity:** P1 research

**Classification:** Research evidence gap

**Evidence:** public primary downstream results are seed 42. K128 exceeds the
46.60 operational threshold by only about 0.0057 Macro points.

**Implemented:** a matched 3-seed minimum / preferably 5-seed plan with
per-seed disclosure, paired differences, confidence intervals, and explicit
non-inferiority logic using the predeclared -0.50 Macro margin.

**Important interpretation:** a non-significant difference test is not
evidence of equivalence. The current single-seed result does not support a
statistical non-inferiority claim.

**Remaining blocker:** GPU runs have not been executed.

### Issue 6 — Generalization

**Severity:** P1 research

**Classification:** Research evidence gap

**Implemented plan, in information-per-cost order:**

1. response-length stratification;
2. selection-ratio sensitivity;
3. one additional model size/family condition.

**Remaining blocker:** no new generalization experiment has been run.

### Issue 7 — Candidate-pool and calibration provenance

**Severity:** P1 reproducibility

**Classification:** Reproducibility gap

**Evidence:** the public release records pool composition, path, seed and
SHA-256, but does not preserve enough verified lineage to rebuild that exact
SHA from raw source datasets.

**Missing/unverified public information:**

- exact source-dataset snapshots/version identifiers;
- exact source ordering and sampling implementation;
- duplicate-removal/filtering procedure, if any;
- exact JSONL serialization process;
- complete calibration-subset reconstruction recipe.

Upstream InstructDiff configurations reference domain files, but those
references are not sufficient proof of the exact ShallowFrontier pool
lineage.

**Implemented:** `data/README.md` and `docs/reproducibility.md` now expose
three reproduction levels:

1. final fine-tuning from published selected sets;
2. 20K pool to scoring/selection;
3. raw sources to complete experiment.

Level 3 is explicitly marked not reproducible with current public artifacts.

**Remaining blocker:** recover original source manifests/pool-construction
code from the experiment workspace or backups. Existing hashes must remain
unchanged.

### Issue 8 — GPU energy auditability

**Severity:** P1 reproducibility/research

**Classification:** Reproducibility gap

**Evidence:**

- historical `logger_meta.json` and `idle_baseline_300s.json` reference
  protocol SHA
  `d0fe60258fce097511640167f50898c295fa5a0db5d02379095cd682c65a12c1`;
- the public protocol file has SHA
  `b52f3a4a2dc83254ff63f0c16f6553934c7b3bf94949e2fa7bdd2a9571d9e350`;
- the public protocol already had its current content in the first clean
  public-release commit;
- historical raw 1 Hz NVML CSV is not public; the published stage-marker
  CSV is byte-identical to the recovered original.

**Root cause:** public release preparation preserved aggregate historical
metadata and stage markers, but omitted the historical protocol text and
raw NVML counter stream. Both were subsequently found on the original server.

**Implemented:**

- `docs/energy_provenance.md` preserves and explains the discrepancy instead
  of rewriting historical hashes;
- `scripts/energy/aggregate_energy.py` can recompute per-stage GPU0/GPU1
  counter deltas, gross kWh, boundary gaps and optional idle-adjusted kWh from
  recovered/future logs;
- synthetic unit tests validate the arithmetic, UTC Z timestamps, nearest
  and linear boundary methods, and rejection of bad counters or markers;
- on-server reaggregation of seven complete historical stages reproduces
  all frozen T1/K256/K128 gross and idle-adjusted E2E energy values
  within `1e-9 kWh` using linear boundary interpolation and 46.953 W idle power;
- recovered historical protocol SHA `d0fe...` is verified; direct comparison
  shows the public `b52f...` copy generalizes only the Python executable path.

**Measurement boundary:** GPU-only operational board energy. CPU, RAM,
storage, networking, cooling and datacenter PUE are excluded.

**Remaining blockers:**

- publish or otherwise provide integrity-verified recovered measurement
  evidence before claiming public sample-level reproduction;
- repeated energy runs are needed to estimate run-to-run uncertainty;
- the exact original historical aggregation source code has not been
  recovered, although its published numeric outputs were reconstructed.

### Issue 9 — Environment and dependency freezing

**Severity:** P1 reproducibility

**Classification:** Reproducibility gap, partially improved

**Existing evidence:** the repository records the key train/eval package
versions, Python version and 2x RTX 5090 hardware.

**Implemented:** `scripts/repro/check_environment.py` reports and optionally
strict-checks the recorded train/eval Python, key package versions, CUDA
runtime and GPU topology.

**Remaining risk:** the public requirements are not an exact historical
transitive lock/pip-freeze. A complete lock should only be published if it can
be recovered from the surviving historical environment or contemporaneous
artifact; it should not be fabricated retrospectively.

### Issue 10 — CI expansion

**Severity:** P2 engineering

**Classification:** Reproducibility/software-quality gap, improved

**Implemented CI checks:**

- Ruff F821;
- Bash syntax;
- CPU official-eval preflight;
- release-integrity tests;
- selector semantic tests;
- energy aggregation tests;
- coverage report.

**Observed validation:** completed audit-branch run: 28 passed, 8 pre-existing
warnings; selector-directory coverage report total 42%. The coverage number is
descriptive, not used as a release threshold.

**Remaining risk:** GPU model loading/training/evaluation are intentionally not
executed in CPU CI.

### Issue 11 — Figure and result reporting rigor

**Severity:** P2 publication

**Classification:** Documentation problem

**Historical issue:** the previous `fig2_pareto.py` mixed a Full20K
energy estimate with measured E2E methods in one Pareto-style plot.

**Phase 2 resolution:**

- Replaced the previous plot with `figs/scoring_cost_utility.png`,
  generated by `scripts/plots/fig2_scoring_cost_utility.py`.
- The new main figure reads the frozen comparison CSV and visualizes
  processed candidate-scoring tokens versus downstream Macro utility.
- The estimated Full20K energy point is not included in this figure.
- The Scoring, Energy, SUE, Prefix Depth, and Carbon Scenario figures
  use consistent method colors where applicable.
- Public method labels use T1, K256, and K128. Historical machine-readable
  identifiers remain unchanged.
- The old `figs/performance_energy_pareto.png` was retired.
- The approved `figs/overview.png` and `figs/overview.pdf` are retained.

The stagewise-energy chart displays measurements from discrete pipeline
stages. Its shaded regions are visual comparisons, not integrals representing
total E2E energy. E2E energy is the sum of the measured stage energies.

All regenerated Phase 2 figures underwent local visual review. Future
modifications must be regenerated and visually inspected before release.

**Carbon scenario finding:** the scale-out script used
`0.4541 kgCO2eq/kWh`, but the public repository did not preserve its source,
reference year or publication date.

**Implemented:** the value is preserved in
`results/energy/carbon_factor_status.json` with status
`UNVERIFIED_PUBLIC_PROVENANCE`; the plotting script now displays that status.
No missing citation is invented. The figure remains a scenario estimate.

### Issue 12 — Licensing and upstream attribution

**Severity:** P2 release/legal

**Classification:** Reproducibility/release gap

**Historical finding:** this repository initially lacked a root license.
The upstream InstructDiff README displays an MIT badge, but the linked root
LICENSE was not retrievable during the audit. The vendored Math evaluator
and latex2sympy retain separate MIT license texts.

**Phase 2 resolution:** a scoped root `LICENSE` was added for original
ShallowFrontier contributions to the extent owned by the named copyright
holder and within their licensing authority. `NOTICE`, `README.md`, and
`docs/licensing.md` describe the scope and third-party exclusions.

**Remaining blocker:** the exact rights governing adapted InstructDiff
code, other third-party materials, dataset redistribution, and applicable
institutional IP requirements remain to be verified. The root license
does not grant blanket permission for the entire repository.

## Research questions

### Q1. Does performance preservation repeat across seeds?

**Unresolved.** Only seed-42 downstream evidence is currently frozen.

### Q2. Does asymmetric correction improve downstream utility over raw prefix?

**Unresolved.** It improves T1 selection overlap in the published ablation;
matched downstream ablation is still required.

### Q3. What is the relationship between T1 overlap and downstream utility?

**Unresolved.** The current study has too few independently trained ablation
points to establish the relationship.

### Q4. How much token reduction becomes GPU-energy reduction?

**Partially answered for the frozen seed-42 runs:** scoring-token reductions
are 29.14% (K256) and 46.92% (K128), while matched gross E2E GPU-energy
reductions are 5.70% and 11.37%. This shows the E2E conversion is much smaller
because calibration and final fine-tuning are not eliminated. Statistical
uncertainty is not available.

### Q5. Does the effect generalize across model/data conditions?

**Unresolved.**

### Q6. What does the 46.60 threshold mean?

It is a predeclared operational utility tolerance of 0.50 Macro points below
the T1 reference. It is not a significance cutoff, confidence bound or
field-wide standard.

### Q7. What is ShallowFrontier's contribution relative to InstructDiff?

InstructDiff provides the underlying differential-entropy data-selection
framework and upstream components. ShallowFrontier's evaluated modification is
prefix-limited candidate scoring plus asymmetric ΔH extrapolation, together
with K128/K256 operating points and matched selection-cost/energy analysis.
Claims should not attribute the broader InstructDiff framework to
ShallowFrontier.

## Frozen-artifact integrity

No frozen selected dataset, frozen performance summary, historical manifest or
historical energy result is intentionally modified by this audit branch.

Any new ablation output is required to use a new namespace and is marked
experimental/unfrozen.

## Merge blockers

Before this draft PR should be considered release-ready:

1. current branch CI must be green;
2. changed figures must be regenerated and visually checked;
3. retain the now-verified K256 and K128 on-server selector replay
   evidence and clearly distinguish it from public-only reproducibility;
4. do not convert the open research-evidence gaps into claims without the
   required GPU experiments.
