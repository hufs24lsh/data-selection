# Reproducibility Status

This table distinguishes what the public release actually supports from what is
only documented or still missing. "Verified" never means a GPU experiment was
rerun during the repository audit unless explicitly stated.

| Component | Status | Evidence / limitation |
|---|---|---|
| Published K128/K256 selected-set files and SHA-256 | **Verified against frozen artifacts** | CI checks file presence, 2,000 rows and frozen SHA-256. |
| Published K128/K256 Macro values | **Verified against frozen artifacts** | CI checks the frozen JSON summaries; no downstream rerun was performed by this audit. |
| Selector asymmetric-extrapolation formula | **Independently verified (CPU synthetic)** | Closed-form and boundary tests exercise L<=K and L>K behavior. |
| q10-q90-style trimming, ranking, exact target size and ties | **Independently verified (CPU synthetic)** | Semantic unit tests cover ranking and deterministic tie behavior. |
| Selector malformed-input rejection | **Independently verified (CPU synthetic)** | Duplicate/out-of-range/reordered indices, missing fields, NaN/Inf and length mismatch are tested. |
| Refactored selector reproduces frozen selected SHA from historical score files | **Not yet verified** | Historical base/calibration score files are not public; replay is a merge blocker if they can be recovered. |
| Official evaluation wrapper PATH/absolute Python handling | **Independently verified (CPU preflight)** | GitHub Actions executes both wrappers in `--preflight-only` mode. |
| Official K128/K256 downstream evaluation itself | **Documented but not rerun** | GPU/model-dependent; frozen summaries are preserved. |
| Final fine-tuning from published selected sets | **Documented but not rerun by this audit** | Entry scripts, hashes and key environment constraints are public; exact GPU rerun was not performed. |
| 20K pool to scoring/selection | **Not reproducible with repository alone** | Exact pool and calibration checkpoint/subset artifact are not redistributed. |
| Raw upstream data to exact 20K pool | **Not reproducible with public artifacts** | Source snapshots, builder/dedup/filter/serialization lineage is incomplete. |
| Historical gross/idle-adjusted energy arithmetic from raw logs | **Not reproducible with public artifacts** | Historical 1 Hz CSVs and stage markers are absent. A reaggregation utility is now available for recovered/future logs. |
| Energy aggregation implementation | **Independently verified (CPU synthetic)** | Known counter deltas, idle subtraction, boundary-gap rejection and counter-regression rejection are tested. |
| Historical energy protocol identity | **Unresolved provenance gap** | Historical metadata references `d0fe...`; public protocol is `b52f...`. Original `d0fe...` text is not in public Git history. |
| Energy reduction uncertainty | **Not reproducible / insufficient evidence** | Only one public matched run per method; no repeated-run variance estimate. |
| CO2 conversion factor source/year | **Not yet verified** | 0.4541 is preserved for scenario reproduction but source metadata is absent; status is `UNVERIFIED_PUBLIC_PROVENANCE`. |
| Correction downstream benefit vs raw prefix | **Not yet investigated experimentally** | Existing evidence is T1 selection overlap only; ablation tooling/plan is prepared. |
| Multi-seed utility preservation / non-inferiority | **Not yet investigated experimentally** | Single-seed public result cannot support a statistical non-inferiority conclusion. |
| Generalization across model/data/selection ratio | **Not yet investigated experimentally** | Planned in `docs/research_validation_plan.md`. |
| Exact transitive train/eval environment lock | **Not reproducible from public artifacts** | Key versions are documented and a strict checker is provided, but no contemporaneous full lock/pip-freeze is public. |
| Repository-wide licensing | **Not yet resolved** | Component licenses exist; upstream root-license state for adapted code must be verified before blanket relicensing. |

## CI meaning

CPU CI establishes syntax, release-artifact integrity, evaluation preflight,
selection semantics and synthetic energy arithmetic. It does **not** establish
that CUDA training/evaluation succeeds on arbitrary hardware or that published
scientific results have been independently replicated.
