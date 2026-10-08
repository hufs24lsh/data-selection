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
| Refactored selector reproduces frozen selected SHA from historical score files | **Verified on historical server (CPU replay)** | K256 and independently scored K128 both reproduce frozen selected files byte-for-byte. The underlying score inputs are not public, so this is not public-only reproduction. |
| Official evaluation wrapper PATH/absolute Python handling | **Independently verified (CPU preflight)** | GitHub Actions executes both wrappers in `--preflight-only` mode. |
| Official K128/K256 downstream evaluation itself | **Documented but not rerun** | GPU/model-dependent; frozen summaries are preserved. |
| Final fine-tuning from published selected sets | **Documented but not rerun by this audit** | Entry scripts, hashes and key environment constraints are public; exact GPU rerun was not performed. |
| 20K pool to scoring/selection | **Not reproducible with repository alone** | Exact pool and calibration checkpoint/subset artifact are not redistributed. |
| Raw upstream data to exact 20K pool | **Not reproducible with public artifacts** | Source snapshots, builder/dedup/filter/serialization lineage is incomplete. |
| Historical gross/idle-adjusted energy arithmetic from raw logs | **Numerically reconstructed on historical server; not public-only reproducible** | Recovered 1 Hz raw log, publicly available matching stage markers, seven-stage snapshot and linear boundary interpolation reproduce T1/K256/K128 gross and idle-adjusted frozen values within 1e-9 kWh. The raw snapshot is not public. |
| Energy aggregation implementation | **Verified (CPU synthetic and historical raw-log replay)** | Explicit nearest/linear boundary policies, UTC Z parsing, idle subtraction and input safeguards are tested; Python 3.10 CPU tests also cover exact sample boundaries and invalid boundary-gap rejection. Historical aggregation source-code identity is not verified. |
| Historical energy protocol identity | **Recovered and compared on historical server** | Original `d0fe...` protocol was located and SHA-verified. Public `b52f...` version differs by the logger Python executable path. Original historical protocol is not redistributed in the public repository. |
| Energy reduction uncertainty | **Not reproducible / insufficient evidence** | Only one public matched run per method; no repeated-run variance estimate. |
| CO2 conversion factor source/year | **Not yet verified** | 0.4541 is preserved for scenario reproduction but source metadata is absent; status is `UNVERIFIED_PUBLIC_PROVENANCE`. |
| Correction downstream benefit vs raw prefix | **Not yet investigated experimentally** | Existing evidence is T1 selection overlap only; ablation tooling/plan is prepared. |
| Multi-seed utility preservation / non-inferiority | **Not yet investigated experimentally** | Single-seed public result cannot support a statistical non-inferiority conclusion. |
| Generalization across model/data/selection ratio | **Not yet investigated experimentally** | Planned in `docs/research_validation_plan.md`. |
| Exact transitive train/eval environment lock | **Not reproducible from public artifacts** | Key versions are documented and a strict checker is provided, but no contemporaneous full lock/pip-freeze is public. |
| Original-contribution licensing and repository-wide rights | **Scoped MIT added; blanket licensing unresolved** | The root LICENSE covers original ShallowFrontier contributions only to the extent the named holder is authorized to license them. Upstream InstructDiff permissions, other third-party material, and institutional IP obligations require separate verification. |

## CI meaning

CPU CI establishes syntax, release-artifact integrity, evaluation preflight,
selection semantics and synthetic energy arithmetic. It does **not** establish
that CUDA training/evaluation succeeds on arbitrary hardware or that published
scientific results have been independently replicated.
