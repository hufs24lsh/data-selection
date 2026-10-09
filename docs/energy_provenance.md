# Energy Measurement Provenance Audit

This note separates historical experiment evidence from the public rerun
protocol. Numeric experimental results and frozen manifests are unchanged.
Current public reporting identifiers have been standardized.

## What is independently inspectable

The repository publishes:

- the NVML logger implementation in `scripts/energy/energy_logger.py`;
- the idle-baseline implementation in
  `scripts/energy/measure_idle_baseline.py`;
- hardware and measurement-scope metadata;
- the historical 300 s idle-baseline summary;
- aggregate stage/E2E results under `results/cost/` and
  `results/energy/`.

The primary reported physical quantity is **gross GPU board energy** from
NVML cumulative energy-counter differences. The scope is GPU-only
operational energy; CPU, RAM, storage, networking, cooling, and PUE are not
measured.

## Raw-log recovery and public availability

The original NVML 1 Hz counter log was recovered from the historical
experiment server at:

`Instruct-diff/runs/shallow_instructdiff_v1_seed42/energy/device_energy_1hz.csv`

It is **not included in the public repository**. The original logger was
still writing to that file during the 2026-10-09 audit. Accordingly, a
fixed, filtered historical-window snapshot was used for reaggregation
without stopping or modifying the logger.

Recovered evidence:

- Historical protocol SHA-256:
  `d0fe60258fce097511640167f50898c295fa5a0db5d02379095cd682c65a12c1`.
- Historical stage-marker SHA-256:
  `059d2adf837e4c7a3c088072b21157d2f7f544d24a5b8afb784e658c0a378e26`.
- The original published stage-marker CSV was byte-identical to the
  historical measurement log.
- The current CSV standardizes K128 stage labels only. All timestamps,
  events and exit codes are unchanged.
- The original file is recoverable from Git commit `5b8410beb0cc093a71441e01f9b0f793fbfbc300`.
- The standardized stage-marker SHA-256 is `4ea47c4ed4e3c4c2649c44c294bc29e0d3a596aa622eb60bf8e1bb514f8962ff`.
- Historical-window snapshot: 17,000 CSV data rows, SHA-256
  `48ee1f3d2c835de809a50ce69d774bee4448d586b0e9dc4ad1ff702cd22816da`.
- Seven complete matched-comparison stages were reaggregated.
  The unmatched `COST_FULL20K_E2E_RERUN` START marker has no END marker
  and was excluded. Official evaluation stages were also excluded.
- The reconstructed seven-stage report has SHA-256
  `25c17063ceadf44b4acd884aeb96a7dfd831bca1999709c7cefea169b0b289af`.
- The frozen-comparison verification report has SHA-256
  `f06a0918521059f0129c63728c71a274c7f8294882da1f1d4167e3298468092a`.

These snapshots and verification reports are currently local research
runtime artifacts, not public repository artifacts. The public release
alone therefore still cannot independently reconstruct historical
energy from raw counters. Recovery on the historical server is not
equivalent to public raw-log redistribution.

## Historical protocol identity

The original `energy_measurement_protocol.json` was recovered on the
historical server and matches SHA-256
`d0fe60258fce097511640167f50898c295fa5a0db5d02379095cd682c65a12c1`.

The public protocol SHA-256 is
`b52f3a4a2dc83254ff63f0c16f6553934c7b3bf94949e2fa7bdd2a9571d9e350`.

A direct file comparison identified one changed field:

- Historical: `"python"` was set to a server-specific absolute interpreter path (the exact value is preserved in the recovered SHA-verified protocol).
- Public: `"python": "python"`

Thus the protocol hash discrepancy is explained by the generalized
Python executable path in the public copy. Historical metadata must
continue to reference the historical hash; the new-rerun protocol
and its separate hash must not be retroactively substituted.

The original historical protocol content has not yet been published.

## Gross and idle-adjusted energy

For a measured interval:

```text
gross_gpu_kWh =
    ((GPU0_counter_end - GPU0_counter_start)
   + (GPU1_counter_end - GPU1_counter_start))
    / 3.6e9
```

where NVML cumulative counters are in mJ.

The secondary idle-adjusted value is:

```text
idle_adjusted_gpu_kWh =
    gross_gpu_kWh
    - idle_two_gpu_power_W * wall_seconds / 3.6e6
```

The study reports gross GPU energy as primary. Idle-adjusted energy is
secondary because idle subtraction adds an additional measurement assumption.

## Historical E2E numerical reconstruction

The audited `scripts/energy/aggregate_energy.py` supports explicit
`--boundary-policy linear` in addition to the original `nearest` default.
Linear interpolation of NVML cumulative energy counters at the recorded
stage-marker times, with `--idle-power-w 46.953`, reproduces all three
frozen matched E2E gross and idle-adjusted values within `1e-9 kWh`.

| Method | Frozen gross GPU kWh | Reaggregated gross GPU kWh |
|---|---:|---:|
| T1 | 0.682982462952126 | 0.682982462933451 |
| K256 | 0.644047837541385 | 0.644047837532730 |
| K128 | 0.605340396994137 | 0.605340396994129 |

The historical idle-baseline measurement recorded
`46.95324080267559 W`. The frozen idle-adjusted results are numerically
consistent with `46.953 W`, the value rounded to three decimal places.

The exact original historical aggregation source code was not recovered.
Therefore this establishes a numerical reconstruction from original
measurement evidence, **not proof of historical source-code identity**.
No new energy measurement or repeated-run uncertainty estimate was made.


## Measurement uncertainty

Only one historical idle-baseline window and one reported matched E2E run per
method are public. This is insufficient to estimate run-to-run energy
variance or a confidence interval for the reported energy reduction.

Future confirmatory runs should preserve, per run:

- raw 1 Hz logger CSV;
- stage-marker CSV;
- start/end cumulative counters for each GPU;
- logger and protocol hashes;
- GPU model, driver, NVML version and visible-device mapping;
- concurrent-process preflight;
- wall-clock timestamps;
- aggregate stage table produced from the raw log.

Repeated measurements, rather than a single idle subtraction, are required
before making a strong uncertainty claim.

## Carbon conversion

Carbon emissions are derived, not directly measured:

```text
operational_CO2eq = measured_GPU_kWh * grid_carbon_intensity
```

For current carbon reporting, the official Republic of Korea
**2023 consumption-end electricity emissions factor** is used:

- Factor: **0.4173 kgCO2eq/kWh**.
- Reference year: **2023**.
- Announcement date: **2025-12-18**.
- Issuer: Ministry of Climate, Energy and Environment,
  Republic of Korea.
- Official source: https://mcee.go.kr/home/web/board/read.do?pagerOffset=530&maxPageItems=10&maxIndexPages=10&searchKey=&searchValue=&menuId=10598&orgCd=&boardMasterId=939&boardCategoryId=&boardId=1829260&decorator=
- Boundary: national-average consumption-end electricity.

The official unit of tCO2eq/MWh is numerically identical to
kgCO2eq/kWh. Current reporting metadata is stored in
`results/energy/carbon_factor_status.json`.

The historical energy measurement protocol remains frozen.
Its prospective verification flag describes the status when
that protocol was originally created, not the status of current
reporting.

The factor converts measured gross GPU energy into modeled
operational CO2eq. It does not represent a direct measurement
of emissions and excludes unmeasured infrastructure overhead.
