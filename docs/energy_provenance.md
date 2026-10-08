# Energy Measurement Provenance Audit

This note separates historical experiment evidence from the public rerun
protocol. It does not alter any frozen result or manifest.

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

## Raw-log availability

The raw 1 Hz `device_energy_1hz.csv` files and historical stage-marker CSV
used for the reported runs are **not present in the public repository**.

Therefore the published T1/K256/K128 E2E kWh values can be checked for
internal consistency against the frozen summaries, but cannot currently be
independently reconstructed sample-by-sample from public raw counters.

If the original raw logs are recovered, they should be added without
modification, with SHA-256 checksums and a separate aggregation script. Their
absence must not be represented as full raw-energy reproducibility.

## Protocol-hash discrepancy

Two different protocol hashes are present in the public release:

- historical logger metadata and the historical idle result reference
  `d0fe60258fce097511640167f50898c295fa5a0db5d02379095cd682c65a12c1`;
- the public `results/energy/energy_measurement_protocol.json` has SHA-256
  `b52f3a4a2dc83254ff63f0c16f6553934c7b3bf94949e2fa7bdd2a9571d9e350`.

The public protocol file already had its present content in the first clean
public-release commit. The public Git history therefore does not contain the
historical `d0fe...` protocol content and cannot establish exactly which
textual changes produced the new hash.

Consequences:

1. Historical results that record `d0fe...` remain historical artifacts and
   must not be rewritten to `b52f...`.
2. The current public idle-baseline script intentionally checks the public
   `b52f...` protocol for **new reruns**; this does not retroactively prove
   that the historical run used that file.
3. Until the original `d0fe...` protocol is recovered from the experiment
   workspace or backup, this discrepancy remains an unresolved provenance
   gap.

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

A carbon-intensity value must be accompanied by its source, geography,
reference year, publication date, and whether it is average or marginal grid
intensity. The placeholder `0.4173 kgCO2eq/kWh` inside the public protocol is
explicitly marked unverified and must not be used as a final reported factor.

Any scale-out CO2 figure is a scenario estimate, not a measured emission.
