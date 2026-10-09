# Data

Large training datasets are not duplicated in this repository.

## Candidate Pool

The final ShallowFrontier experiments used a fixed 20,000-example mixed-domain pool:

- 10,000 mathematical reasoning examples
- 10,000 medical examples
- random seed: 42

Expected experiment path:

```text
data/mixed/math10k_med10k_seed42.jsonl
```

SHA-256:

```text
3c31c43d47065b1b850568588ae17fdd06e7b9f8e7f9f819a812d459b54b1f05
```


## Provenance and Reconstruction Status

The public release currently identifies the frozen candidate pool by its
composition, expected path, row count, seed, and SHA-256. It does **not** yet
contain enough verified information to reconstruct that exact SHA-256 from
upstream raw datasets alone.

In particular, the following items remain missing or unverified in the public
artifacts:

- exact upstream dataset snapshot/version identifiers for the Math and Medical
  source pools;
- the exact sampling implementation and source ordering;
- any duplicate-removal or filtering procedure used before the 10K+10K merge;
- the exact JSONL serialization rules used when the frozen pool was written;
- the complete public recipe that regenerates the frozen calibration subset.

These omissions are treated as a reproducibility gap rather than filled in by
inference. Upstream InstructDiff configuration files reference Math and Medical
training files, but those references alone are not proof of the exact
ShallowFrontier pool lineage.

If the original pool-construction code and source manifests are recovered,
they should be added as new provenance artifacts while preserving the existing
pool and warmup hashes unchanged.

## Warmup Calibration Subset

The InstructDiff T1 reference and ShallowFrontier cost-matched experiments use the same fixed 2,000-example warmup subset.

SHA-256:

```text
ada4cb631860df9039fb4aced93ef56fba2ab92ceb723f993fb70d2ede2a0326
```

## Final Selected Sets

The exact final selected training sets are included under:

```text
results/selections/k256_selected.jsonl
results/selections/k128_selected.jsonl
```

Expected SHA-256 values:

```text
K256:
61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22

K128:
c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2
```

The repository intentionally excludes model checkpoints and large raw dataset copies.
