# Licensing and Upstream Attribution Audit

This note documents the licensing state observed during the public-release
audit. It is not legal advice.

## Repository-level status

The ShallowFrontier repository currently has no top-level `LICENSE` file.

The upstream InstructDiff README displays an MIT-license badge linking to a
root `LICENSE`, but the current upstream repository does not expose that root
`LICENSE` file through the GitHub contents API. Because the underlying
license text is missing, this audit does not infer or recreate a project-wide
license for adapted InstructDiff code.

Accordingly, the ShallowFrontier repository should **not** add a blanket
project-wide license that purports to relicense upstream-derived files unless
the relevant rights and upstream license text are first verified.

## Components with explicit licenses

The vendored Math evaluator retains an MIT license under:

```text
eval/math_evaluation/LICENSE
```

The bundled `latex2sympy` component also retains its MIT license under:

```text
eval/math_evaluation/latex2sympy/LICENSE.txt
```

Those component licenses should remain attached to the corresponding code.

## Attribution

`NOTICE` records that ShallowFrontier builds on InstructDiff and distinguishes
the ShallowFrontier-specific research modifications from the upstream
framework and evaluator components.

For publication and release, attribution should continue to distinguish:

- InstructDiff's original iterative/differential-entropy data-selection work;
- third-party evaluation code and datasets;
- ShallowFrontier-specific prefix scoring, asymmetric extrapolation,
  cost/energy measurement, and K128/K256 experiments.

## Required follow-up before declaring a repository-wide license

1. Recover or verify the exact license governing the InstructDiff source files
   that were copied or adapted.
2. Inventory copied/adapted files and identify their upstream commit.
3. Preserve every third-party component's original notice and license.
4. Review dataset redistribution terms separately from source-code licensing.
5. Only then choose a license for code written solely for ShallowFrontier and
   decide whether a root license can accurately describe the mixed repository.

Until that work is complete, the absence of a root license is preferable to
an unsupported relicensing claim.
