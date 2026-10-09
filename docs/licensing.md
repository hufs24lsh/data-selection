# Licensing and Attribution

## Scope of the Root License

The root [`LICENSE`](../LICENSE) makes the original, copyrightable
ShallowFrontier contributions available under the MIT License,
**only to the extent that the named copyright holder owns those
rights and has authority to license them**.

This includes independently authored project-specific source
code and documentation, such as original functionality in
`scripts/`, `tests/`, and project documentation.

This is **not a blanket MIT license for the entire repository**.
Where files incorporate or adapt third-party material, the MIT
grant covers only the independently owned ShallowFrontier
contributions, not the underlying third-party code.

## Upstream InstructDiff Code

ShallowFrontier builds on:

[InstructDiff: Domain-Adaptive Data Selection via Differential
Entropy for Efficient LLM Fine-Tuning](https://github.com/zhuchichi56/Instruct-diff)

The public upstream README identifies MIT through a license
badge, but the linked root `LICENSE` file was not present
in the upstream repository during this audit.

The exact upstream license grant for the adapted source
therefore remains to be verified. The root ShallowFrontier
license does not resolve this uncertainty or relicense
upstream-derived code.

The release includes code inherited from or adapted around
InstructDiff, including `src/instdiff/` and evaluation
components. Retaining attribution does not, by itself,
establish permission to redistribute upstream material.

## Explicit Third-Party Licenses

The following existing licenses remain in force:

| Component | License file |
|---|---|
| Math evaluator | `eval/math_evaluation/LICENSE` |
| Bundled latex2sympy | `eval/math_evaluation/latex2sympy/LICENSE.txt` |

Both contain MIT license terms and their own copyright
notices. They are not replaced by the root `LICENSE`.

Other evaluation materials must not be assumed to be covered
by these two files without verifying their provenance.

## Data, Models, and Research Artifacts

The root MIT license does not independently grant rights
to redistribute or reuse:

- third-party evaluation or training datasets;
- source material inside selected JSONL examples;
- model weights, checkpoints, or tokenizer assets;
- third-party content contained in figures or results.

Those materials remain subject to their respective terms.
Research results and numerical measurements are documented
for reproducibility, without claiming that all underlying
content is owned by the ShallowFrontier copyright holder.

## Attribution

[`NOTICE`](../NOTICE) records the original InstructDiff
framework and the ShallowFrontier-specific research work.
Academic citation and software licensing serve different
purposes; both should be preserved where applicable.

## Outstanding Release Checks

Before declaring the complete repository freely reusable
under a single license:

1. Verify the upstream InstructDiff license text or obtain
   appropriate authorization for the inherited code.
2. Confirm any institutional or research-project IP
   obligations that apply to the original contributions.
3. Audit provenance and redistribution permissions for
   remaining evaluation code and included datasets.
4. Distinguish new standalone code from adapted upstream
   material at file or component level where practical.

The root license does not waive these requirements.

This document describes the repository licensing scope
and is not legal advice.
