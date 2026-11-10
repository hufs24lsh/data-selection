# Method

Goal: pick k = 10% of a pool D for full fine-tuning of a base model M_base with a fixed recipe (learning rate, global batch b,
maximum length, 3 epochs).

## 1. One shallow calibration

A random 10% subset (InstructDiff's warm-up rule: `random.seed(seed); random.sample(D, k)`) trains a calibration model M_cal with the
same recipe. This equals InstructDiff's first round; there are no further rounds.

## 2. Scores

InstructDiff's per-row statistics over the response tokens, under M_base and M_cal (`src/shallowfrontier/scoring.py`):

- `base_avg_entropy` H_base, the mean token entropy of M_base;
- `diff_entropy` ΔH = H_base − H_cal and `diff_nll` ΔNLL = NLL_base − NLL_cal (means over response tokens).

## 3. Label-free grouping (`src/shallowfrontier/grouping.py`)

- **Features:** each row's prompt only (the response is never read) is encoded by M_base. The feature is the last-layer hidden state
  averaged over the prompt tokens (prompts cut at 512 tokens).
- **Clustering:** cosine k-means with k-means++ seeding, for k = 2, …, ⌊√(n/2)⌋. The k with the largest exact mean silhouette wins.
- **No structure:** if the best silhouette is below 0.25, the pool stays one group. On our mixed pools this gives two groups that are
  97–99% math or medical, without any domain tag.

## 4. Per-group gradient noise scale (`src/shallowfrontier/noise.py`)

- **Estimator (McCandlish et al., 2018):** B_g = tr(Σ_g) / |G_g|². G_g is the mean and Σ_g the covariance of per-row gradients
  g_i of the summed response-token NLL on M_cal.
- **Sketch:** each gradient keeps a fixed random 0.1% of every tensor's coordinates. The ratio is unbiased under random coordinate
  sampling.
- **Error estimate:** rows go to 16 blocks, so B_g gets a jackknife standard error se_g.
- **Rows:** each group's held-out rows (not used for calibration), in one seeded random order, measured in rounds of b rows per group,
  all groups in lockstep.
- **Stopping rule:** after each round, with s_g = (n_g/B_g) / Σ_h(n_h/B_h), k_g = k·s_g and r_g = se_g/B_g:

  ```
  share error  sd_g = k_g · sqrt((1 − s_g)² r_g² + Σ_{h≠g} s_h² r_h²)
  route error  k_g · P(flip),   P(flip) = Φ(−|ln(B_g / b)| / r_g)
  E_g = sd_g + k_g · P(flip)
  ```

  A group is settled when E_g ≤ b (the expected change of the selection is below one batch) and N_g ≥ B_g, or when it has no
  held-out rows left. No row count is fixed in advance; in our runs the measurement used 2–26% of a pool.

## 5. Shares and routes (`src/shallowfrontier/selection.py`)

- **Share:** k_g ∝ n_g / B_g, rounded with the largest-remainder rule. A group with clean gradient signal (small B) gets more rows,
  and no group gets zero.
- **Route:**
  - B_g / b < 1: the group's rows with the lowest H_base ("easy" rows).
  - B_g / b ≥ 1: InstructDiff's rule inside the group. Drop 10% at each tail of ΔNLL, then take the lowest ΔH.

Why the boundary sits at B = b and why each side gets its rule: section 6.

## 6. Why two routes

**What B / b measures (theory).** The gradient noise scale B is the number of examples needed to average out how much per-example
gradients disagree (McCandlish et al., 2018). With a global batch of b examples:

- **B < b:** one batch already averages the disagreement away. Every step follows the group's shared direction, and training on the
  group is limited by the number of steps rather than by what individual rows contain.
- **B ≥ b:** one batch cannot average it away. Rows pull in different directions, so which rows are in the batch matters.

**Where B is measured.** B grows as the loss falls (McCandlish et al., 2018), so it is measured at the end of calibration, on M_cal,
with each group's held-out rows. Calibration rows are not used: after three epochs they are close to memorized, their mean gradient is
near zero, and B would come out large for every group. B / b therefore describes what is left to learn in a group after a 10% run.

**Why each side gets its rule (interpretation).**

- **B < b, lowest H_base:** what is left is a shared direction, typically the answer format and style. Rows the base model already
  predicts with low entropy show that pattern with the least row-specific variation, so they teach it at the lowest cost.
- **B ≥ b, InstructDiff's rule:** the shared part is learned, and what is left is row-specific content. InstructDiff's contrast between
  M_base and M_cal (ΔNLL filter, then lowest ΔH) targets rows that still carry learnable content.
- This matches earlier findings that easy examples help when the training budget is small and hard ones when it is large
  (Sorscher et al., 2022; Hacohen et al., 2022). B / b reads that regime per group, from one calibration run, instead of fixing it per
  dataset.

**Evidence.**

- **Same rows, different regime** (Qwen2.5-7B, math pool, 1,000 selected rows, seed 42). With lr 5e-5 and batch 256 (12 steps) the
  calibrated model ends at B / b = 0.27, and lowest-entropy selection beats InstructDiff's round 1 by 2.22 points. With lr 2e-5 and
  batch 64 (48 steps) it ends at B / b = 1.49, and InstructDiff's round 1 wins by 1.84 points.
- **Routes in our runs** (seed 42, the runs behind the README tables):

| Testbed | Groups and B / b | Route | ShallowFrontier | Best InstructDiff round |
|---|---|---|---:|---:|
| Qwen2.5-7B · Math 10K | two math groups: 0.25, 0.24 | lowest H_base | 29.42 | 28.16 (T2) |
| Llama-3-8B · Medical 10K | one group: 2.51 | InstructDiff's rule | 59.16 | 59.16 (T1, same rows) |
| Qwen2.5-7B · Mixed 20K | math 2.43, medical 5.60 | InstructDiff's rule | 47.38 | 47.31 (T1) |
| Llama-3-8B · Mixed 20K | math 0.18, medical 0.73 | lowest H_base | 34.61 | 33.95 (T1) |
| Llama-3-8B · Mixed 10K | math 0.22, medical 0.75 | lowest H_base | 34.05 | 33.34 (T2) |

- Math groups fall on both sides: lowest H_base in the Qwen math pool and on Llama-3-8B, InstructDiff's rule in the Qwen mixed pool.
  The route follows the measured regime, not the domain.

**Limitations.**

- The threshold B / b = 1 comes from the critical batch size. Which rule goes to which side was chosen during development from runs like
  the first example, and the reading above is an interpretation, not a proof.
- On Llama-3-8B the development evidence was weaker: a few small groups pointed the other way.
- In every mixed pool the math group had a smaller B than the medical group, so a domain effect is not fully separated from the noise
  effect. The Qwen pools, where math groups fall on both sides, are the main evidence against a domain-only rule.

## Constants and where they come from

| Constant | Value | Source |
|---|---|---|
| Selection and calibration size | 10% of the pool | InstructDiff's setting |
| Routing threshold | B / b = 1 | critical batch size: one batch averages the gradient noise |
| No-structure floor | silhouette 0.25 | Kaufman & Rousseeuw |
| Largest k | ⌊√(n/2)⌋ | rule of thumb; k = 2 won on every mixed pool |
| Stopping tolerance | one batch | a smaller change barely alters training |
| ΔNLL trim | 10% per tail | InstructDiff's rule |

None of them is fitted to our pools.

## References

- G. Hacohen, A. Dekel and D. Weinshall. Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets. ICML 2022.
- L. Kaufman and P. J. Rousseeuw. Finding Groups in Data: An Introduction to Cluster Analysis. Wiley, 1990.
- S. McCandlish, J. Kaplan, D. Amodei and the OpenAI Dota Team. An Empirical Model of Large-Batch Training. arXiv:1812.06162, 2018.
- B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli and A. S. Morcos. Beyond Neural Scaling Laws: Beating Power Law Scaling via Data
  Pruning. NeurIPS 2022.
- J. Su et al. InstructDiff: Domain-Adaptive Data Selection via Contrastive Entropy for Efficient LLM Fine-Tuning. ACL 2026.
  [aclanthology.org/2026.acl-long.486](https://aclanthology.org/2026.acl-long.486/)
