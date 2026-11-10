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

**Reading of the boundary (interpretation).** When B < b, one batch already averages out the disagreement between rows: the group's
rows push the model in a shared direction, typically the answer format, and low-entropy rows teach it cheaply. When B ≥ b, rows disagree
and carry content of their own; InstructDiff's contrastive order is then the better guide.

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
