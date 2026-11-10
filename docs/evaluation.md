# Evaluation

## Benchmarks and scores

- **Math:** MATH500, Minerva Math, OlympiadBench, AIME24, AMC23. The score is the mean accuracy over 16 samples at temperature 1,
  averaged over the five benchmarks (`eval/math_evaluation`).
- **Medical:** MedQA, MMLU medical subsets, MedMCQA, scored with the multiple-choice formulation (MCF): only the option letters
  compete. Every token that decodes to A, B, C or D (any case, any leading space) counts for that letter, and a letter's score is the
  log-sum-exp of its tokens' log-probabilities (`eval/medeval/mcf_eval.py`).
- **Mixed pools:** (math average + medical average) / 2.

## The released InstructDiff medical rule

`mcf_eval.py --protocol released` reproduces the rule of InstructDiff's released medical script, for comparison.

- **The rule:** the script reads the first generated token's top-10 log-probabilities and stores one probability per option letter.
  When a letter appears in several forms, the later form (lower probability) overwrites the earlier one.
- **Example (Llama-3-8B base, an MMLU item):** `' B'` has probability 99.69%, but `' b'` at rank 8 overwrites B's entry, and A (0.15%)
  becomes the prediction. The duplicates are the tail of the top-10 list, which a confident model fills with near-copies of its answer
  token.
- **Measured effect (our reproduction, seed 42, Llama-3-8B on the medical-only pool):** points lost by each model under the released
  rule compared with MCF.

| Model | Released rule | MCF | Lost |
|---|---:|---:|---:|
| Base model | 47.88 | 62.22 | 14.3 |
| Full data | 43.45 | 59.42 | 16.0 |
| Random 10% | 54.94 | 58.04 | 3.1 |
| InstructDiff T1 | 58.70 | 59.16 | 0.5 |
| InstructDiff T2 | 58.21 | 58.70 | 0.5 |
| InstructDiff T3 | 49.65 | 57.59 | 7.9 |

- **Who loses:** every model, but by very different amounts. Retraining the same cell repeats the loss closely (rank correlation 0.96),
  yet no property we measured predicted it.
- **Mixed pools:** models trained with more medical rows lost more medical points in our runs. On the Llama-3-8B 20K pool (seed 42),
  ShallowFrontier (17% medical rows) lost 7.0 points, InstructDiff T1 and T3 (0.4% and 0.2% medical rows) lost 1.5 and 0.8. Under the
  released rule, a selection that drops the medical domain therefore looks better than it is.
- **The base model:** it loses about 14 points (the published base score, 47.87, matches the released rule), and the duplicates we
  inspected in its top-10 tails were lowercase variants of the option letters.

## Reporting the best round

Picking the best of InstructDiff's three rounds after seeing test scores adds an optimistic bonus. We measure it as the best minus the
mean of the three rounds, averaged over seeds 42–44.

| Testbed | MCF | Released rule |
|---|---:|---:|
| Llama-3-8B · Medical 10K | 0.38 | 1.42 |
| Llama-3-8B · Mixed 20K | 0.28 | 0.71 |
| Qwen2.5-7B · Mixed 20K | 0.41 | 0.71 (seeds 43–44) |
| Llama-3-8B · Mixed 10K | 0.86 | 1.08 |

- The released rule makes round-to-round scores noisier, so the bonus grows (1.3–3.7×).
- Our tables list every round, and the cost of InstructDiff's best round is counted as all three rounds.
- These are observations from our reproduction. We make no claim about the size of any effect on published numbers.
