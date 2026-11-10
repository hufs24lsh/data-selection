# Data

ShallowFrontier uses the training pools of [InstructDiff](https://github.com/zhuchichi56/Instruct-diff) (commit `28adfe3`). The data are not
redistributed here; this folder holds only the instructions and the row orders of the mixed pools.

## 1. Source files

Clone InstructDiff with Git LFS (the math file is an LFS object) and copy two files into this folder:

```bash
git lfs install
git clone https://github.com/zhuchichi56/Instruct-diff.git && git -C Instruct-diff checkout 28adfe3 && git -C Instruct-diff lfs pull
cp Instruct-diff/data/numina_cot_10k.jsonl Instruct-diff/data/train_medmcqa_alpaca_10k.jsonl data/
```

| File | Rows | SHA-256 |
|---|---:|---|
| `data/numina_cot_10k.jsonl` (math) | 10,000 | `a172916f9661937772d123e614cf1066b6c0586d2a871a8548988750d5a96b18` |
| `data/train_medmcqa_alpaca_10k.jsonl` (medical) | 10,000 | `43c2aed3caf6be5afc973c5733d07029ec318e764c27f2cc77560210c51bca37` |

## 2. Mixed pools

```bash
python scripts/make_mixed_pools.py
```

writes the two mixed pools byte for byte and checks their SHA-256. Each row is a source row with a `domain` field appended (`math` or
`med`); `data/mixed/*.order.json` stores only which source line goes where.

| Pool | Rows | SHA-256 |
|---|---:|---|
| `data/mixed/math10k_med10k_seed42.jsonl` (all 10K math + all 10K medical) | 20,000 | `3c31c43d47065b1b850568588ae17fdd06e7b9f8e7f9f819a812d459b54b1f05` |
| `data/mixed/math5k_med5k_seed42.jsonl` (5K math + 5K medical) | 10,000 | `7cf5dfa483818bb7aa1bce9b169510b26c31daff2ec452b7e738d95fe60b99f9` |

The `domain` field is used only to report the composition of a selection; ShallowFrontier never reads it.
