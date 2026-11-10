#!/usr/bin/env bash
# End-to-end ShallowFrontier on one testbed:
#   calibration subset -> calibration training -> scoring -> prompt features -> grouping -> noise -> selection -> final training -> evaluation
# Usage:  bash scripts/run_shallowfrontier.sh configs/testbeds/qwen_mix20k.yaml [seed (42)] [work dir (runs/<config>_s<seed>)]
# Training uses the visible GPUs (two in the paper); feature extraction, scoring and noise measurement use the first visible GPU.
set -euo pipefail
CFG=$1
SEED=${2:-42}
NAME=$(basename "$CFG" .yaml)
W=${3:-runs/${NAME}_s${SEED}}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

get() { python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); v = c
for k in sys.argv[2].split('.'): v = v[k]
print('' if v is None else v)" "$CFG" "$1"; }
BASE=$(get base_model); POOL=$(get pool); TEMPLATE=$(get template); K=$(get k)
CAL_K=$(get calibration.size); CAL_LIMIT=$(get calibration.limit)
LR=$(get recipe.lr); B=$(get recipe.batch); LEN=$(get recipe.max_length); EVAL=$(get eval)
TPL=(); [ -n "$TEMPLATE" ] && TPL=(--template "$TEMPLATE")
mkdir -p "$W" && W=$(cd "$W" && pwd)
step() { echo "== $(date '+%F %T') $*"; }

step "1/9 calibration subset"
python -m shallowfrontier.cli calibration --pool "$POOL" --k "$CAL_K" --limit "$CAL_LIMIT" --seed "$SEED" --out "$W/calibration.jsonl"
step "2/9 calibration training"
python scripts/train.py --model "$BASE" --data "$W/calibration.jsonl" --out "$W/calibration_model" --lr "$LR" --batch "$B" --max-length "$LEN" --seed "$SEED"
step "3/9 scoring (base vs calibrated)"
python -m shallowfrontier.cli score --base "$BASE" --calibrated "$W/calibration_model" --pool "$POOL" ${TPL[@]+"${TPL[@]}"} --out "$W/scored.jsonl"
step "4/9 prompt features"
python -m shallowfrontier.cli features --model "$BASE" --pool "$POOL" ${TPL[@]+"${TPL[@]}"} --out "$W/features.npy"
step "5/9 grouping"
python -m shallowfrontier.cli group --features "$W/features.npy" --out "$W/groups.json"
step "6/9 noise measurement"
python -m shallowfrontier.cli noise --model "$W/calibration_model" --tokenizer "$BASE" --pool "$POOL" --calibration "$W/calibration.jsonl" \
  --groups "$W/groups.json" --k "$K" --batch "$B" --seed "$SEED" --max-length "$LEN" ${TPL[@]+"${TPL[@]}"} --out "$W/noise.json"
step "7/9 selection"
python -m shallowfrontier.cli select --pool "$POOL" --scores "$W/scored.jsonl" --groups "$W/groups.json" --noise "$W/noise.json" \
  --k "$K" --batch "$B" --out "$W/selected.jsonl"
step "8/9 final training"
python scripts/train.py --model "$BASE" --data "$W/selected.jsonl" --out "$W/final_model" --lr "$LR" --batch "$B" --max-length "$LEN" --seed "$SEED"
step "9/9 evaluation ($EVAL)"
if [ "$EVAL" = math ] || [ "$EVAL" = mixed ]; then
  (cd eval/math_evaluation && bash sh/eval.sh alpaca "$W/final_model" "$W/eval_math" 16 1)       # mean@16, temperature 1
fi
if [ "$EVAL" = med ] || [ "$EVAL" = mixed ]; then
  python eval/medeval/mcf_eval.py --model "$W/final_model" --out "$W/eval_med_mcf.json"
fi
step "done: $W"
