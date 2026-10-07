#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

ROOT="outputs/k128"
MODEL="$ROOT/model"

NAME="shallowfrontier_k128"
RESULT_DIR="eval/results/$NAME"
SUMMARY="$RESULT_DIR/summary.json"

EVAL_PY="${EVAL_PY:-python}"

# Execution-only environment fidelity:
# Math evaluator launches bare python3 internally.
# Keep evaluator code frozen and force that python3
# to resolve from the same indiff-eval environment.
# Activate the evaluation environment before running this script.


EXPECTED_RUN_ALL_SHA="4dc4b27d3749f9e3538bd5a2945c78ceb2942e668ef3c9b96931eebf6b61528a"
EXPECTED_MATH_SHA="21e29539517f77a6b8cb46ebbf21932fb9ad21bd99e79068bfddc1bd04748d75"
EXPECTED_MED_SHA="b0c67c3e626cc04f350226841a4474785c86db6566a1af9eb3be464a3743b7c7"

echo "===== ASYMMETRIC K128 OFFICIAL EVAL PREFLIGHT ====="

[ -f "$MODEL/.TRAIN_COMPLETE" ] || {
    echo "ERROR: K128 training is not complete"
    exit 1
}

[ -f "$MODEL/training_args.bin" ] || {
    echo "ERROR: training_args.bin missing"
    exit 1
}

[ -f "$MODEL/shallowfrontier_k128_training_config.json" ] || {
    echo "ERROR: K128 training provenance missing"
    exit 1
}

[ -x "$EVAL_PY" ] || {
    echo "ERROR: indiff-eval python missing"
    exit 1
}

for f in   eval/run_all_eval.py   eval/math_evaluation/sh/eval.sh   eval/medeval/vllm_medical_test.py
do
    [ -f "$f" ] || {
        echo "ERROR: missing evaluation file: $f"
        exit 1
    }
done

ACTUAL_RUN_ALL_SHA="$(sha256sum eval/run_all_eval.py | awk '{print $1}')"
ACTUAL_MATH_SHA="$(sha256sum eval/math_evaluation/sh/eval.sh | awk '{print $1}')"
ACTUAL_MED_SHA="$(sha256sum eval/medeval/vllm_medical_test.py | awk '{print $1}')"

[ "$ACTUAL_RUN_ALL_SHA" = "$EXPECTED_RUN_ALL_SHA" ] || {
    echo "ERROR: run_all_eval.py SHA mismatch"
    exit 1
}

[ "$ACTUAL_MATH_SHA" = "$EXPECTED_MATH_SHA" ] || {
    echo "ERROR: math evaluator SHA mismatch"
    exit 1
}

[ "$ACTUAL_MED_SHA" = "$EXPECTED_MED_SHA" ] || {
    echo "ERROR: medical evaluator SHA mismatch"
    exit 1
}

if [ -e "$RESULT_DIR" ]; then
    echo "ERROR: refusing to overwrite existing evaluation:"
    echo "$RESULT_DIR"
    exit 1
fi

echo "MODEL=$MODEL"
echo "NAME=$NAME"

echo
echo "===== GPU STATE ====="
nvidia-smi   --query-gpu=index,memory.used,memory.total,utilization.gpu   --format=csv,noheader,nounits

export CUDA_VISIBLE_DEVICES=0,1

unset WORLD_SIZE LOCAL_RANK RANK MASTER_ADDR MASTER_PORT || true

echo
echo "===== OFFICIAL MATH + MEDICAL EVAL ====="

"$EVAL_PY" -u eval/run_all_eval.py   --model "$MODEL"   --name "$NAME"

RC=$?

echo "ASYM_TRUE_K128_OFFICIAL_EVAL_EXIT_CODE=$RC"

[ "$RC" -eq 0 ] || exit "$RC"

[ -f "$SUMMARY" ] || {
    echo "ERROR: evaluation summary missing"
    exit 1
}

"$EVAL_PY" - "$SUMMARY" "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
model = Path(sys.argv[2]).resolve()

obj = json.loads(
    summary.read_text(
        encoding="utf-8"
    )
)

saved_model = Path(
    obj["model"]
).resolve()

if saved_model != model:
    raise RuntimeError(
        f"Summary model mismatch: "
        f"{saved_model} != {model}"
    )

required_math = {
    "math_oai",
    "minerva_math",
    "olympiadbench",
    "aime24",
    "amc23",
    "avg",
}

required_med = {
    "medqa",
    "mmlu_medical",
    "medmcqa",
    "avg",
}

assert required_math.issubset(
    obj["math"]
)

assert required_med.issubset(
    obj["medical"]
)

assert "macro" in obj
assert "worst_domain" in obj

print(
    "Math avg =",
    obj["math"]["avg"],
)

print(
    "Medical avg =",
    obj["medical"]["avg"],
)

print(
    "Macro =",
    obj["macro"],
)

print(
    "Worst domain =",
    obj["worst_domain"],
)
PY

cp   "$SUMMARY"   "$ROOT/official_eval_summary.json"

printf '%s\n'   "$(realpath "$MODEL")"   > "$ROOT/official_eval_model_path.txt"

touch   "$ROOT/.OFFICIAL_EVAL_COMPLETE"

echo
echo "===== EAR OFFICIAL EVAL COMPLETE ====="
cat "$ROOT/official_eval_summary.json"
