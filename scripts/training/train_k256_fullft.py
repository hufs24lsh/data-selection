#!/usr/bin/env python3

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from instdiff import train_v2


DATA = ROOT / "results/selections/k256_selected.jsonl"

BASE = Path(
    os.environ.get(
        "SHALLOWFRONTIER_BASE_MODEL",
        str(ROOT / "models/Qwen2.5-7B"),
    )
).expanduser()

OUT = Path(
    os.environ.get(
        "SHALLOWFRONTIER_OUTPUT_DIR",
        str(ROOT / "outputs/k256/model"),
    )
).expanduser()


EXPECTED_DATA_SHA = (
    "61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22"
)


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def enum_value(x):
    return getattr(x, "value", x)


# ------------------------------------------------------------
# Required two-GPU execution topology
# ------------------------------------------------------------

require(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1",
    "CUDA_VISIBLE_DEVICES must be 0,1",
)

require(
    os.environ.get("WORLD_SIZE") == "1",
    "WORLD_SIZE must be 1",
)

for key in [
    "LOCAL_RANK",
    "RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
]:
    require(
        os.environ.get(key) in (None, ""),
        f"{key} must be unset",
    )

require(
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    == "expandable_segments:True",
    "allocator config mismatch",
)

require(
    torch.cuda.is_available(),
    "CUDA unavailable",
)

require(
    torch.cuda.device_count() == 2,
    "exactly two visible GPUs required",
)

require(
    not torch.backends.cuda.matmul.allow_tf32,
    "CUDA matmul TF32 must remain disabled",
)

require(
    torch.get_float32_matmul_precision() == "highest",
    "float32 matmul precision must be highest",
)


# ------------------------------------------------------------
# Input integrity
# ------------------------------------------------------------

require(DATA.is_file(), f"missing data: {DATA}")
require(BASE.is_dir(), f"missing base: {BASE}")
require(
    sha256_file(DATA) == EXPECTED_DATA_SHA,
    "selected data SHA mismatch",
)

rows = sum(
    1
    for line in DATA.open(encoding="utf-8")
    if line.strip()
)

require(
    rows == 2000,
    f"training rows != 2000: {rows}",
)


# ------------------------------------------------------------
# Fresh output only
# ------------------------------------------------------------

require(
    not OUT.exists(),
    f"refusing to overwrite {OUT}",
)

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------
# Exact T1 Full-FT recipe; selected 2K is the only change.
# tf32 intentionally omitted.
# ------------------------------------------------------------

config = {
    "model_name_or_path": str(BASE),
    "data_path": str(DATA),
    "output_dir": str(OUT),
    "model_max_length": 2048,
    "per_device_train_batch_size": 1,
    "global_batch_size": 64,
    "num_train_epochs": 3.0,
    "learning_rate": 2e-5,
    "mode": "sft",
    "use_lora": False,
    "bf16": True,
    "optim": "adamw_bnb_8bit",
    "gradient_checkpointing": True,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "report_to": [],
}

require(
    "tf32" not in config,
    "tf32 must be omitted",
)


# ------------------------------------------------------------
# Validate resolved TrainingArguments against the matched training recipe.
# ------------------------------------------------------------

OriginalTrainingArguments = train_v2.TrainingArguments


class AsymTrainingArguments(OriginalTrainingArguments):
    def __init__(self, *args, **kwargs):
        require(
            kwargs.get("tf32", None) is None,
            "TrainingArguments input tf32 must be None",
        )

        super().__init__(*args, **kwargs)

        require(
            self.tf32 is None,
            "TrainingArguments.tf32 must resolve to None",
        )

        require(
            not torch.backends.cuda.matmul.allow_tf32,
            "TF32 became enabled",
        )

        require(
            torch.get_float32_matmul_precision() == "highest",
            "float32 matmul precision changed",
        )


train_v2.TrainingArguments = AsymTrainingArguments


record = {
    "study": "Asymmetric Shallow InstructDiff v3",
    "stage": "primary K256 Full-FT",
    "seed": 42,
    "K": 256,
    "method": {
        "DeltaNLL": "raw prefix-256",
        "DeltaH": "drift-corrected 128->256",
    },
    "base_model": str(BASE),
    "training_data": str(DATA),
    "training_data_rows": rows,
    "training_data_sha256": sha256_file(DATA),
    "fresh_from_original_base": True,
    "training_recipe": config,
    "difference_from_authoritative_T1":
        "Training subset only; all Full-FT settings matched.",
    "execution_fidelity": {
        "cuda_visible_devices": "0,1",
        "world_size": 1,
        "single_process": True,
        "device_map": "auto",
        "tf32_config_key": "omitted",
        "expected_training_arguments_tf32": None,
        "expected_cuda_matmul_allow_tf32": False,
        "expected_float32_matmul_precision": "highest",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    },
}

(
    OUT
    / "shallowfrontier_k256_training_config.json"
).write_text(
    json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print(
    "===== ASYMMETRIC SHALLOW K256 FULL-FT =====",
    flush=True,
)

print(
    json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
    ),
    flush=True,
)


# ------------------------------------------------------------
# Full-FT
# ------------------------------------------------------------

train_v2.train(
    **config
)


# ------------------------------------------------------------
# Post-training fidelity validation
# ------------------------------------------------------------

args_path = OUT / "training_args.bin"

require(
    args_path.is_file(),
    "training completed without training_args.bin",
)

final_args = torch.load(
    args_path,
    map_location="cpu",
    weights_only=False,
)

expected_final = {
    "model_max_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "num_train_epochs": 3.0,
    "learning_rate": 2e-5,
    "bf16": True,
    "tf32": None,
    "gradient_checkpointing": True,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
}

for key, wanted in expected_final.items():
    got = getattr(final_args, key)

    if isinstance(wanted, float):
        require(
            math.isclose(
                float(got),
                wanted,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            f"final {key}: {got!r} != {wanted!r}",
        )
    else:
        require(
            got == wanted,
            f"final {key}: {got!r} != {wanted!r}",
        )

require(
    enum_value(getattr(final_args, "optim"))
    in ("adamw_bnb_8bit", "adamw_bnb"),
    "final optimizer mismatch",
)

require(
    enum_value(getattr(final_args, "lr_scheduler_type"))
    == "cosine",
    "final scheduler mismatch",
)

require(
    enum_value(getattr(final_args, "save_strategy"))
    == "no",
    "final save strategy mismatch",
)

require(
    not torch.backends.cuda.matmul.allow_tf32,
    "TF32 unexpectedly enabled after training",
)

(
    OUT
    / ".TRAIN_COMPLETE"
).write_text(
    "complete\n",
    encoding="utf-8",
)

print(
    "===== ASYMMETRIC SHALLOW K256 TRAIN COMPLETE =====",
    flush=True,
)
