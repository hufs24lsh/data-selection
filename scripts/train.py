#!/usr/bin/env python3
"""Full-parameter SFT on a JSONL subset with the paper's recipe: 3 epochs, cosine schedule, 5% warm-up, weight decay 0.01,
gradient clipping 1.0, 8-bit AdamW, bf16, gradient checkpointing, per-device batch 1 with gradient accumulation to the global batch.
The same script trains the calibration model (on the calibration subset) and the final model (on the selected subset).

  python scripts/train.py --model Qwen/Qwen2.5-7B --data selected.jsonl --out runs/final --lr 2e-5 --batch 64 --max-length 2048 --seed 42
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

RECIPE = dict(per_device_train_batch_size=1, num_train_epochs=3.0, mode="sft", use_lora=False, bf16=True, optim="adamw_bnb_8bit",
              gradient_checkpointing=True, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0, report_to=[])


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch", type=int, required=True, help="global batch size")
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("output exists (not overwritten): " + a.out)
    set_seed(a.seed)
    from instdiff.train_v2 import train
    train(**dict(RECIPE, model_name_or_path=a.model, data_path=a.data, output_dir=a.out, learning_rate=a.lr, global_batch_size=a.batch,
                 model_max_length=a.max_length, seed=a.seed))


if __name__ == "__main__":
    main()
