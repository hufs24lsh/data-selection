"""ShallowFrontier: label-free, energy-efficient data selection for LLM fine-tuning.

One shallow calibration run, prompt-only grouping, a per-group gradient noise scale B measured until the decision is settled,
shares proportional to n_g / B_g, and a route per group: B_g < b (batch size) -> lowest base-entropy rows, otherwise
InstructDiff's contrastive-entropy order.
"""

__version__ = "1.0.0"
