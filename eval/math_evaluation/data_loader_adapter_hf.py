import os

from utils import load_jsonl, lower_keys


def load_data(
    data_name,
    split,
    data_dir="./data",
):
    """
    HF adapter evaluation loader.

    For the five frozen Math benchmarks used in this project,
    evaluation data must already exist as local JSONL.

    This deliberately avoids HuggingFace `datasets` so the
    training environment does not need datasets/pyarrow/pandas.
    """

    data_file = (
        f"{data_dir}/"
        f"{data_name}/"
        f"{split}.jsonl"
    )

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            "HF adapter evaluation is frozen to local "
            f"benchmark files; missing: {data_file}"
        )

    examples = list(
        load_jsonl(data_file)
    )

    examples = [
        lower_keys(example)
        for example in examples
    ]

    if not examples:
        raise RuntimeError(
            f"Empty evaluation file: {data_file}"
        )

    if "idx" not in examples[0]:
        examples = [
            {
                "idx": i,
                **example,
            }
            for i, example in enumerate(examples)
        ]

    examples = sorted(
        examples,
        key=lambda x: x["idx"],
    )

    return examples
