import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATH_DIR = REPO_ROOT / "eval" / "math_evaluation"
MED_DIR = REPO_ROOT / "eval" / "medeval"
RESULTS_ROOT = REPO_ROOT / "eval" / "results"

MATH_DATASETS = {
    "math_oai": "Math-OAI",
    "minerva_math": "Minerva",
    "olympiadbench": "OlympiadBench",
    "aime24": "AIME24",
    "amc23": "AMC23",
}


def run_math(model_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash",
        "sh/eval.sh",
        "alpaca",
        str(model_path),
        str(output_dir),
        "16",
        "1",
    ]

    print("\n" + "=" * 70)
    print("Running Math evaluation")
    print("=" * 70)

    subprocess.run(cmd, cwd=MATH_DIR, check=True)


def collect_math(output_dir: Path):
    results = {}

    for key, display_name in MATH_DATASETS.items():
        path = output_dir / f"{key}_metrics.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing Math result: {path}")

        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        score = float(obj["mean_acc"])

        results[key] = {
            "name": display_name,
            "score": score,
        }

    results["avg"] = sum(
        results[key]["score"] for key in MATH_DATASETS
    ) / len(MATH_DATASETS)

    return results


def run_medical(model_path: Path, output_path: Path):
    sys.path.insert(0, str(MED_DIR))

    import ray
    from vllm import LLM, SamplingParams
    from vllm_medical_test import load_jsonl, test_dataset

    datasets = {
        "medqa": MED_DIR / "test_data" / "medqa_test.jsonl",
        "mmlu": MED_DIR / "test_data" / "mmlu_medical_test.jsonl",
        "medmcqa": MED_DIR / "test_data" / "medmcqa_test.jsonl",
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=10,
    )

    print("\n" + "=" * 70)
    print("Running Medical evaluation")
    print("=" * 70)

    ray.init(ignore_reinit_error=True)

    try:
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=1,
        )

        try:
            results = {}

            for dataset_name, dataset_path in datasets.items():
                print(f"\nTesting {dataset_name}...")

                data = load_jsonl(str(dataset_path))
                accuracy = test_dataset(
                    llm,
                    data,
                    dataset_name,
                    sampling_params,
                )

                results[dataset_name] = {
                    "accuracy": accuracy,
                    "total_samples": len(data),
                }

                print(
                    f"{dataset_name}: "
                    f"{accuracy:.4f} "
                    f"({int(accuracy * len(data))}/{len(data)})"
                )

        finally:
            del llm
            gc.collect()

    finally:
        ray.shutdown()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def build_summary(name, model_path, math_results, med_results):
    med_scores = {
        key: float(value["accuracy"]) * 100.0
        for key, value in med_results.items()
    }

    med_avg = sum(med_scores.values()) / len(med_scores)
    math_avg = math_results["avg"]

    macro = (math_avg + med_avg) / 2.0
    worst_domain = min(math_avg, med_avg)

    return {
        "name": name,
        "model": str(model_path),
        "math": {
            "math_oai": math_results["math_oai"]["score"],
            "minerva_math": math_results["minerva_math"]["score"],
            "olympiadbench": math_results["olympiadbench"]["score"],
            "aime24": math_results["aime24"]["score"],
            "amc23": math_results["amc23"]["score"],
            "avg": math_avg,
        },
        "medical": {
            "medqa": med_scores["medqa"],
            "mmlu_medical": med_scores["mmlu"],
            "medmcqa": med_scores["medmcqa"],
            "avg": med_avg,
        },
        "macro": macro,
        "worst_domain": worst_domain,
    }


def print_summary(summary):
    m = summary["math"]
    d = summary["medical"]

    print("\n" + "=" * 70)
    print(f"Evaluation Summary: {summary['name']}")
    print("=" * 70)

    print("\nMath")
    print(f"  Math-OAI       {m['math_oai']:8.2f}")
    print(f"  Minerva        {m['minerva_math']:8.2f}")
    print(f"  OlympiadBench  {m['olympiadbench']:8.2f}")
    print(f"  AIME24         {m['aime24']:8.2f}")
    print(f"  AMC23          {m['amc23']:8.2f}")
    print(f"  Math Avg       {m['avg']:8.2f}")

    print("\nMedical")
    print(f"  MedQA          {d['medqa']:8.2f}")
    print(f"  MMLU-med       {d['mmlu_medical']:8.2f}")
    print(f"  MedMCQA        {d['medmcqa']:8.2f}")
    print(f"  Med Avg        {d['avg']:8.2f}")

    print("\nOverall")
    print(f"  Macro Avg      {summary['macro']:8.2f}")
    print(f"  Worst Domain   {summary['worst_domain']:8.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    experiment_dir = RESULTS_ROOT / args.name
    math_dir = experiment_dir / "math"
    med_output = experiment_dir / "medical" / "results.json"
    summary_path = experiment_dir / "summary.json"

    run_math(model_path, math_dir)
    math_results = collect_math(math_dir)

    med_results = run_medical(model_path, med_output)

    summary = build_summary(
        args.name,
        model_path,
        math_results,
        med_results,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
