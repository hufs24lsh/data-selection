#!/usr/bin/env python3
"""Report or strictly verify the recorded ShallowFrontier environments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys

EXPECTED = {
    "train": {
        "python": "3.10.21",
        "packages": {
            "torch": "2.13.0+cu132",
            "transformers": "4.47.1",
            "bitsandbytes": "0.50.2",
        },
        "cuda_runtime": "13.2",
        "hardware": "2 x NVIDIA GeForce RTX 5090",
    },
    "eval": {
        "python": "3.10.21",
        "packages": {
            "torch": "2.11.0+cu130",
            "transformers": "5.17.0",
            "vllm": "0.26.0",
        },
        "cuda_runtime": "13.0",
        "hardware": "2 x NVIDIA GeForce RTX 5090",
    },
}


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_info() -> list[str] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def inspect(profile: str) -> dict:
    expected = EXPECTED[profile]
    packages = {
        name: installed_version(name)
        for name in expected["packages"]
    }

    cuda_runtime = None
    torch_version = packages.get("torch")
    if torch_version is not None:
        try:
            import torch

            cuda_runtime = torch.version.cuda
            torch_version = torch.__version__
            packages["torch"] = torch_version
        except Exception as exc:
            cuda_runtime = f"IMPORT_ERROR: {type(exc).__name__}: {exc}"

    return {
        "profile": profile,
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda_runtime": cuda_runtime,
        "gpus": gpu_info(),
        "expected": expected,
    }


def mismatches(report: dict) -> list[str]:
    expected = report["expected"]
    failures = []

    if report["python"] != expected["python"]:
        failures.append(
            f"python: {report['python']} != {expected['python']}"
        )

    for package, wanted in expected["packages"].items():
        actual = report["packages"].get(package)
        if actual != wanted:
            failures.append(f"{package}: {actual} != {wanted}")

    if report["torch_cuda_runtime"] != expected["cuda_runtime"]:
        failures.append(
            "torch CUDA runtime: "
            f"{report['torch_cuda_runtime']} != {expected['cuda_runtime']}"
        )

    gpus = report["gpus"]
    if gpus is not None:
        if len(gpus) != 2:
            failures.append(f"GPU count: {len(gpus)} != 2")
        elif not all("RTX 5090" in line for line in gpus):
            failures.append(f"GPU model mismatch: {gpus}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(EXPECTED), required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless the active environment matches exactly",
    )
    args = parser.parse_args()

    report = inspect(args.profile)
    failures = mismatches(report)
    report["mismatches"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
