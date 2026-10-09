"""Canonical identifiers and official reporting factor."""

import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_reporting_data():
    with (ROOT / "results/cost/final_cost_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert [row["method"] for row in rows] == ["T1", "K256", "K128"]

    assert int(rows[-1]["scoring_processed_tokens_2models"]) == 7702010
    assert abs(float(rows[-1]["macro"]) - 46.60570178330249) < 1e-12

    meta = json.loads((ROOT / "results/energy/carbon_factor_status.json").read_text())

    assert meta["kgCO2eq_per_kWh"] == 0.4173
    assert meta["status"] == "VERIFIED_OFFICIAL_SOURCE"
    assert meta["reference_year"] == 2023
    assert meta["publication_date"] == "2025-12-18"
    assert meta["factor_boundary"] == "consumption-end"
    assert meta["source"] == (
        "https://mcee.go.kr/home/web/board/read.do"
        "?pagerOffset=530&maxPageItems=10&maxIndexPages=10"
        "&searchKey=&searchValue=&menuId=10598&orgCd="
        "&boardMasterId=939&boardCategoryId=&boardId=1829260&decorator="
    )

    with (ROOT / "results/energy/stage_markers.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        stages = {row["stage"] for row in csv.DictReader(stream)}

    assert "K128_FINAL2K_FT" in stages
    assert "K128_OFFICIAL_EVAL" in stages



def test_all_published_performance_summaries():
    expected_macro = {
        "base": 38.62887510790447,
        "full20k": 43.26664461950279,
        "random2k": 44.81374561171734,
        "t1": 47.10085981289461,
        "t2": 46.13921258053263,
        "t3": 45.82554081776559,
        "k256": 46.68940115767946,
        "k128": 46.60570178330249
    }

    math_keys = ("math_oai", "minerva_math", "olympiadbench", "aime24", "amc23")
    med_keys = ("medqa", "mmlu_medical", "medmcqa")

    for method, wanted in expected_macro.items():
        path = ROOT / "results" / "performance" / f"{method}_summary.json"
        obj = json.loads(path.read_text(encoding="utf-8"))

        math_avg = sum(obj["math"][key] for key in math_keys) / 5
        medical_avg = sum(obj["medical"][key] for key in med_keys) / 3
        macro = (math_avg + medical_avg) / 2

        assert abs(obj["math"]["avg"] - math_avg) < 1e-10
        assert abs(obj["medical"]["avg"] - medical_avg) < 1e-10
        assert abs(obj["macro"] - macro) < 1e-10
        assert abs(obj["macro"] - wanted) < 1e-10
        assert abs(obj["worst_domain"] - min(math_avg, medical_avg)) < 1e-10
        assert not obj["model"].startswith("/home/")



def test_no_deprecated_reporting_tokens():
    prohibited = (
        "true" + "-K128",
        "TRUE" + "_K128",
        "0.45" + "41",
        "UNVERIFIED" + "_PUBLIC_PROVENANCE",
    )

    data = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
    )

    paths = [ROOT / os.fsdecode(part) for part in data.split(bytes([0])) if part]

    suffixes = {
        ".py",
        ".sh",
        ".md",
        ".csv",
        ".json",
        ".txt",
        ".yml",
        ".yaml",
        ".toml",
    }

    for path in paths:
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue

        content = path.read_text(encoding="utf-8")

        for token in prohibited:
            assert token not in content, str(path)
