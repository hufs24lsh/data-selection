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
        "?boardCategoryId=39&boardId=1829260&boardMasterId=1"
    )

    with (ROOT / "results/energy/stage_markers.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        stages = {row["stage"] for row in csv.DictReader(stream)}

    assert "K128_FINAL2K_FT" in stages
    assert "K128_OFFICIAL_EVAL" in stages


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
