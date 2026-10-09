from pathlib import Path
import hashlib
import json
import py_compile
import re

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_training_scripts_validate_resolved_arguments():
    for name in ["train_k256_fullft.py", "train_k128_fullft.py"]:
        p = ROOT / "scripts" / "training" / name
        source = p.read_text()
        assert "OriginalTrainingArguments = train_v2.TrainingArguments" in source
        assert "train_v2.TrainingArguments = AsymTrainingArguments" in source


def test_selected_dataset_integrity():
    expected = {
        "k256_selected.jsonl":
            "61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22",
        "k128_selected.jsonl":
            "c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2",
    }

    for name, wanted_sha in expected.items():
        p = ROOT / "results" / "selections" / name
        assert p.is_file()
        assert sha256(p) == wanted_sha
        assert sum(1 for line in p.open() if line.strip()) == 2000


def test_frozen_performance_results():
    expected = {
        "k256_summary.json": 46.68940115767946,
        "k128_summary.json": 46.60570178330249,
    }

    for name, wanted_macro in expected.items():
        obj = json.loads(
            (ROOT / "results" / "performance" / name).read_text()
        )
        assert abs(obj["macro"] - wanted_macro) < 1e-12


def test_readme_local_images_exist():
    text = (ROOT / "README.md").read_text()

    refs = re.findall(
        r'<img[^>]+src=["\']([^"\']+)["\']',
        text,
        re.I,
    )
    refs += re.findall(
        r'!\[[^\]]*\]\(([^)]+)\)',
        text,
    )

    for ref in refs:
        if ref.startswith(("http://", "https://")):
            continue
        assert (ROOT / ref).is_file(), f"missing README asset: {ref}"


def test_no_personal_absolute_paths():
    suffixes = {
        ".py", ".sh", ".json", ".yaml", ".yml",
        ".md", ".txt", ".csv", ".toml",
    }

    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix.lower() not in suffixes:
            continue
        if "results/manifests" in p.as_posix():
            continue

        text = p.read_text(errors="ignore")
        personal_home = "/home/" + "hufs"
        conda_marker = "mini" + "conda3"
        assert personal_home not in text, str(p)
        assert conda_marker not in text, str(p)


def test_python_syntax():
    for base in ["scripts", "src", "eval"]:
        for p in (ROOT / base).rglob("*.py"):
            py_compile.compile(str(p), doraise=True)


def test_selector_expected_hashes_match_published_sets():
    text = (
        ROOT / "scripts" / "selection" / "select_from_scores.py"
    ).read_text()

    expected = {
        256:
            "61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22",
        128:
            "c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2",
    }

    for wanted in expected.values():
        assert wanted in text



def test_official_eval_guards_match_public_evaluators():
    evaluator_paths = {
        "EXPECTED_RUN_ALL_SHA":
            ROOT / "eval" / "run_all_eval.py",
        "EXPECTED_MATH_SHA":
            ROOT / "eval" / "math_evaluation" / "sh" / "eval.sh",
        "EXPECTED_MED_SHA":
            ROOT / "eval" / "medeval" / "vllm_medical_test.py",
    }

    for wrapper_name in [
        "eval_k256_official.sh",
        "eval_k128_official.sh",
    ]:
        wrapper = (
            ROOT / "scripts" / "evaluation" / wrapper_name
        ).read_text()

        for variable, evaluator in evaluator_paths.items():
            wanted = sha256(evaluator)
            pattern = rf'{variable}="([0-9a-f]{{64}})"'
            match = re.search(pattern, wrapper)

            assert match is not None, (
                f"{wrapper_name}: missing {variable}"
            )
            assert match.group(1) == wanted, (
                f"{wrapper_name}: stale {variable}: "
                f"{match.group(1)} != {wanted}"
            )


def test_public_reproduction_paths_and_guards():
    energy_script = (
        ROOT / "scripts" / "energy" / "measure_idle_baseline.py"
    ).read_text()

    assert (
        'PROTOCOL = ROOT / '
        '"results/energy/archive/energy_measurement_protocol.json"'
        in energy_script
    )

    protocol = (
        ROOT / "results" / "energy" / "archive"
        / "energy_measurement_protocol.json"
    )

    assert protocol.is_file()
    assert sha256(protocol) == (
        "b52f3a4a2dc83254ff63f0c16f6553934c7b3bf94949e2fa7bdd2a9571d9e350"
    )

    eval_requirements = (
        ROOT / "requirements" / "eval.txt"
    ).read_text()

    for token in [
        "sympy==1.12",
        "antlr4-python3-runtime==4.11.1",
        "-e ./eval/math_evaluation/latex2sympy",
    ]:
        assert token in eval_requirements

    readme = (ROOT / "README.md").read_text()

    for token in [
        "Run all commands below from the **repository root**.",
        "CUDA_VISIBLE_DEVICES=0,1",
        "WORLD_SIZE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "bash scripts/evaluation/eval_k256_official.sh",
        "bash scripts/evaluation/eval_k128_official.sh",
        "47.10 - 0.50 = 46.60",
    ]:
        assert token in readme

    plot = (
        ROOT / "scripts" / "plots"
        / "fig3_energy_breakdown.py"
    ).read_text()

    assert 'save(fig, "figs/stagewise_energy.png")' in plot


if __name__ == "__main__":
    tests = [
        test_training_scripts_validate_resolved_arguments,
        test_selected_dataset_integrity,
        test_frozen_performance_results,
        test_readme_local_images_exist,
        test_no_personal_absolute_paths,
        test_python_syntax,
        test_selector_expected_hashes_match_published_sets,
        test_official_eval_guards_match_public_evaluators,
        test_public_reproduction_paths_and_guards,
    ]

    for test in tests:
        test()
        print("PASS", test.__name__)

    print("ALL RELEASE INTEGRITY TESTS PASSED")
