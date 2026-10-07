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


def test_training_scripts_have_no_stale_t1_args_reference():
    for name in ["train_k256_fullft.py", "train_k128_fullft.py"]:
        p = ROOT / "scripts" / "training" / name
        assert "ORIGINAL_T1_ARGS" not in p.read_text()


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


if __name__ == "__main__":
    tests = [
        test_training_scripts_have_no_stale_t1_args_reference,
        test_selected_dataset_integrity,
        test_frozen_performance_results,
        test_readme_local_images_exist,
        test_no_personal_absolute_paths,
        test_python_syntax,
    ]

    for test in tests:
        test()
        print("PASS", test.__name__)

    print("ALL RELEASE INTEGRITY TESTS PASSED")
