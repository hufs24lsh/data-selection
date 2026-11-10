import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_mixed_pools", ROOT / "scripts" / "make_mixed_pools.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_build_appends_domain_and_follows_the_order(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "m.jsonl").write_text('{"instruction":"x","response":"1"}\n{"instruction":"é","response":"2"}\n', encoding="utf-8")
    (tmp_path / "data" / "d.jsonl").write_text('{"instruction":"y","response":"3"}\n', encoding="utf-8")
    order = dict(sources={"m": "data/m.jsonl", "d": "data/d.jsonl"}, domain_names={"m": "math", "d": "med"}, domains="dmm", indices=[0, 1, 0])
    blob = M.build(order, tmp_path)
    assert blob.decode("utf-8").splitlines() == ['{"instruction": "y", "response": "3", "domain": "med"}',
                                                 '{"instruction": "é", "response": "2", "domain": "math"}',
                                                 '{"instruction": "x", "response": "1", "domain": "math"}']


def test_order_files_are_consistent():
    for f in (ROOT / "data" / "mixed").glob("*.order.json"):
        o = json.loads(f.read_text())
        assert len(o["domains"]) == len(o["indices"]) and set(o["domains"]) <= {"m", "d"} and len(o["sha256"]) == 64
        for d in "md":                                            # every source line is used at most once
            idx = [i for x, i in zip(o["domains"], o["indices"]) if x == d]
            assert len(idx) == len(set(idx))
