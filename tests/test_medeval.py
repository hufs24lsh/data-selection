import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("mcf_eval", Path(__file__).resolve().parents[1] / "eval" / "medeval" / "mcf_eval.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_released_rule_lets_a_low_variant_overwrite_the_answer():
    # ' B' is 99.7% likely, but a lower-ranked ' b' in the top-10 overwrites B's entry, so A wins under the released rule
    top = {" B": math.log(0.997), " A": math.log(0.0015), " C": math.log(0.0005), " b": math.log(0.00001)}
    assert M.released_prediction(top, " B") == ("A", "top10_option")


def test_mcf_sums_the_variants_of_a_letter():
    ids = {"A": [1, 2], "B": [3], "C": [4], "D": [5]}
    lp = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.35), 4: math.log(0.15), 5: math.log(0.1)}
    pred, scores = M.mcf_prediction(lp, ids)
    assert pred == "A" and abs(math.exp(scores["A"]) - 0.4) < 1e-12


def test_fallbacks_of_the_released_rule():
    assert M.released_prediction({" The": -0.1}, "The answer") == ("A", "text_fallback")
    assert M.released_prediction({}, "...") == ("A", "default_A")
