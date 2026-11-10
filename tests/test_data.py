from shallowfrontier.data import match_to_pool, prompt_of


def test_match_to_pool_is_a_multiset():
    pool = [dict(instruction="a", response="x"), dict(instruction="a", response="x"), dict(instruction="b", output="y")]
    idx, miss = match_to_pool([dict(instruction="a", response="x"), dict(instruction="b", response="y"),
                               dict(instruction="a", response="x"), dict(instruction="c", response="z")], pool)
    assert idx == [0, 2, 1] and miss == 1


def test_prompt_templates():
    assert prompt_of(dict(instruction="Q")).endswith("### Instruction:\nQ\n\n### Response:")
    assert prompt_of(dict(instruction="Q", input="I"), "{instruction}|{input}") == "Q|I"
