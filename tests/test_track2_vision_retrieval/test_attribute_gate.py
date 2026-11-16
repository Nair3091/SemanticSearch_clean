import numpy as np

from src.track2_vision_retrieval.scoring import softmax_confidence as sc


def test_color_variants_swap_only_the_color_word():
    variants = sc.color_variants("a man in a red shirt")
    assert len(variants) == 1 and len(variants[0]) == len(sc.COLORS) - 1
    assert "a man in a pink shirt" in variants[0]
    assert "a man in a red shirt" not in variants[0]


def test_color_aliases_and_no_color():
    assert "a gray car" not in sc.color_variants("a gray car")[0]       # gray == grey, not a rival
    assert "a grey car" not in sc.color_variants("a gray car")[0]
    assert sc.color_variants("a dog looking at a crab") == []
    assert len(sc.color_variants("a red shirt and blue jeans")) == 2


def test_attribute_gate_keeps_top_ranks_only():
    q = np.array([0.30, 0.20, 0.25])
    variants = np.array([[0.10, 0.20, 0.25],      # query best -> rank 1
                         [0.25, 0.26, 0.27],      # three rivals above -> rank 4
                         [0.26, 0.27, 0.10]])     # two rivals above -> rank 3
    assert sc.attribute_gate(q, variants, 3).tolist() == [1.0, 0.0, 1.0]
