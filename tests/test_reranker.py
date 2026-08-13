"""First run downloads bge-reranker-base (~1.1GB). Cached after."""
from app.reranker import Reranker


def test_reranker_picks_relevant_over_irrelevant():
    r = Reranker()
    candidates = [
        (1, "The stock market closed higher yesterday on strong earnings reports."),
        (2, "The transformer architecture uses multi-head self-attention to model long-range dependencies."),
        (3, "Cats are known for sleeping up to 16 hours a day."),
    ]
    ranked = r.rerank("what is self-attention in transformers", candidates, top_k=3)
    assert ranked[0][0] == 2                                   # transformer chunk wins
    # its score should be clearly higher than the noise
    assert ranked[0][1] > ranked[1][1]


def test_reranker_empty():
    assert Reranker().rerank("anything", [], top_k=5) == []
