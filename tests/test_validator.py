"""Pure logic. No LLM, no network."""
from app.answer import Citation, Claim, ComposedAnswer
from app.validator import validate, _overlap_ratio


def test_overlap_ratio_stopwords_ignored():
    # "the model" vs "the sky" — both share "the" (stopword), but no content match
    assert _overlap_ratio("the model", "the sky") == 0.0


def test_supported_when_citation_matches():
    answer = ComposedAnswer(
        summary="s",
        claims=[Claim(text="The transformer achieved 92 percent accuracy.",
                      citations=[Citation(chunk_id="c1")])],
    )
    chunks = {"c1": "We report that the transformer model reached 92 percent accuracy on the benchmark."}
    r = validate(answer, chunks)
    assert r.per_claim[0].support == "SUPPORTED"
    assert not r.should_abstain


def test_unsupported_when_citation_unrelated():
    answer = ComposedAnswer(
        summary="s",
        claims=[Claim(text="The transformer achieved 92 percent accuracy on ImageNet.",
                      citations=[Citation(chunk_id="c1")])],
    )
    chunks = {"c1": "Cats commonly sleep for over sixteen hours per day."}
    r = validate(answer, chunks)
    assert r.per_claim[0].support == "UNSUPPORTED"
    assert r.should_abstain


def test_abstain_when_majority_unsupported():
    answer = ComposedAnswer(
        summary="s",
        claims=[
            Claim(text="X worked well.", citations=[Citation(chunk_id="c1")]),
            Claim(text="Y worked well.", citations=[Citation(chunk_id="c2")]),
            Claim(text="Z worked well.", citations=[Citation(chunk_id="c3")]),
        ],
    )
    chunks = {"c1": "totally unrelated aaa bbb ccc",
              "c2": "totally unrelated ddd eee fff",
              "c3": "X worked well in experiments"}
    r = validate(answer, chunks)
    assert r.should_abstain
    assert r.supported_ratio < 0.5
