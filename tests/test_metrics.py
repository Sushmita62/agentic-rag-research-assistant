"""Pure math on the metrics functions. No LLM, no graph."""
from eval.metrics import aggregate, score_question


def test_supported_question_scored_correctly():
    bench = {"id": "q", "question": "?", "expected_verdict": "SUPPORTED",
             "gold_pages": [3, 4], "must_contain": ["attention", "softmax"]}
    result = {
        "answer": "Attention uses softmax over dot products.",
        "abstained": False,
        "claims": [{"text": "c", "citations": [{"chunk_id": "cA"}, {"chunk_id": "cB"}]}],
        "evidence": [
            {"chunk_id": "cA", "page": 3, "section": "x", "title": "t"},
            {"chunk_id": "cB", "page": 4, "section": "x", "title": "t"},
            {"chunk_id": "cC", "page": 9, "section": "x", "title": "t"},
        ],
        "total_latency_ms": 1000,
    }
    m = score_question(bench, result)
    assert m.verdict_correct
    assert m.recall_at_5 == 1.0
    assert m.citation_precision == 1.0                # both cites on gold pages
    assert m.must_contain_hit_ratio == 1.0


def test_should_have_abstained_but_answered():
    bench = {"id": "q", "question": "?", "expected_verdict": "ABSTAIN",
             "gold_pages": [], "must_contain": []}
    result = {"answer": "some answer", "abstained": False, "claims": [],
              "evidence": [], "total_latency_ms": 500}
    m = score_question(bench, result)
    assert not m.verdict_correct


def test_recall_zero_when_wrong_pages():
    bench = {"id": "q", "question": "?", "expected_verdict": "SUPPORTED",
             "gold_pages": [7], "must_contain": []}
    result = {"answer": "x", "abstained": False,
              "claims": [{"text": "c", "citations": [{"chunk_id": "c1"}]}],
              "evidence": [{"chunk_id": "c1", "page": 12, "section": "x", "title": "t"}],
              "total_latency_ms": 100}
    m = score_question(bench, result)
    assert m.recall_at_5 == 0.0
    assert m.citation_precision == 0.0


def test_aggregate_averages_and_percentiles():
    from eval.metrics import QuestionMetrics
    per_q = [
        QuestionMetrics("a", True, 1.0, 1.0, 1.0, 100),
        QuestionMetrics("b", True, 0.0, 0.5, 0.5, 200),
        QuestionMetrics("c", False, None, None, None, 300),
    ]
    a = aggregate(per_q)
    assert a.n == 3
    assert a.abstain_accuracy == 2 / 3
    assert a.recall_at_5 == 0.5                       # (1.0 + 0.0) / 2
    assert a.citation_precision == 0.75
    assert a.p50_latency_ms == 200
