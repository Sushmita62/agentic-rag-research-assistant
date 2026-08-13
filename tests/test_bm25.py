from pathlib import Path

from app.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("BERT achieved 92.3% on CIFAR-10!") == [
        "bert", "achieved", "92", "3", "on", "cifar", "10",
    ]


def test_search_finds_exact_term(tmp_path: Path):
    idx = BM25Index(path=tmp_path / "bm25.pkl")
    idx.rebuild([
        (10, "The transformer architecture uses self-attention."),
        (11, "Convolutional networks dominate image recognition."),
        (12, "BERT was pretrained on BookCorpus and Wikipedia."),
    ])
    hits = idx.search("BERT", k=3)
    assert hits[0][0] == 12                                    # BERT chunk wins
    assert hits[0][1] > 0


def test_persist_and_reload(tmp_path: Path):
    p = tmp_path / "bm25.pkl"
    idx = BM25Index(path=p)
    idx.rebuild([(1, "cats sleep on rugs"), (2, "dogs fetch balls")])
    idx.save()

    idx2 = BM25Index(path=p)
    hits = idx2.search("cats", k=2)
    assert hits[0][0] == 1


def test_empty_search_returns_empty(tmp_path: Path):
    idx = BM25Index(path=tmp_path / "b.pkl")
    assert idx.search("anything") == []
