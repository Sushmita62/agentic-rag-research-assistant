from pathlib import Path

from app import storage
from app.storage import Paper, Chunk, get_engine, session, reset_engine


def test_roundtrip(tmp_path: Path):
    reset_engine()
    get_engine(tmp_path / "t.db")

    with session() as s:
        p = Paper(id="abc123def456", title="Attention Is All You Need",
                  filename="1706.03762.pdf", num_pages=15)
        p.authors = ["Vaswani", "Shazeer"]
        s.add(p)
        s.add(Chunk(id="abc123def456_001_00", paper_id=p.id, page=1,
                    section="Abstract", text="The dominant sequence...",
                    token_count=42, embedding_model="bge-small-en-v1.5"))

    with session() as s:
        p = s.get(Paper, "abc123def456")
        assert p.title.startswith("Attention")
        assert p.authors == ["Vaswani", "Shazeer"]
        assert len(p.chunks) == 1
        assert p.chunks[0].section == "Abstract"

    reset_engine()
