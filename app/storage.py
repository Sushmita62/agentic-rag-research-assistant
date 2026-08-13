"""SQLite storage. Papers + chunks only. One file until it hurts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import ForeignKey, LargeBinary, String, Integer, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

from app.config import settings


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # sha256[:12]
    title: Mapped[str] = mapped_column(Text)
    authors_json: Mapped[str] = mapped_column(Text, default="[]")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filename: Mapped[str] = mapped_column(Text)
    num_pages: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # queued | processing | indexed | failed

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")

    @property
    def authors(self) -> list[str]:
        return json.loads(self.authors_json or "[]")

    @authors.setter
    def authors(self, v: list[str]) -> None:
        self.authors_json = json.dumps(v)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # f"{paper_id}_{page:03d}_{seq:02d}"
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(64), default="body")
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(64))
    faiss_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class EmbedCache(Base):
    __tablename__ = "embed_cache"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    vec: Mapped[bytes] = mapped_column(LargeBinary)


# ── engine / session ──────────────────────────────────────────────────────────
_engine = None


def get_engine(db_path: Path | None = None):
    global _engine
    if _engine is None:
        path = db_path or settings.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", future=True)
        Base.metadata.create_all(_engine)
    return _engine


def session() -> Iterator[Session]:
    """Use as: `with session() as s: ...` — commits on exit unless exception."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        s = Session(get_engine(), future=True)
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _ctx()


def reset_engine() -> None:
    """For tests only."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
