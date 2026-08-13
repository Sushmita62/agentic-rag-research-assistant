"""Token-window chunker. Section-aware, page-preserving via section.page_starts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import tiktoken

from app.sections import Section, page_at

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunked:
    id: str
    paper_id: str
    page: int
    section: str
    text: str
    token_count: int


def chunk_document(
    sections: list[Section],
    paper_id: str,
    target: int = 500,
    overlap: int = 75,
    min_tokens: int = 40,
) -> list[Chunked]:
    stride = target - overlap
    if stride <= 0:
        raise ValueError("overlap must be < target")

    out: list[Chunked] = []
    seq_by_page: dict[int, int] = defaultdict(int)

    for sec in sections:
        tokens = _ENC.encode(sec.text)
        n = len(tokens)
        if n < min_tokens:
            continue
        for start in range(0, n, stride):
            piece = tokens[start : start + target]
            if len(piece) < min_tokens:
                break
            text = _ENC.decode(piece)
            char_start = len(_ENC.decode(tokens[:start])) if start else 0
            page = page_at(sec.page_starts, char_start) if sec.page_starts else sec.start_page
            seq = seq_by_page[page]
            seq_by_page[page] += 1
            out.append(Chunked(
                id=f"{paper_id}_{page:03d}_{seq:02d}",
                paper_id=paper_id,
                page=page,
                section=sec.name,
                text=text,
                token_count=len(piece),
            ))
            if start + target >= n:
                break
    return out
