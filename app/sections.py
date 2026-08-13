"""Section detection + per-page text cleanup. Heuristic, no ML."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.pdf import Page


SECTION_VOCAB = (
    r"abstract|introduction|related\s+work|background|"
    r"methods?|methodology|approach|"
    r"experiments?|evaluation|results?|findings|"
    r"discussion|analysis|"
    r"conclusions?|future\s+work|"
    r"references|bibliography|"
    r"acknowledg[e]?ments?|appendix"
)

_HEADING_RE = re.compile(
    rf"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?({SECTION_VOCAB})\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    name: str
    start_page: int
    end_page: int
    text: str                                    # cleaned, page-joined
    page_starts: list[tuple[int, int]] = field(default_factory=list)
    # (char_offset_in_text, page_num) sorted by offset; first is always (0, start_page)


def clean_text(s: str) -> str:
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)      # dehyphenate within a segment
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def page_at(page_starts: list[tuple[int, int]], offset: int) -> int:
    page = page_starts[0][1]
    for off, p in page_starts:
        if off <= offset:
            page = p
        else:
            break
    return page


def _build_section(name: str, lines: list[tuple[int, str]]) -> Section | None:
    """lines = [(page, raw_line), ...]. Group by page, clean per page, join."""
    if not lines:
        return None
    # group contiguous same-page runs
    segments: list[tuple[int, str]] = []
    cur_page = lines[0][0]
    buf: list[str] = []
    for page, line in lines:
        if page != cur_page:
            cleaned = clean_text("\n".join(buf))
            if cleaned:
                segments.append((cur_page, cleaned))
            buf, cur_page = [], page
        buf.append(line)
    cleaned = clean_text("\n".join(buf))
    if cleaned:
        segments.append((cur_page, cleaned))
    if not segments:
        return None

    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []
    offset = 0
    for page, seg in segments:
        page_starts.append((offset, page))
        parts.append(seg)
        offset += len(seg) + 2                   # "\n\n" separator
    text = "\n\n".join(parts)
    return Section(name, segments[0][0], segments[-1][0], text, page_starts)


def detect_sections(pages: list[Page]) -> list[Section]:
    if not pages:
        return []
    lines: list[tuple[int, str]] = []
    for p in pages:
        for line in p.text.splitlines():
            lines.append((p.num, line))

    boundaries: list[tuple[int, str]] = []      # (line_idx, name)
    for i, (_, line) in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            boundaries.append((i, m.group(1).lower().replace(" ", "_")))

    if not boundaries:
        sec = _build_section("body", lines)
        return [sec] if sec else []

    out: list[Section] = []
    for j, (start_idx, name) in enumerate(boundaries):
        end_idx = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        sec = _build_section(name, lines[start_idx + 1 : end_idx])
        if sec:
            out.append(sec)
    return out
