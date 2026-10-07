"""Repo walking and chunking (functions/classes for code, headings for markdown)."""
from __future__ import annotations

import ast
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

from .schemas import Chunk

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "node_modules", "build", "dist", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", ".next", "coverage", ".repopilot", ".pytest_cache", ".tox",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "uv.lock", "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
}
LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".md": "markdown", ".mdx": "markdown",
    ".go": "go", ".java": "java", ".rb": "ruby", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cs": "csharp", ".php": "php", ".sh": "shell",
    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
}
MAX_FILE_BYTES = 500 * 1024
MAX_CHUNK_LINES = 120
WINDOW_LINES = 60
OVERLAP = 10


class EmptyRepoError(RuntimeError):
    """Raised when a repo has no indexable files."""


# ---------------------------------------------------------------- walking

def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield indexable files, skipping junk dirs, lockfiles, and oversized files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            if name in SKIP_FILES:
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in LANG_BY_EXT:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > MAX_FILE_BYTES:
                continue
            yield path


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    if b"\0" in data[:1024]:  # binary
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ---------------------------------------------------------------- chunk helpers

def _mk(rel: str, lang: str, lines: list[str], start: int, end: int,
        symbol: str, ctype: str) -> Chunk:
    return Chunk(
        text="\n".join(lines[start - 1:end]),
        file_path=rel, language=lang, start_line=start, end_line=end,
        symbol_name=symbol, chunk_type=ctype,
    )


def _emit(rel: str, lang: str, lines: list[str], start: int, end: int,
          symbol: str, ctype: str = "code") -> list[Chunk]:
    """Trim blank edges, then split anything longer than MAX_CHUNK_LINES."""
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    if start > end:
        return []
    if end - start + 1 <= MAX_CHUNK_LINES:
        return [_mk(rel, lang, lines, start, end, symbol, ctype)]
    parts: list[Chunk] = []
    s, n = start, 1
    while s <= end:
        e = min(s + MAX_CHUNK_LINES - 1, end)
        parts.append(_mk(rel, lang, lines, s, e, f"{symbol} (part {n})", ctype))
        if e == end:
            break
        s = e - OVERLAP + 1
        n += 1
    return parts


def _chunk_windows(rel: str, lang: str, lines: list[str], ctype: str = "code") -> list[Chunk]:
    """Fallback: fixed line windows with overlap."""
    chunks: list[Chunk] = []
    start = 1
    while start <= len(lines):
        end = min(start + WINDOW_LINES - 1, len(lines))
        chunks += _emit(rel, lang, lines, start, end, f"lines {start}-{end}", ctype)
        if end == len(lines):
            break
        start = end - OVERLAP + 1
    return chunks


# ---------------------------------------------------------------- per-language chunkers

def _chunk_python(rel: str, lines: list[str], text: str) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        logger.debug("Syntax error in %s; using line windows", rel)
        return _chunk_windows(rel, "python", lines)

    covered = [False] * (len(lines) + 2)
    chunks: list[Chunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = node.end_lineno or node.lineno
            for i in range(start, end + 1):
                covered[i] = True
            chunks += _emit(rel, "python", lines, start, end, node.name)

    run_start: int | None = None
    for i in range(1, len(lines) + 2):
        uncovered = i <= len(lines) and not covered[i]
        if uncovered and run_start is None:
            run_start = i
        elif not uncovered and run_start is not None:
            chunks += _emit(rel, "python", lines, run_start, i - 1, "<module>")
            run_start = None
    return sorted(chunks, key=lambda c: c.start_line)


_JS_PATTERNS = [
    re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)"),
    re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"),
    re.compile(r"^(?:export\s+)?(?:interface|type|enum)\s+(\w+)"),
    re.compile(
        r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?"
        r"(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|\w+\s*=>)"
    ),
]


def _chunk_js(rel: str, lang: str, lines: list[str]) -> list[Chunk]:
    """Regex-based top-level splitter for JS/TS (column-0 declarations only)."""
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        for pat in _JS_PATTERNS:
            m = pat.match(line)
            if m:
                starts.append((i, m.group(1)))
                break
    if not starts:
        return _chunk_windows(rel, lang, lines)

    # Pull preceding comment/decorator lines into the declaration that follows them.
    adjusted: list[tuple[int, str]] = []
    prev = 0
    for lineno, name in starts:
        s = lineno
        while s - 1 > prev and lines[s - 2].lstrip().startswith(("//", "/*", "*", "@")):
            s -= 1
        adjusted.append((s, name))
        prev = lineno

    chunks = _emit(rel, lang, lines, 1, adjusted[0][0] - 1, "<module>")
    for idx, (s, name) in enumerate(adjusted):
        e = adjusted[idx + 1][0] - 1 if idx + 1 < len(adjusted) else len(lines)
        chunks += _emit(rel, lang, lines, s, e, name)
    return chunks


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _chunk_markdown(rel: str, lines: list[str]) -> list[Chunk]:
    sections: list[tuple[int, int, str]] = []
    in_fence = False
    cur_start, cur_title = 1, "<intro>"
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING.match(line)
        if m:
            if i > cur_start:
                sections.append((cur_start, i - 1, cur_title))
            cur_start, cur_title = i, m.group(2)
    sections.append((cur_start, len(lines), cur_title))
    chunks: list[Chunk] = []
    for s, e, title in sections:
        chunks += _emit(rel, "markdown", lines, s, e, title, "doc")
    return chunks


# ---------------------------------------------------------------- public API

def chunk_file(path: Path, root: Path) -> list[Chunk]:
    """Chunk one file; returns [] for unreadable/binary files."""
    text = _read_text(path)
    if text is None:
        return []
    rel = path.relative_to(root).as_posix()
    lang = LANG_BY_EXT.get(path.suffix.lower(), "text")
    lines = text.splitlines()
    if not lines:
        return []
    if lang == "python":
        return _chunk_python(rel, lines, text)
    if lang in ("javascript", "typescript"):
        return _chunk_js(rel, lang, lines)
    if lang == "markdown":
        return _chunk_markdown(rel, lines)
    return _chunk_windows(rel, lang, lines)


def chunk_repo(root: Path) -> Iterator[Chunk]:
    for path in iter_source_files(root):
        yield from chunk_file(path, root)
