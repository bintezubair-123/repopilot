"""Unified-diff helpers: split per file, drop noise, enforce a size budget, number lines."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NOISE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "uv.lock", "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
}
NOISE_SUFFIXES = (".min.js", ".min.css", ".map", ".lock", ".png", ".jpg", ".jpeg",
                  ".gif", ".ico", ".pdf", ".woff", ".woff2")
DIGEST_MAX_CHARS = 6000
DIGEST_PER_FILE_CHARS = 1500

_HEADER = re.compile(r"^diff --git a/(.*?) b/(.*)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class FileDiff:
    path: str
    text: str

    @property
    def has_hunks(self) -> bool:
        return any(_HUNK.match(line) for line in self.text.splitlines())


@dataclass
class PreparedDiff:
    included: list[FileDiff] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)  # reviewable but over budget
    skipped: list[str] = field(default_factory=list)  # noise / no textual hunks
    truncated: bool = False
    annotated: str = ""  # included diff with new-file line numbers in the left margin
    digest: str = ""  # short list of added lines, used by the Researcher

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.included]


def is_noise(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in NOISE_NAMES or path.lower().endswith(NOISE_SUFFIXES)


def split_diff(diff_text: str) -> list[FileDiff]:
    """Split a unified diff into one FileDiff per file."""
    lines = diff_text.splitlines()
    sections: list[FileDiff] = []
    path: str | None = None
    cur: list[str] = []
    seen_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            if path is not None:
                sections.append(FileDiff(path, "\n".join(cur)))
            m = _HEADER.match(line)
            path = m.group(2).strip('"') if m else line[len("diff --git "):]
            cur, seen_hunk = [line], False
        elif path is not None:
            if _HUNK.match(line):
                seen_hunk = True
            elif not seen_hunk and line.startswith("+++ b/"):
                path = line[len("+++ b/"):].strip()
            cur.append(line)
    if path is not None:
        sections.append(FileDiff(path, "\n".join(cur)))
    if not sections and any(_HUNK.match(line) for line in lines):  # plain `diff -u` output
        sections.append(FileDiff("(diff)", "\n".join(lines)))
    return sections


def _cut(text: str, limit: int) -> str:
    head = text[:limit].rsplit("\n", 1)[0]
    return head + "\n[... diff for this file truncated ...]"


def annotate_section(text: str) -> str:
    """Prefix each context/added line with its line number in the NEW file."""
    out: list[str] = []
    new_line = 0
    in_hunk = False
    for line in text.splitlines():
        m = _HUNK.match(line)
        if m:
            in_hunk, new_line = True, int(m.group(1))
            out.append(line)
        elif not in_hunk:
            out.append(line)
        elif line.startswith("+") or line.startswith(" ") or line == "":
            out.append(f"{new_line:>5} {line}")
            new_line += 1
        else:  # removed line ("-") or "\ No newline at end of file"
            out.append(f"{'':>5} {line}")
    return "\n".join(out)


def _digest(sections: list[FileDiff]) -> str:
    parts: list[str] = []
    total = 0
    for s in sections:
        added = [ln[1:] for ln in s.text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        body = "\n".join(added)
        if len(body) > DIGEST_PER_FILE_CHARS:
            body = body[:DIGEST_PER_FILE_CHARS] + "\n[...]"
        block = f"### {s.path}\n{body}"
        if total + len(block) > DIGEST_MAX_CHARS:
            parts.append("[... more files omitted from this digest ...]")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def prepare_diff(diff_text: str, max_chars: int) -> PreparedDiff:
    """Drop noise, fit the rest into max_chars (whole files first), and annotate line numbers."""
    sections = split_diff(diff_text)
    skipped = [s.path for s in sections if is_noise(s.path) or not s.has_hunks]
    candidates = [s for s in sections if not (is_noise(s.path) or not s.has_hunks)]

    included: list[FileDiff] = []
    used = 0
    for s in candidates:  # keep original order; a file that doesn't fit is skipped, smaller ones may still fit
        if used + len(s.text) <= max_chars:
            included.append(s)
            used += len(s.text)
    partial = False
    if not included and candidates:  # even the smallest file is too big: keep a prefix of the first
        included = [FileDiff(candidates[0].path, _cut(candidates[0].text, max_chars))]
        partial = True

    kept = {s.path for s in included}
    omitted = [s.path for s in candidates if s.path not in kept]
    return PreparedDiff(
        included=included, omitted=omitted, skipped=skipped,
        truncated=bool(omitted) or partial,
        annotated="\n".join(annotate_section(s.text) for s in included),
        digest=_digest(included),
    )
