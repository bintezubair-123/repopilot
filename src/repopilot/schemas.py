"""Shared data models."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Chunk(BaseModel):
    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str
    chunk_type: str = "code"  # "code" | "doc"

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def citation(self) -> str:
        return self.chunk_id

    @property
    def embed_text(self) -> str:
        """Text used for embedding and BM25 (path + symbol help retrieval)."""
        return f"{self.file_path} :: {self.symbol_name}\n{self.text}"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class IndexStats(BaseModel):
    files: int
    chunks: int


class AskResult(BaseModel):
    answer: str
    sources: list[Chunk]
    unverified_citations: list[str] = []


_SEVERITY_ALIASES = {
    "critical": "high", "blocker": "high", "major": "high", "error": "high",
    "warning": "medium", "moderate": "medium",
    "minor": "low", "info": "low", "nit": "low", "nitpick": "low",
}
_VERDICT_ALIASES = {
    "approved": "approve", "lgtm": "approve",
    "changes_requested": "request_changes", "request_change": "request_changes",
    "comments": "comment",
}


class Finding(BaseModel):
    file: str
    line: int | None = None
    severity: Literal["high", "medium", "low"] = "medium"
    issue: str
    suggestion: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, v: object) -> str:
        s = str(v).strip().lower()
        s = _SEVERITY_ALIASES.get(s, s)
        return s if s in ("high", "medium", "low") else "medium"

    @field_validator("line", mode="before")
    @classmethod
    def _normalise_line(cls, v: object) -> int | None:
        if v is None or isinstance(v, int):
            return v
        m = re.search(r"\d+", str(v))
        return int(m.group()) if m else None


class ReviewReport(BaseModel):
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    verdict: Literal["approve", "comment", "request_changes"] = "comment"

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, v: object) -> str:
        s = re.sub(r"[\s-]+", "_", str(v).strip().lower())
        s = _VERDICT_ALIASES.get(s, s)
        return s if s in ("approve", "comment", "request_changes") else "comment"

    @model_validator(mode="after")
    def _verdict_matches_findings(self) -> "ReviewReport":
        """Keep the verdict consistent with the findings, whatever the model said."""
        severities = {f.severity for f in self.findings}
        if "high" in severities:
            self.verdict = "request_changes"
        elif "medium" in severities and self.verdict == "approve":
            self.verdict = "comment"
        return self


class ReviewResult(BaseModel):
    report: ReviewReport
    reviewed_files: list[str] = Field(default_factory=list)
    omitted_files: list[str] = Field(default_factory=list)  # did not fit the size budget
    skipped_files: list[str] = Field(default_factory=list)  # lockfiles, binaries, generated
    truncated: bool = False
    dropped_findings: int = 0
    parse_failed: bool = False
