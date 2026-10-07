"""CrewAI agents, the retriever tool, and the `ask` and `review` flows."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# Keep CrewAI quiet and offline-friendly. Must be set before crewai is imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import BaseTool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .diffs import PreparedDiff, prepare_diff  # noqa: E402
from .rag import HybridRetriever  # noqa: E402
from .schemas import AskResult, Chunk, ReviewReport, ReviewResult  # noqa: E402
from .settings import Settings  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.getenv("REPOPILOT_CONFIG_DIR", Path(__file__).resolve().parent / "config"))
MAX_CHUNK_CHARS = 2000  # per chunk shown to the LLM, keeps free-tier token use low


# ---------------------------------------------------------------- errors

class MissingAPIKeyError(RuntimeError):
    """The configured LLM provider has no API key set."""


class RateLimitedError(RuntimeError):
    """The LLM provider kept rate limiting us after all retries."""


class EmptyIndexError(RuntimeError):
    """Nothing has been indexed yet."""


class EmptyDiffError(ValueError):
    """The diff is empty or contains nothing reviewable."""


# ---------------------------------------------------------------- credentials & retries

_KEY_ENV = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def check_llm_credentials(model: str) -> None:
    """Fail early with a clear message if the provider's API key is missing."""
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
    env_names = _KEY_ENV.get(provider)
    if env_names is None:  # local (ollama) or unknown provider: let the library decide
        return
    if not any(os.getenv(name) for name in env_names):
        raise MissingAPIKeyError(
            f"No API key found for provider '{provider}' (model '{model}'). "
            f"Set {env_names[0]} in your .env file (see .env.example)."
        )


def _is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("ratelimit", "rate limit", "rate_limit", "429", "resource_exhausted",
                                   "overloaded", "503", "unavailable"))


def with_backoff(fn: Callable[[], Any], retries: int, base_delay: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep) -> Any:
    """Call fn(), retrying rate-limit style errors with exponential backoff + jitter."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            if attempt == retries:
                raise RateLimitedError(
                    f"The LLM provider rate-limited us {retries + 1} times in a row. "
                    "Wait a minute and retry, or switch LLM_MODEL in .env."
                ) from exc
            delay = min(base_delay * 2 ** attempt, 60.0) + random.uniform(0, 1)
            logger.warning("Rate limited (attempt %d/%d); retrying in %.1fs", attempt + 1, retries + 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- retriever tool

class RetrieverInput(BaseModel):
    query: str = Field(..., description="Natural-language question or identifier to search for in the codebase")


class RetrieverTool(BaseTool):
    name: str = "search_codebase"
    description: str = (
        "Search the indexed repository (code and docs). Returns the most relevant chunks, "
        "each labelled with its citation (file_path:start-end). Try different queries or "
        "identifiers if the first search is not enough."
    )
    args_schema: type[BaseModel] = RetrieverInput
    retriever: Any = Field(default=None, exclude=True)
    top_k: int = 6
    seen: dict[str, Chunk] = Field(default_factory=dict)  # every chunk returned, for source tracking

    def _run(self, query: str) -> str:
        hits = self.retriever.search(query, k=self.top_k)
        if not hits:
            return "No matching code found. Try different keywords."
        blocks = []
        for i, hit in enumerate(hits, 1):
            c = hit.chunk
            self.seen[c.chunk_id] = c
            body = c.text if len(c.text) <= MAX_CHUNK_CHARS else c.text[:MAX_CHUNK_CHARS] + "\n... [truncated]"
            blocks.append(f"[{i}] {c.citation} ({c.symbol_name})\n{body}")
        return "\n\n".join(blocks)


# ---------------------------------------------------------------- crew construction

def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render(template: str, **values: str) -> str:
    """Fill {name} placeholders in one pass, so inserted text is never re-scanned."""
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def build_llm(settings: Settings, max_tokens: int = 1200) -> LLM:
    return LLM(model=settings.llm_model, temperature=0.1, max_tokens=max_tokens)


def build_ask_crew(question: str, tool: RetrieverTool, llm: Any) -> Crew:
    """Researcher (uses the retriever tool) -> Explainer (answers with citations)."""
    agents_cfg, tasks_cfg = _load_yaml("agents.yaml"), _load_yaml("tasks.yaml")

    researcher = Agent(**agents_cfg["researcher"], llm=llm, tools=[tool], max_iter=4,
                       allow_delegation=False, verbose=False)
    explainer = Agent(**agents_cfg["explainer"], llm=llm, max_iter=2,
                      allow_delegation=False, verbose=False)

    research = Task(
        description=_render(tasks_cfg["research_task"]["description"], question=question),
        expected_output=tasks_cfg["research_task"]["expected_output"],
        agent=researcher,
    )
    explain = Task(
        description=_render(tasks_cfg["explain_task"]["description"], question=question),
        expected_output=tasks_cfg["explain_task"]["expected_output"],
        agent=explainer,
        context=[research],
    )
    return Crew(agents=[researcher, explainer], tasks=[research, explain],
                process=Process.sequential, verbose=False)


# ---------------------------------------------------------------- citations

_CITATION = re.compile(
    r"(?<![\w/.\\-])((?:[\w.\-]+[/\\])*[\w.\-]+\.[A-Za-z0-9]+):(\d+)(?:\s*[-\u2013]\s*(\d+))?"
)


def extract_citations(text: str) -> list[tuple[str, int, int]]:
    """Find `path:start-end` (or `path:line`) citations, normalised to forward slashes."""
    found: list[tuple[str, int, int]] = []
    for path, start, end in _CITATION.findall(text):
        path = path.replace("\\", "/").removeprefix("./")
        s = int(start)
        e = int(end) if end else s
        found.append((path, s, max(s, e)))
    return found


def _overlaps(chunk: Chunk, path: str, start: int, end: int) -> bool:
    return chunk.file_path == path and start <= chunk.end_line and end >= chunk.start_line


def verify_citations(answer: str, seen: dict[str, Chunk]) -> tuple[list[Chunk], list[str]]:
    """Return (chunks actually cited, citations that match nothing retrieved)."""
    cited: dict[str, Chunk] = {}
    unverified: list[str] = []
    for path, start, end in extract_citations(answer):
        label = f"{path}:{start}-{end}" if end != start else f"{path}:{start}"
        matches = [c for c in seen.values() if _overlaps(c, path, start, end)]
        if matches:
            for c in matches:
                cited[c.chunk_id] = c
        elif label not in unverified:
            unverified.append(label)
    return list(cited.values()), unverified


# ---------------------------------------------------------------- ask flow

Runner = Callable[[str, RetrieverTool], str]


def _default_runner(settings: Settings, llm: Any | None) -> Runner:
    def run(question: str, tool: RetrieverTool) -> str:
        crew = build_ask_crew(question, tool, llm or build_llm(settings))
        result = with_backoff(crew.kickoff, settings.llm_max_retries)
        return str(result.raw)
    return run


def ask_question(question: str, retriever: HybridRetriever, settings: Settings, *,
                 llm: Any | None = None, runner: Runner | None = None) -> AskResult:
    """Answer a question about the indexed repo. `runner` lets tests replace the LLM crew."""
    if not question.strip():
        raise ValueError("Question is empty.")
    if retriever.store.count() == 0:
        raise EmptyIndexError("Nothing is indexed yet. Run: python -m repopilot index <repo_path>")
    if runner is None:
        check_llm_credentials(settings.llm_model)
        runner = _default_runner(settings, llm)

    tool = RetrieverTool(retriever=retriever, top_k=settings.top_k)
    answer = runner(question, tool).strip()

    cited, unverified = verify_citations(answer, tool.seen)
    sources = cited or list(tool.seen.values())
    if not tool.seen:
        logger.warning("The Researcher did not retrieve anything; the answer may be unreliable.")
    return AskResult(answer=answer, sources=sources, unverified_citations=unverified)


# ---------------------------------------------------------------- review flow

def build_review_crew(prepared: PreparedDiff, tool: RetrieverTool, llm: Any) -> Crew:
    """Researcher (finds context for the change) -> Reviewer (returns a JSON report)."""
    agents_cfg, tasks_cfg = _load_yaml("agents.yaml"), _load_yaml("tasks.yaml")

    researcher = Agent(**agents_cfg["researcher"], llm=llm, tools=[tool], max_iter=4,
                       allow_delegation=False, verbose=False)
    reviewer = Agent(**agents_cfg["reviewer"], llm=llm, max_iter=2,
                     allow_delegation=False, verbose=False)

    research = Task(
        description=_render(
            tasks_cfg["review_research_task"]["description"],
            changed_files="\n".join(f"- {p}" for p in prepared.paths),
            digest=prepared.digest,
        ),
        expected_output=tasks_cfg["review_research_task"]["expected_output"],
        agent=researcher,
    )
    review = Task(
        description=_render(tasks_cfg["review_task"]["description"], diff=prepared.annotated),
        expected_output=tasks_cfg["review_task"]["expected_output"],
        agent=reviewer,
        context=[research],
    )
    return Crew(agents=[researcher, reviewer], tasks=[research, review],
                process=Process.sequential, verbose=False)


def _json_candidates(raw: str) -> list[str]:
    candidates = []
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fenced:
        candidates.append(fenced.group(1).strip())
    return candidates


def parse_review(raw: str) -> tuple[ReviewReport, bool]:
    """Parse the Reviewer's JSON. Returns (report, parse_failed); never raises."""
    for candidate in _json_candidates(raw):
        try:
            return ReviewReport.model_validate(json.loads(candidate)), False
        except ValueError:  # JSONDecodeError and pydantic's ValidationError both subclass ValueError
            continue
    logger.warning("Reviewer output was not valid JSON")
    snippet = raw.strip()[:1500]
    return ReviewReport(
        summary="The reviewer's answer could not be parsed as structured output. Raw answer:\n\n" + snippet,
        findings=[], verdict="comment",
    ), True


def _normalise_path(path: str, allowed: set[str]) -> str:
    path = path.replace("\\", "/").removeprefix("./")
    if path not in allowed and path[:2] in ("a/", "b/") and path[2:] in allowed:
        return path[2:]
    return path


def filter_findings(report: ReviewReport, allowed_files: set[str]) -> tuple[ReviewReport, int]:
    """Drop findings that point at files the reviewer never saw (diff files or retrieved files)."""
    kept = []
    for f in report.findings:
        f = f.model_copy(update={"file": _normalise_path(f.file, allowed_files)})
        if f.file in allowed_files:
            kept.append(f)
    dropped = len(report.findings) - len(kept)
    return ReviewReport(summary=report.summary, findings=kept, verdict=report.verdict), dropped


ReviewRunner = Callable[[PreparedDiff, RetrieverTool], str]


def _default_review_runner(settings: Settings, llm: Any | None) -> ReviewRunner:
    def run(prepared: PreparedDiff, tool: RetrieverTool) -> str:
        crew = build_review_crew(prepared, tool, llm or build_llm(settings, max_tokens=2500))
        result = with_backoff(crew.kickoff, settings.llm_max_retries)
        return str(result.raw)
    return run


def review_diff(diff_text: str, retriever: HybridRetriever, settings: Settings, *,
                llm: Any | None = None, runner: ReviewRunner | None = None) -> ReviewResult:
    """Review a unified diff. `runner` lets tests replace the LLM crew."""
    if not diff_text.strip():
        raise EmptyDiffError("The diff is empty, so there is nothing to review.")
    prepared = prepare_diff(diff_text, settings.max_diff_chars)
    if not prepared.included:
        raise EmptyDiffError(
            "The diff has no reviewable changes (only lockfiles, binary, generated, or rename-only files)."
        )
    if retriever.store.count() == 0:
        raise EmptyIndexError("Nothing is indexed yet. Run: python -m repopilot index <repo_path>")
    if runner is None:
        check_llm_credentials(settings.llm_model)
        runner = _default_review_runner(settings, llm)

    tool = RetrieverTool(retriever=retriever, top_k=settings.top_k)
    raw = runner(prepared, tool)
    report, parse_failed = parse_review(raw)

    allowed = set(prepared.paths) | {c.file_path for c in tool.seen.values()}
    report, dropped = filter_findings(report, allowed)
    if prepared.truncated:
        logger.warning("Diff truncated: %d file(s) not reviewed", len(prepared.omitted))
    return ReviewResult(
        report=report, reviewed_files=prepared.paths, omitted_files=prepared.omitted,
        skipped_files=prepared.skipped, truncated=prepared.truncated,
        dropped_findings=dropped, parse_failed=parse_failed,
    )
