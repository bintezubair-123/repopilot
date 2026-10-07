"""Format a review as markdown and post it as a single, updatable PR comment."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from .schemas import ReviewResult

logger = logging.getLogger(__name__)

MARKER = "<!-- repopilot-review -->"  # lets us find and update our own comment
MAX_COMMENT_CHARS = 60_000  # GitHub's limit is 65,536
API_URL = "https://api.github.com"

SEVERITY_LABEL = {"high": "\U0001F534 High", "medium": "\U0001F7E0 Medium", "low": "\U0001F7E1 Low"}
VERDICT_LABEL = {
    "approve": "\u2705 Looks good",
    "comment": "\U0001F4AC Comments",
    "request_changes": "\u26A0\uFE0F Changes requested",
}


class GitHubAPIError(RuntimeError):
    """A GitHub API call failed."""


# ---------------------------------------------------------------- formatting

def _inline(text: str) -> str:
    """Collapse whitespace so multi-line model text can't break the list layout."""
    return " ".join(text.split())


def _file_list(paths: list[str], limit: int = 10) -> str:
    shown = ", ".join(f"`{p}`" for p in paths[:limit])
    return shown + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def format_review_markdown(result: ReviewResult) -> str:
    report = result.report
    parts = [MARKER, "## \U0001F916 RepoPilot review", "",
             f"**Verdict:** {VERDICT_LABEL[report.verdict]}", "",
             report.summary.strip() or "_No summary provided._"]

    if result.parse_failed:
        parts += ["", "> \u26A0\uFE0F The model's answer was not valid structured output, "
                      "so no individual findings are shown."]

    if report.findings:
        parts += ["", f"### Findings ({len(report.findings)})"]
        for severity in ("high", "medium", "low"):
            group = sorted((f for f in report.findings if f.severity == severity),
                           key=lambda f: (f.file, f.line or 0))
            if not group:
                continue
            parts += ["", f"#### {SEVERITY_LABEL[severity]}"]
            for f in group:
                location = f"{f.file}:{f.line}" if f.line else f.file
                parts.append(f"- **`{location}`** \u2014 {_inline(f.issue)}")
                if f.suggestion.strip():
                    parts.append(f"  - \U0001F4A1 {_inline(f.suggestion)}")
    elif not result.parse_failed:
        parts += ["", "No issues found in the reviewed changes."]

    notes = []
    if result.truncated:
        if result.omitted_files:
            notes.append(f"**Large diff:** reviewed {len(result.reviewed_files)} file(s); "
                         f"not reviewed: {_file_list(result.omitted_files)}.")
        else:
            notes.append("**Large diff:** only the first part of the diff was reviewed "
                         "(cut to the size limit).")
    if result.skipped_files:
        notes.append(f"Skipped (lockfiles, binary or generated): {_file_list(result.skipped_files)}.")
    if result.dropped_findings:
        notes.append(f"{result.dropped_findings} finding(s) were dropped because they referenced "
                     "files outside the diff.")
    if notes:
        parts += [""] + [f"> {n}" for n in notes]

    parts += ["", "<sub>Automated review by RepoPilot. AI can be wrong, so verify before acting.</sub>"]
    body = "\n".join(parts)
    if len(body) > MAX_COMMENT_CHARS:
        body = body[:MAX_COMMENT_CHARS].rsplit("\n", 1)[0] + "\n\n> \u26A0\uFE0F Comment truncated (too long)."
    return body


# ---------------------------------------------------------------- GitHub API

def _api(method: str, url: str, token: str, payload: dict | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "repopilot",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        hint = ""
        if exc.code in (401, 403):
            hint = (" Check that the workflow has `permissions: pull-requests: write` "
                    "(pull requests from forks only get a read-only token).")
        raise GitHubAPIError(f"GitHub API returned {exc.code} for {method} {url}: {detail}{hint}") from exc
    except urllib.error.URLError as exc:
        raise GitHubAPIError(f"Could not reach GitHub: {exc.reason}") from exc


def _find_existing(comments_url: str, token: str, max_pages: int = 10) -> dict | None:
    for page in range(1, max_pages + 1):
        comments = _api("GET", f"{comments_url}?per_page=100&page={page}", token) or []
        for comment in comments:
            if MARKER in (comment.get("body") or ""):
                return comment
        if len(comments) < 100:
            break
    return None


def upsert_pr_comment(repo: str, pr_number: int, body: str, token: str,
                      api_url: str = API_URL) -> str:
    """Create the RepoPilot comment on a PR, or update it if it already exists."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError(f"--repo must look like 'owner/name', got '{repo}'.")
    if pr_number <= 0:
        raise ValueError("--pr must be a positive pull request number.")
    comments_url = f"{api_url}/repos/{repo}/issues/{pr_number}/comments"
    existing = _find_existing(comments_url, token)
    if existing:
        _api("PATCH", f"{api_url}/repos/{repo}/issues/comments/{existing['id']}", token, {"body": body})
        logger.info("Updated review comment %s on PR #%d", existing["id"], pr_number)
        return "updated"
    _api("POST", comments_url, token, {"body": body})
    logger.info("Created review comment on PR #%d", pr_number)
    return "created"
