import json
import urllib.error

import pytest

from repopilot import cli, github_comment as gh
from repopilot.agents import (
    EmptyDiffError, EmptyIndexError, MissingAPIKeyError, _render, filter_findings, parse_review,
    review_diff,
)
from repopilot.diffs import annotate_section, prepare_diff, split_diff
from repopilot.rag import ChromaStore, HybridRetriever, index_repo
from repopilot.schemas import Finding, ReviewReport, ReviewResult
from repopilot.settings import Settings

APP_DIFF = """diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 import os
+import sys
 
-def f():
+def g():
"""

AUTH_DIFF = """diff --git a/auth.py b/auth.py
--- a/auth.py
+++ b/auth.py
@@ -1,3 +1,3 @@
 def hash_password(password: str) -> str:
-    return md5(password)
+    return md5(password.encode()).hexdigest()
"""


def file_diff(name: str, added_lines: int) -> str:
    body = "".join(f"+line {i}\n" for i in range(added_lines))
    return f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -0,0 +1,{added_lines} @@\n{body}"


@pytest.fixture
def settings(tmp_path):
    return Settings(llm_model="scripted/test", llm_max_retries=0, embedding_model="x",
                    chroma_dir=tmp_path / "c", top_k=4, log_level="INFO", max_diff_chars=24000)


@pytest.fixture
def retriever(store, sample_repo):
    index_repo(sample_repo, store)
    return HybridRetriever(store)


def report_json(**over):
    data = {"summary": "Weak hashing.", "verdict": "approve", "findings": [
        {"file": "auth.py", "line": 2, "severity": "High", "issue": "md5 is not safe for passwords",
         "suggestion": "Use bcrypt."}]}
    data.update(over)
    return json.dumps(data)


# ---------------------------------------------------------------- diffs

def test_split_diff_finds_each_file():
    d = APP_DIFF + file_diff("b.py", 2)
    assert [s.path for s in split_diff(d)] == ["app.py", "b.py"]


def test_annotate_numbers_new_file_lines():
    out = annotate_section(split_diff(APP_DIFF)[0].text)
    assert "    1  import os" in out
    assert "    2 +import sys" in out
    assert "      -def f():" in out  # removed lines get no number
    assert "    4 +def g():" in out


def test_prepare_skips_lockfiles_and_binary_sections():
    d = APP_DIFF + file_diff("package-lock.json", 3) + "diff --git a/logo.bin b/logo.bin\nBinary files differ\n"
    prepared = prepare_diff(d, 24000)
    assert prepared.paths == ["app.py"]
    assert set(prepared.skipped) == {"package-lock.json", "logo.bin"}
    assert not prepared.truncated


def test_prepare_omits_files_over_budget_but_keeps_ones_that_fit():
    d = file_diff("small.py", 2) + file_diff("huge.py", 500) + file_diff("tiny.py", 1)
    prepared = prepare_diff(d, 600)
    assert prepared.paths == ["small.py", "tiny.py"]
    assert prepared.omitted == ["huge.py"]
    assert prepared.truncated


def test_prepare_truncates_first_file_when_nothing_fits():
    prepared = prepare_diff(file_diff("huge.py", 500), 400)
    assert prepared.paths == ["huge.py"] and prepared.truncated
    assert "truncated" in prepared.annotated
    assert len(prepared.annotated) < 1200


def test_digest_contains_only_added_lines():
    digest = prepare_diff(APP_DIFF, 24000).digest
    assert "import sys" in digest and "def f()" not in digest


def test_render_is_single_pass():
    assert _render("a {x} {y} {z}", x="{y}", y="B") == "a {y} B {z}"


# ---------------------------------------------------------------- parsing & filtering

def test_parse_review_normalises_model_output():
    report, failed = parse_review("Here you go:\n```json\n" + report_json() + "\n```")
    assert not failed
    f = report.findings[0]
    assert (f.severity, f.line) == ("high", 2)
    assert report.verdict == "request_changes"  # a high finding overrides "approve"


def test_parse_review_accepts_loose_values():
    raw = json.dumps({"summary": "s", "verdict": "Changes Requested", "findings": [
        {"file": "a.py", "line": "12-15", "severity": "critical", "issue": "i"}]})
    report, failed = parse_review(raw)
    assert not failed and report.findings[0].line == 12 and report.verdict == "request_changes"


def test_parse_review_survives_backticks_inside_json_strings():
    raw = report_json(findings=[{"file": "a.py", "line": 1, "severity": "low", "issue": "i",
                                 "suggestion": "use ```code``` instead"}], verdict="comment")
    report, failed = parse_review(raw)
    assert not failed and "```code```" in report.findings[0].suggestion


def test_parse_review_falls_back_on_garbage():
    report, failed = parse_review("Sorry, I cannot do that.")
    assert failed and report.findings == [] and "Sorry" in report.summary


def test_medium_finding_blocks_approve():
    r = ReviewReport(verdict="approve", findings=[Finding(file="a", severity="medium", issue="x")])
    assert r.verdict == "comment"


def test_filter_findings_drops_unknown_files_and_fixes_prefixes():
    report = ReviewReport(findings=[
        Finding(file="b/app.py", issue="ok"), Finding(file="ghost.py", issue="made up"),
        Finding(file="ctx.py", issue="seen in retrieval")])
    kept, dropped = filter_findings(report, {"app.py", "ctx.py"})
    assert [f.file for f in kept.findings] == ["app.py", "ctx.py"] and dropped == 1


# ---------------------------------------------------------------- review_diff (LLM mocked)

def test_review_diff_end_to_end_with_fake_runner(retriever, settings):
    def runner(prepared, tool):
        tool._run("password hash")
        return report_json(findings=[
            {"file": "auth.py", "line": 2, "severity": "high", "issue": "md5", "suggestion": "bcrypt"},
            {"file": "ghost.py", "line": 1, "severity": "low", "issue": "made up"}])

    result = review_diff(AUTH_DIFF + file_diff("yarn.lock", 2), retriever, settings, runner=runner)
    assert result.reviewed_files == ["auth.py"] and result.skipped_files == ["yarn.lock"]
    assert [f.file for f in result.report.findings] == ["auth.py"]
    assert result.dropped_findings == 1 and result.report.verdict == "request_changes"


def test_review_diff_reports_truncation(retriever, settings):
    small = Settings(**{**settings.__dict__, "max_diff_chars": 600})
    d = file_diff("small.py", 2) + file_diff("huge.py", 500)
    result = review_diff(d, retriever, small, runner=lambda p, t: report_json(findings=[], verdict="approve"))
    assert result.truncated and result.omitted_files == ["huge.py"]


def test_review_diff_error_cases(retriever, settings, tmp_path, embedder, monkeypatch):
    with pytest.raises(EmptyDiffError):
        review_diff("  \n", retriever, settings, runner=lambda p, t: "")
    with pytest.raises(EmptyDiffError):
        review_diff(file_diff("poetry.lock", 3), retriever, settings, runner=lambda p, t: "")
    with pytest.raises(EmptyIndexError):
        empty = HybridRetriever(ChromaStore(tmp_path / "empty_db", embedder))
        review_diff(APP_DIFF, empty, settings, runner=lambda p, t: "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    gem = Settings(**{**settings.__dict__, "llm_model": "gemini/test"})
    with pytest.raises(MissingAPIKeyError):
        review_diff(APP_DIFF, retriever, gem)


def test_full_review_crew_with_scripted_llm(retriever, settings, scripted_llm):
    """Runs the real Researcher -> Reviewer crew against a scripted fake LLM."""
    llm = scripted_llm([
        'Thought: I need context.\nAction: search_codebase\nAction Input: {"query": "password hash"}',
        "Thought: I now know the final answer\nFinal Answer: `auth.py:1-3` is the existing hash function.",
        "Thought: I now can give a great answer\nFinal Answer: " + report_json(),
    ])
    result = review_diff(AUTH_DIFF, retriever, settings, llm=llm)
    assert llm.calls == 3 and not result.parse_failed
    assert result.report.findings[0].file == "auth.py"
    assert result.report.verdict == "request_changes"


# ---------------------------------------------------------------- markdown

def make_result(**over):
    base = dict(report=ReviewReport(summary="Looks risky.", verdict="comment", findings=[
        Finding(file="b.py", line=5, severity="low", issue="minor\nthing", suggestion="tidy"),
        Finding(file="a.py", line=9, severity="high", issue="bug", suggestion="fix it")]),
        reviewed_files=["a.py", "b.py"])
    base.update(over)
    return ReviewResult(**base)


def test_markdown_has_marker_labels_locations_and_order():
    md = gh.format_review_markdown(make_result())
    assert md.startswith(gh.MARKER)
    assert md.index("High") < md.index("Low")
    assert "`a.py:9`" in md and "minor thing" in md and "Changes requested" in md


def test_markdown_notes_for_truncation_skips_and_drops():
    md = gh.format_review_markdown(make_result(
        truncated=True, omitted_files=["big.py"], skipped_files=["x.lock"], dropped_findings=2))
    assert "Large diff" in md and "`big.py`" in md and "x.lock" in md and "2 finding(s)" in md


def test_markdown_with_no_findings_and_parse_failure():
    clean = gh.format_review_markdown(ReviewResult(report=ReviewReport(summary="Fine.", verdict="approve")))
    assert "No issues found" in clean
    failed = gh.format_review_markdown(ReviewResult(report=ReviewReport(summary="raw"), parse_failed=True))
    assert "not valid structured output" in failed and "No issues found" not in failed


def test_markdown_is_capped():
    huge = ReviewReport(summary="x" * 70_000)
    assert len(gh.format_review_markdown(ReviewResult(report=huge))) <= gh.MAX_COMMENT_CHARS + 100


# ---------------------------------------------------------------- GitHub API

class FakeAPI:
    def __init__(self, pages=None):
        self.pages, self.calls = pages or [[]], []

    def __call__(self, method, url, token, payload=None):
        self.calls.append((method, url, payload))
        if method == "GET":
            page = int(url.rsplit("page=", 1)[1])
            return self.pages[page - 1] if page <= len(self.pages) else []
        return {}


def test_upsert_creates_when_no_existing_comment(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(gh, "_api", api)
    assert gh.upsert_pr_comment("o/r", 7, "body", "tok") == "created"
    assert api.calls[-1][0] == "POST" and api.calls[-1][1].endswith("/repos/o/r/issues/7/comments")


def test_upsert_updates_existing_comment_even_on_page_two(monkeypatch):
    page1 = [{"id": i, "body": "hi"} for i in range(100)]
    page2 = [{"id": 555, "body": f"{gh.MARKER}\nold"}]
    api = FakeAPI([page1, page2])
    monkeypatch.setattr(gh, "_api", api)
    assert gh.upsert_pr_comment("o/r", 7, "new", "tok") == "updated"
    method, url, payload = api.calls[-1]
    assert method == "PATCH" and url.endswith("/issues/comments/555") and payload == {"body": "new"}


def test_upsert_validates_inputs():
    with pytest.raises(ValueError):
        gh.upsert_pr_comment("not-a-repo", 1, "b", "t")
    with pytest.raises(ValueError):
        gh.upsert_pr_comment("o/r", 0, "b", "t")


def test_api_error_includes_permissions_hint(monkeypatch):
    def boom(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(gh.urllib.request, "urlopen", boom)
    with pytest.raises(gh.GitHubAPIError, match="pull-requests: write"):
        gh._api("POST", "https://api.github.com/x", "tok", {"a": 1})


# ---------------------------------------------------------------- CLI wiring

@pytest.fixture
def cli_env(tmp_path, monkeypatch, embedder):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda name: embedder)
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    diff = tmp_path / "pr.diff"
    diff.write_text(APP_DIFF)
    return diff


def test_cli_review_prints_markdown_and_posts_comment(cli_env, monkeypatch, capsys, tmp_path):
    import repopilot.agents as agents

    monkeypatch.setattr(agents, "review_diff", lambda diff, retriever, settings: make_result())
    posted = {}
    monkeypatch.setattr(gh, "upsert_pr_comment",
                        lambda repo, pr, body, token: posted.update(repo=repo, pr=pr, token=token) or "created")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    out_file = tmp_path / "review.md"
    code = cli.main(["review", "--diff", str(cli_env), "--post-comment", "--repo", "o/r", "--pr", "3",
                     "--markdown-out", str(out_file)])
    assert code == 0 and posted == {"repo": "o/r", "pr": 3, "token": "tok"}
    assert "RepoPilot review" in capsys.readouterr().out
    assert out_file.read_text(encoding="utf-8").startswith(gh.MARKER)


def test_cli_review_needs_token_before_calling_llm(cli_env, monkeypatch, capsys):
    import repopilot.agents as agents

    def must_not_run(*a, **k):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(agents, "review_diff", must_not_run)
    code = cli.main(["review", "--diff", str(cli_env), "--post-comment", "--repo", "o/r", "--pr", "3"])
    assert code == 1 and "GITHUB_TOKEN" in capsys.readouterr().err


def test_cli_review_empty_diff_is_not_an_error(cli_env, capsys):
    cli_env.write_text("")
    assert cli.main(["review", "--diff", str(cli_env)]) == 0
    assert "Nothing to review" in capsys.readouterr().out


def test_cli_review_missing_diff_file(cli_env, capsys):
    assert cli.main(["review", "--diff", str(cli_env) + ".nope"]) == 1
    assert "error" in capsys.readouterr().err
