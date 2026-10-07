import pytest

from repopilot.agents import (
    CONFIG_DIR, EmptyIndexError, MissingAPIKeyError, RateLimitedError, RetrieverTool,
    ask_question, build_ask_crew, check_llm_credentials, extract_citations, verify_citations,
    with_backoff,
)
from repopilot.rag import HybridRetriever, index_repo
from repopilot.schemas import Chunk
from repopilot.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(llm_model="gemini/test", llm_max_retries=2, embedding_model="x",
                    chroma_dir=tmp_path / "c", top_k=4, log_level="INFO")


@pytest.fixture
def retriever(store, sample_repo):
    index_repo(sample_repo, store)
    return HybridRetriever(store)


# ---------------------------------------------------------------- citations

def test_extract_citations_handles_common_formats():
    text = "See `auth.py:1-3`, (src/a/b.py:10), ./x.py:5\u20136 and src\\m.py:7-9."
    assert extract_citations(text) == [
        ("auth.py", 1, 3), ("src/a/b.py", 10, 10), ("x.py", 5, 6), ("src/m.py", 7, 9),
    ]


def test_verify_citations_flags_unknown_and_accepts_overlap():
    chunk = Chunk(text="x", file_path="auth.py", language="python", start_line=1, end_line=3, symbol_name="f")
    seen = {chunk.chunk_id: chunk}
    cited, bad = verify_citations("Uses `auth.py:2-2` but also `ghost.py:1-9`.", seen)
    assert [c.chunk_id for c in cited] == ["auth.py:1-3"]
    assert bad == ["ghost.py:1-9"]


# ---------------------------------------------------------------- retriever tool

def test_retriever_tool_formats_and_records_sources(retriever):
    tool = RetrieverTool(retriever=retriever, top_k=3)
    out = tool._run("password hash")
    assert "auth.py:1-3 (hash_password)" in out
    assert "auth.py:1-3" in tool.seen


def test_retriever_tool_truncates_long_chunks(store, tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "big.py").write_text("def huge():\n" + "    value = 'x' * 10\n" * 100)
    index_repo(repo, store)
    out = RetrieverTool(retriever=HybridRetriever(store))._run("huge value")
    assert "[truncated]" in out


def test_retriever_tool_handles_no_hits():
    class EmptyRetriever:
        def search(self, query, k=6):
            return []

    assert "No matching code" in RetrieverTool(retriever=EmptyRetriever())._run("anything")


# ---------------------------------------------------------------- ask flow (LLM mocked)

def test_ask_question_returns_verified_sources(retriever, settings):
    def fake_runner(question, tool):
        tool._run("password hash")  # simulate the Researcher calling the tool
        return "Passwords are hashed with bcrypt in `auth.py:1-3`. See also `ghost.py:4-5`."

    result = ask_question("How are passwords hashed?", retriever, settings, runner=fake_runner)
    assert [c.chunk_id for c in result.sources] == ["auth.py:1-3"]
    assert result.unverified_citations == ["ghost.py:4-5"]


def test_ask_question_falls_back_to_retrieved_sources_when_nothing_cited(retriever, settings):
    def fake_runner(question, tool):
        tool._run("password hash")
        return "I could not find this in the indexed code."

    result = ask_question("What colour is the logo?", retriever, settings, runner=fake_runner)
    assert result.sources and not result.unverified_citations


def test_ask_question_requires_an_index(store, settings):
    with pytest.raises(EmptyIndexError):
        ask_question("anything", HybridRetriever(store), settings, runner=lambda q, t: "x")


def test_ask_question_rejects_blank_question(retriever, settings):
    with pytest.raises(ValueError):
        ask_question("   ", retriever, settings, runner=lambda q, t: "x")


def test_ask_question_checks_api_key_before_calling_llm(retriever, settings, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        ask_question("How are passwords hashed?", retriever, settings)


# ---------------------------------------------------------------- credentials & retries

def test_check_llm_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        check_llm_credentials("groq/llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    check_llm_credentials("groq/llama-3.3-70b-versatile")
    check_llm_credentials("ollama/llama3.2")  # local: no key needed
    check_llm_credentials("someprovider/model")  # unknown: left to the library


def test_backoff_retries_rate_limits_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 Too Many Requests: rate limit")
        return "ok"

    assert with_backoff(flaky, retries=3, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_backoff_gives_clear_error_when_exhausted():
    def always():
        raise RuntimeError("RateLimitError: slow down")

    with pytest.raises(RateLimitedError, match="LLM_MODEL"):
        with_backoff(always, retries=1, sleep=lambda s: None)


def test_backoff_does_not_retry_other_errors():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        with_backoff(broken, retries=3, sleep=lambda s: None)
    assert len(calls) == 1


# ---------------------------------------------------------------- crew wiring

def test_config_files_exist():
    assert (CONFIG_DIR / "agents.yaml").exists() and (CONFIG_DIR / "tasks.yaml").exists()


def test_crew_wiring(retriever):
    from crewai import LLM

    tool = RetrieverTool(retriever=retriever)
    crew = build_ask_crew("How are passwords hashed?", tool, LLM(model="groq/test", api_key="fake"))
    researcher, explainer = crew.agents
    research, explain = crew.tasks
    assert [t.name for t in researcher.tools] == ["search_codebase"]
    assert not explainer.tools
    assert explain.context == [research]
    assert "How are passwords hashed?" in research.description
    assert "{question}" not in explain.description


# ---------------------------------------------------------------- full crew with a scripted LLM

def test_full_crew_runs_tool_then_answers_with_citations(retriever, settings, scripted_llm):
    """Runs the real CrewAI crew (Researcher -> Explainer) against a scripted fake LLM."""
    llm = scripted_llm([
        'Thought: I should search.\nAction: search_codebase\nAction Input: {"query": "password hash"}',
        "Thought: I now know the final answer\nFinal Answer: `auth.py:1-3` hash_password uses bcrypt.",
        "Thought: I now can give a great answer\nFinal Answer: Passwords are hashed with bcrypt in `auth.py:1-3`.",
    ])
    scripted = Settings(llm_model="scripted/test", llm_max_retries=0, embedding_model="x",
                        chroma_dir=settings.chroma_dir, top_k=4, log_level="INFO")
    result = ask_question("How are passwords hashed?", retriever, scripted, llm=llm)
    assert llm.calls == 3
    assert "bcrypt" in result.answer
    assert [c.chunk_id for c in result.sources] == ["auth.py:1-3"]
    assert result.unverified_citations == []
