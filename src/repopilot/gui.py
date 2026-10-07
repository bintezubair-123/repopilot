"""Streamlit interface for the RepoPilot CLI workflows."""
from __future__ import annotations

import os
from importlib.resources import files

import streamlit as st

from repopilot.agents import EmptyDiffError, ask_question, review_diff
from repopilot.github_comment import format_review_markdown, upsert_pr_comment
from repopilot.rag import ChromaStore, HybridRetriever, SentenceTransformerEmbedder, index_repo
from repopilot.settings import Settings


st.set_page_config(
    page_title="RepoPilot",
    page_icon=":material/code:",
    layout="wide",
)
st.html(f"<style>{files('repopilot').joinpath('ui.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource
def _runtime(chroma_dir: str, embedding_model: str) -> tuple[ChromaStore, HybridRetriever]:
    store = ChromaStore(chroma_dir, SentenceTransformerEmbedder(embedding_model))
    return store, HybridRetriever(store)


settings = Settings.from_env()
store, retriever = _runtime(str(settings.chroma_dir), settings.embedding_model)

with st.sidebar:
    st.title("RepoPilot")
    st.caption("AI teammate for your repo")
    page = st.radio(
        "Workflow",
        ("Index", "Ask", "Search", "Review"),
        label_visibility="visible",
    )
    st.divider()
    st.caption(f"{store.count()} chunks indexed")

st.title("RepoPilot")
st.caption(
    "A multi-agent AI teammate for GitHub repos, built with CrewAI and RAG on a free stack."
)

if page == "Index":
    st.subheader("Index a repo into the local vector store")
    with st.form("index_form"):
        repo_path = st.text_input("Repository path", value=".")
        index_submitted = st.form_submit_button(
            "Index repository", type="primary", icon=":material/folder_open:"
        )

    if index_submitted:
        try:
            with st.spinner("Indexing repository..."):
                stats = index_repo(repo_path, store)
            _runtime.clear()
            st.success(f"Indexed {stats.chunks} chunks from {stats.files} files.")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"error: {exc}")

elif page == "Ask":
    st.subheader("Ask a question about the indexed repo")
    with st.form("ask_form"):
        question = st.text_area("Question", height=120)
        ask_submitted = st.form_submit_button(
            "Ask RepoPilot", type="primary", icon=":material/chat:"
        )

    if ask_submitted:
        try:
            with st.spinner("Researching the indexed repo..."):
                result = ask_question(question, retriever, settings)
            with st.container(border=True):
                st.markdown(result.answer)
            if result.sources:
                st.markdown("**Sources**")
                for chunk in result.sources:
                    with st.expander(f"{chunk.citation}  [{chunk.symbol_name}]"):
                        st.code(chunk.text, language=chunk.language)
            if result.unverified_citations:
                st.warning(
                    "Warning: these citations do not match any retrieved code: "
                    + ", ".join(result.unverified_citations)
                )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"error: {exc}")

elif page == "Search":
    st.subheader("Debug: show the top retrieved chunks for a query")
    with st.form("search_form"):
        query = st.text_input("Query")
        result_count = st.number_input(
            "Results", min_value=1, value=settings.top_k, step=1
        )
        search_submitted = st.form_submit_button(
            "Search", type="primary", icon=":material/search:"
        )

    if search_submitted:
        try:
            hits = retriever.search(query, k=int(result_count))
            if not hits:
                st.info("No results. Did you run `repopilot index <repo_path>` first?")
            for hit in hits:
                lines = hit.chunk.text.strip().splitlines()
                snippet = lines[0][:100] if lines else ""
                with st.container(border=True):
                    st.markdown(
                        f"**{hit.score:.4f}**  `{hit.chunk.citation}`  "
                        f"[{hit.chunk.symbol_name}]"
                    )
                    st.code(snippet, language=hit.chunk.language)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"error: {exc}")

else:
    st.subheader("Review a unified diff")
    with st.form("review_form"):
        diff_text = st.text_area("Paste a unified diff", height=240)
        diff_upload = st.file_uploader(
            "Or upload a unified diff", type=["diff", "patch", "txt"]
        )
        post_comment = st.checkbox("Post the review as a PR comment")
        repo_name = st.text_input(
            "Repository (owner/name)", value=os.getenv("GITHUB_REPOSITORY", "")
        )
        pr_number = st.number_input(
            "Pull request number", min_value=1, value=1, step=1
        )
        token = st.text_input(
            "GitHub token",
            value=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "",
            type="password",
        )
        review_submitted = st.form_submit_button(
            "Review diff", type="primary", icon=":material/rate_review:"
        )

    if review_submitted:
        diff = (
            diff_upload.getvalue().decode("utf-8", errors="replace")
            if diff_upload is not None
            else diff_text
        )
        if post_comment and not repo_name:
            st.error("--post-comment needs --repo (or $GITHUB_REPOSITORY).")
        elif post_comment and not token:
            st.error("--post-comment needs a GITHUB_TOKEN environment variable.")
        else:
            try:
                with st.spinner("Reviewing diff..."):
                    result = review_diff(diff, retriever, settings)
                    markdown = format_review_markdown(result)
                    action = (
                        upsert_pr_comment(
                            repo_name, int(pr_number), markdown, token
                        )
                        if post_comment
                        else None
                    )
                with st.container(border=True):
                    st.markdown(markdown)
                if action:
                    st.success(f"PR comment {action}.")
            except (OSError, RuntimeError, ValueError) as exc:
                if isinstance(exc, EmptyDiffError):
                    st.info(f"Nothing to review: {exc}")
                else:
                    st.error(f"error: {exc}")
