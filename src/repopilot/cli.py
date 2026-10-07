"""Command line interface: index, search, ask, review."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .ingest import EmptyRepoError
from .rag import ChromaStore, HybridRetriever, SentenceTransformerEmbedder, index_repo
from .settings import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repopilot", description="AI teammate for your repo")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index a repo into the local vector store")
    idx.add_argument("repo_path")

    ask = sub.add_parser("ask", help="Ask a question about the indexed repo")
    ask.add_argument("question")

    review = sub.add_parser("review", help="Review a unified diff (for example a pull request)")
    review.add_argument("--diff", required=True, help="Path to a unified diff file, or - for stdin")
    review.add_argument("--markdown-out", default=None, help="Also write the review markdown to this file")
    review.add_argument("--post-comment", action="store_true",
                        help="Post (or update) the review as a PR comment; needs GITHUB_TOKEN")
    review.add_argument("--repo", default=None, help="owner/name (default: $GITHUB_REPOSITORY)")
    review.add_argument("--pr", type=int, default=None, help="Pull request number")

    search = sub.add_parser("search", help="Debug: show the top retrieved chunks for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None)
    return parser


def _read_diff(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8", errors="replace")


def _run_review(args: argparse.Namespace, store: ChromaStore, settings: Settings) -> int:
    from .agents import EmptyDiffError, review_diff  # lazy: CrewAI is slow to import
    from .github_comment import format_review_markdown, upsert_pr_comment

    repo = token = None
    if args.post_comment:  # validate before spending any LLM calls
        repo = args.repo or os.getenv("GITHUB_REPOSITORY")
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not repo or not args.pr:
            raise ValueError("--post-comment needs --repo (or $GITHUB_REPOSITORY) and --pr.")
        if not token:
            raise ValueError("--post-comment needs a GITHUB_TOKEN environment variable.")

    try:
        result = review_diff(_read_diff(args.diff), HybridRetriever(store), settings)
    except EmptyDiffError as exc:  # not a failure: nothing to do
        print(f"Nothing to review: {exc}")
        return 0

    markdown = format_review_markdown(result)
    print(markdown)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    if args.post_comment:
        action = upsert_pr_comment(repo, args.pr, markdown, token)
        print(f"\nPR comment {action}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    try:
        store = ChromaStore(settings.chroma_dir, SentenceTransformerEmbedder(settings.embedding_model))
        if args.command == "index":
            stats = index_repo(args.repo_path, store)
            print(f"Indexed {stats.chunks} chunks from {stats.files} files.")
        elif args.command == "ask":
            from .agents import ask_question  # imported lazily: CrewAI is slow to import

            result = ask_question(args.question, HybridRetriever(store), settings)
            print(result.answer)
            if result.sources:
                print("\nSources:")
                for c in result.sources:
                    print(f"  {c.citation}  [{c.symbol_name}]")
            if result.unverified_citations:
                print("\nWarning: these citations do not match any retrieved code: "
                      + ", ".join(result.unverified_citations))
        elif args.command == "review":
            return _run_review(args, store, settings)
        elif args.command == "search":
            hits = HybridRetriever(store).search(args.query, k=args.k or settings.top_k)
            if not hits:
                print("No results. Did you run `repopilot index <repo_path>` first?")
            for hit in hits:
                snippet = hit.chunk.text.strip().splitlines()[0][:100]
                print(f"{hit.score:.4f}  {hit.chunk.citation}  [{hit.chunk.symbol_name}]  {snippet}")
    except (EmptyRepoError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
