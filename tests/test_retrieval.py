import pytest

from repopilot.ingest import EmptyRepoError
from repopilot.rag import (
    ChromaStore, HybridRetriever, index_repo, reciprocal_rank_fusion, tokenize,
)


def test_tokenize_splits_camel_and_snake_case():
    toks = tokenize("getUserName hash_password")
    assert {"getusername", "get", "user", "name", "hash_password", "hash", "password"} <= set(toks)


def test_rrf_prefers_items_ranked_by_both_lists():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert max(scores, key=scores.get) == "b"


def test_index_stats(store, sample_repo):
    stats = index_repo(sample_repo, store)
    assert stats.files == 4
    assert stats.chunks >= 5
    assert store.count() == stats.chunks


def test_search_finds_password_hashing(store, sample_repo):
    index_repo(sample_repo, store)
    top = HybridRetriever(store).search("password hash function", k=3)[0].chunk
    assert (top.file_path, top.symbol_name) == ("auth.py", "hash_password")


def test_search_finds_payment_code(store, sample_repo):
    index_repo(sample_repo, store)
    top = HybridRetriever(store).search("charge a credit card", k=3)[0].chunk
    assert top.symbol_name == "charge_card"


def test_search_matches_camel_case_subwords(store, sample_repo):
    index_repo(sample_repo, store)
    top = HybridRetriever(store).search("format currency", k=3)[0].chunk
    assert top.symbol_name == "formatCurrency"


def test_search_respects_k(store, sample_repo):
    index_repo(sample_repo, store)
    assert len(HybridRetriever(store).search("password", k=2)) <= 2


def test_search_on_empty_store_returns_nothing(store):
    assert HybridRetriever(store).search("anything") == []


def test_index_persists_across_store_instances(tmp_path, embedder, sample_repo):
    first = ChromaStore(tmp_path / "db", embedder)
    stats = index_repo(sample_repo, first)
    reopened = ChromaStore(tmp_path / "db", embedder)
    assert reopened.count() == stats.chunks


def test_empty_repo_raises_and_keeps_existing_index(store, sample_repo, tmp_path):
    stats = index_repo(sample_repo, store)
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    with pytest.raises(EmptyRepoError):
        index_repo(empty, store)
    assert store.count() == stats.chunks


def test_missing_repo_path_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        index_repo(tmp_path / "nope", store)
