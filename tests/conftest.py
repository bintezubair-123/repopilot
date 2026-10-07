import math
import zlib

import pytest

from repopilot.rag import ChromaStore, tokenize


class HashEmbedder:
    """Deterministic fake embedder (hashed bag of words) so tests need no model download."""

    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    return ChromaStore(tmp_path / "chroma", embedder)


@pytest.fixture
def sample_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "auth.py").write_text(
        'def hash_password(password: str) -> str:\n'
        '    """Hash a password with bcrypt before storing it."""\n'
        '    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()\n'
        '\n\n'
        'def verify_login(user, password):\n'
        '    return bcrypt.checkpw(password.encode(), user.pw_hash)\n'
    )
    (root / "payments.py").write_text(
        'def charge_card(card_number, amount_cents):\n'
        '    """Charge a credit card through the payment gateway."""\n'
        '    return gateway.charge(card_number, amount_cents)\n'
    )
    (root / "README.md").write_text(
        "# Demo\n\nA tiny demo app.\n\n## Setup\nRun `pip install -r requirements.txt`.\n"
    )
    (root / "utils.js").write_text(
        "export function formatCurrency(value) {\n  return value.toFixed(2);\n}\n"
    )
    return root


@pytest.fixture
def scripted_llm():
    """Factory for a fake CrewAI LLM that returns canned replies in order (no network)."""
    from crewai import BaseLLM

    class ScriptedLLM(BaseLLM):
        def __init__(self, replies):
            super().__init__(model="scripted/test")
            object.__setattr__(self, "_replies", list(replies))
            object.__setattr__(self, "calls", 0)

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            object.__setattr__(self, "calls", self.calls + 1)
            return self._replies.pop(0)

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

    return ScriptedLLM
