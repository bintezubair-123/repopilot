from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_gui_workflows_render(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    app_path = Path(__file__).resolve().parents[1] / "src" / "repopilot" / "gui.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    assert not app.exception
    assert [title.value for title in app.title] == ["RepoPilot", "RepoPilot"]

    for workflow in ("Ask", "Search", "Review", "Index"):
        app.radio[0].set_value(workflow).run()
        assert not app.exception
