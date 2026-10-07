from repopilot.ingest import (
    MAX_CHUNK_LINES, chunk_file, chunk_repo, iter_source_files,
)

PY_SAMPLE = '''import os


@decorator
def alpha(x):
    return x + 1


class Beta:
    def run(self):
        return 2


async def gamma():
    pass

if __name__ == "__main__":
    alpha(1)
'''

JS_SAMPLE = '''import x from "y";

/** Adds. */
export function add(a, b) {
  return a + b;
}

export const mul = (a, b) => {
  return a * b;
};

class Foo {
  bar() {}
}
'''

MD_SAMPLE = '''# Title
intro text

## Install
```bash
# not a heading
pip install x
```

## Usage
run it
'''


def _chunks(tmp_path, name, content):
    (tmp_path / name).write_text(content)
    return chunk_file(tmp_path / name, tmp_path)


def _assert_text_matches_lines(chunks, content):
    lines = content.splitlines()
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line])


def test_python_splits_by_symbol_with_exact_lines(tmp_path):
    chunks = _chunks(tmp_path, "m.py", PY_SAMPLE)
    by_symbol = {c.symbol_name: c for c in chunks}
    assert {"alpha", "Beta", "gamma", "<module>"} <= set(by_symbol)
    assert (by_symbol["alpha"].start_line, by_symbol["alpha"].end_line) == (4, 6)  # decorator included
    assert (by_symbol["Beta"].start_line, by_symbol["Beta"].end_line) == (9, 11)
    assert (by_symbol["gamma"].start_line, by_symbol["gamma"].end_line) == (14, 15)
    _assert_text_matches_lines(chunks, PY_SAMPLE)
    assert all(c.file_path == "m.py" and c.language == "python" for c in chunks)


def test_js_splits_functions_arrows_and_classes(tmp_path):
    chunks = _chunks(tmp_path, "m.js", JS_SAMPLE)
    by_symbol = {c.symbol_name: c for c in chunks}
    assert {"add", "mul", "Foo", "<module>"} <= set(by_symbol)
    assert by_symbol["add"].start_line == 3  # JSDoc comment attached to the function
    _assert_text_matches_lines(chunks, JS_SAMPLE)


def test_markdown_splits_by_heading_and_ignores_fenced_hashes(tmp_path):
    chunks = _chunks(tmp_path, "README.md", MD_SAMPLE)
    assert [c.symbol_name for c in chunks] == ["Title", "Install", "Usage"]
    assert all(c.chunk_type == "doc" for c in chunks)
    _assert_text_matches_lines(chunks, MD_SAMPLE)


def test_large_function_is_split_with_overlap(tmp_path):
    content = "def big():\n" + "    x = 1\n" * 300
    chunks = _chunks(tmp_path, "big.py", content)
    assert len(chunks) >= 3
    assert all(c.end_line - c.start_line + 1 <= MAX_CHUNK_LINES for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 301
    assert "(part 2)" in chunks[1].symbol_name


def test_python_syntax_error_falls_back_to_windows(tmp_path):
    chunks = _chunks(tmp_path, "bad.py", "def broken(:\n    pass\n")
    assert chunks and chunks[0].symbol_name.startswith("lines ")


def test_walk_skips_junk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def a():\n    return 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("function x() {}\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks.py").write_text("x = 1\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.js").write_text("function y() {}\n")
    (tmp_path / "poetry.lock").write_text("lock\n")
    (tmp_path / "big.py").write_text("x = 1\n" * 100_000)  # > 500 KB
    (tmp_path / "empty.py").write_text("")
    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path)}
    assert found == {"src/app.py"}


def test_binary_file_is_skipped(tmp_path):
    (tmp_path / "blob.py").write_bytes(b"\x00\x01\x02binary" * 10)
    (tmp_path / "ok.py").write_text("def ok():\n    pass\n")
    files = {c.file_path for c in chunk_repo(tmp_path)}
    assert files == {"ok.py"}
