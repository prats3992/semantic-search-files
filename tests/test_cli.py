"""End-to-end tests of the CLI commands, using the fake embedder from conftest."""
import json
import sys

import pytest

from semantic_search import cli


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["semsearch", *argv])
    cli.main()
    return capsys.readouterr().out


@pytest.fixture
def indexed(app_env, docs_dir, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "index", str(docs_dir))
    assert "successfully indexed 3 files" in out
    return docs_dir


def test_search_ranks_most_relevant_file_first(indexed, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "search", "planet star", "-k", "2")

    results = [line for line in out.splitlines() if line.strip().startswith("- ")]
    assert len(results) == 2
    assert "space.md" in results[0]


def test_search_with_snippet(indexed, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "search", "neural data", "-k", "1", "--snippet")

    assert "ml_notes.txt" in out
    assert "Snippet: Neural networks and deep learning need lots of data." in out


def test_reindex_requires_force(indexed, monkeypatch, capsys):
    assert "Use --force to re-index" in run(monkeypatch, capsys, "index", str(indexed))
    assert "successfully indexed 3 files" in run(monkeypatch, capsys, "index", str(indexed), "--force")


def test_search_before_indexing_reports_error(app_env, monkeypatch, capsys):
    assert "Please index documents first" in run(monkeypatch, capsys, "search", "anything")


def test_index_invalid_path(app_env, tmp_path, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "index", str(tmp_path / "missing"))

    assert "is not a valid directory" in out


def test_unknown_model_is_reported(app_env, docs_dir, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--model", "no-such-model", "index", str(docs_dir))

    assert "Could not determine embedding dimension" in out


def test_list_models(app_env, monkeypatch, capsys):
    assert "Model: fake-model" in run(monkeypatch, capsys, "list-models")


def test_tag_commands(app_env, docs_dir, monkeypatch, capsys):
    file_path = str(docs_dir / "space.md")

    run(monkeypatch, capsys, "tag", "add", "--file-path", file_path, "--tag-name", "Astro")
    assert "astro" in run(monkeypatch, capsys, "tag", "list_tags", "--file-path", file_path)
    assert file_path in run(monkeypatch, capsys, "tag", "list_files", "--tag-name", "astro")
    assert "- astro" in run(monkeypatch, capsys, "tag", "list_all_tags")

    run(monkeypatch, capsys, "tag", "remove", "--file-path", file_path, "--tag-name", "astro")
    assert "No tags found" in run(monkeypatch, capsys, "tag", "list_tags", "--file-path", file_path)


def test_tag_add_requires_arguments(app_env, monkeypatch, capsys):
    assert "--file-path and --tag-name are required" in run(monkeypatch, capsys, "tag", "add")


def test_smart_folder_from_json_string_and_file(app_env, docs_dir, tmp_path, monkeypatch, capsys):
    criteria = {"base_directories": [str(docs_dir)], "file_types": [".md"]}
    criteria_file = tmp_path / "criteria.json"
    criteria_file.write_text(json.dumps(criteria))

    run(monkeypatch, capsys, "sf", "create", "inline", json.dumps(criteria))
    run(monkeypatch, capsys, "sf", "create", "from-file", str(criteria_file))

    for name in ("inline", "from-file"):
        out = run(monkeypatch, capsys, "sf", "get-files", name)
        assert str(docs_dir / "space.md") in out
        assert "python_guide.txt" not in out


def test_smart_folder_invalid_json(app_env, monkeypatch, capsys):
    assert "not valid JSON" in run(monkeypatch, capsys, "sf", "create", "bad", "{oops")
