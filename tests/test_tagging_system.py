import json

import pytest

from semantic_search.tagging_system import TaggingSystem


@pytest.fixture
def tags(app_env):
    return TaggingSystem()


def test_add_and_list_tags(tags, docs_dir):
    file_path = str(docs_dir / "python_guide.txt")

    tags.add_tag_to_file(file_path, "Python")
    tags.add_tag_to_file(file_path, "guide")

    assert tags.get_tags_for_file(file_path) == ["python", "guide"]
    assert tags.get_files_for_tag("PYTHON") == [file_path]
    assert sorted(tags.list_all_tags()) == ["guide", "python"]


def test_adding_same_tag_twice_does_not_duplicate(tags, docs_dir):
    file_path = str(docs_dir / "python_guide.txt")

    tags.add_tag_to_file(file_path, "python")
    tags.add_tag_to_file(file_path, "python")

    assert tags.get_tags_for_file(file_path) == ["python"]
    assert tags.get_files_for_tag("python") == [file_path]


def test_relative_paths_are_stored_as_absolute(tags, docs_dir, monkeypatch):
    monkeypatch.chdir(docs_dir)

    tags.add_tag_to_file("space.md", "space")

    assert tags.get_files_for_tag("space") == [str(docs_dir / "space.md")]
    assert tags.get_tags_for_file(str(docs_dir / "space.md")) == ["space"]


def test_remove_tag(tags, docs_dir):
    file_path = str(docs_dir / "python_guide.txt")
    tags.add_tag_to_file(file_path, "python")

    tags.remove_tag_from_file(file_path, "python")

    assert tags.get_tags_for_file(file_path) == []
    assert tags.get_files_for_tag("python") == []


def test_tag_directory_skips_hidden_files(tags, docs_dir):
    count = tags.add_tag_to_directory(str(docs_dir), "docs")

    assert count == 4
    assert str(docs_dir / ".secret.txt") not in tags.get_files_for_tag("docs")


def test_tag_missing_directory_returns_zero(tags, tmp_path):
    assert tags.add_tag_to_directory(str(tmp_path / "missing"), "x") == 0


def test_tags_persist_to_disk(app_env, docs_dir):
    file_path = str(docs_dir / "space.md")
    TaggingSystem().add_tag_to_file(file_path, "space")

    assert TaggingSystem().get_tags_for_file(file_path) == ["space"]
    saved = json.loads((app_env.parent / "data" / "tags.json").read_text())
    assert saved["tags"]["space"]["embedding_info"]["model_name"] == "fake-model"


def test_corrupt_tag_file_starts_empty(app_env):
    tag_file = app_env.parent / "data" / "tags.json"
    tag_file.parent.mkdir(parents=True, exist_ok=True)
    tag_file.write_text("{ not json")

    assert TaggingSystem().list_all_tags() == []


def test_suggest_ranks_semantically_closest_existing_tag_first(tags, docs_dir):
    tags.add_tag_to_file(str(docs_dir / "space.md"), "planet")
    tags.add_tag_to_file(str(docs_dir / "nested" / "ml_notes.txt"), "neural")

    suggestions = tags.suggest_tags_for_file(str(docs_dir / "space.md"), top_n=3)

    assert suggestions[0] == "planet"
    assert len(suggestions) == 3


def test_suggest_without_existing_tags_uses_keywords(tags, docs_dir):
    suggestions = tags.suggest_tags_for_file(str(docs_dir / "space.md"), top_n=3)

    assert 0 < len(suggestions) <= 3
    assert all(s == s.lower() for s in suggestions)


def test_suggest_for_unreadable_file_returns_empty(tags, docs_dir):
    assert tags.suggest_tags_for_file(str(docs_dir / "archive.zip")) == []


def test_suggest_for_directory_covers_each_file(tags, docs_dir):
    suggestions = tags.suggest_tags_for_directory(str(docs_dir), top_n=2)

    assert set(suggestions) == {
        str(docs_dir / "python_guide.txt"),
        str(docs_dir / "space.md"),
        str(docs_dir / "nested" / "ml_notes.txt"),
        str(docs_dir / "archive.zip"),
    }
    assert suggestions[str(docs_dir / "archive.zip")] == []
