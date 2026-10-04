import os

from semantic_search.file_discovery.discoverer import FileDiscoverer


def names(paths):
    return sorted(os.path.basename(p) for p in paths)


def test_skips_hidden_files_and_directories_by_default(docs_dir):
    found = list(FileDiscoverer().scan_directory(str(docs_dir)))

    assert names(found) == ["archive.zip", "ml_notes.txt", "python_guide.txt", "space.md"]


def test_includes_hidden_entries_when_requested(docs_dir):
    found = list(FileDiscoverer(skip_hidden=False).scan_directory(str(docs_dir)))

    assert ".secret.txt" in names(found)
    assert "inside.txt" in names(found)


def test_yields_absolute_paths(docs_dir, monkeypatch):
    monkeypatch.chdir(docs_dir.parent)

    found = list(FileDiscoverer().scan_directory("docs"))

    assert found and all(os.path.isabs(p) for p in found)


def test_empty_directory_yields_nothing(tmp_path):
    assert list(FileDiscoverer().scan_directory(str(tmp_path))) == []
