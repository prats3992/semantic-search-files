import os

import pytest

from semantic_search.smart_folders import SmartFolderSystem
from semantic_search.tagging_system import TaggingSystem


@pytest.fixture
def sf(app_env):
    return SmartFolderSystem()


def resolve(sf, criteria):
    sf.create_smart_folder("test", criteria)
    return sorted(os.path.basename(p) for p in sf.get_smart_folder_files("test"))


def test_create_requires_base_directories(sf):
    assert sf.create_smart_folder("bad", {"file_types": [".txt"]}) is False
    assert sf.create_smart_folder("bad", {"base_directories": "not-a-list"}) is False
    assert sf.create_smart_folder("bad", ["not", "a", "dict"]) is False


def test_duplicate_names_are_rejected(sf, docs_dir):
    criteria = {"base_directories": [str(docs_dir)]}

    assert sf.create_smart_folder("docs", criteria) is True
    assert sf.create_smart_folder("docs", criteria) is False


def test_list_show_and_remove(sf, docs_dir):
    criteria = {"base_directories": [str(docs_dir)]}
    sf.create_smart_folder("docs", criteria)

    assert sf.list_smart_folders() == ["docs"]
    assert sf.get_smart_folder_criteria("docs") == criteria
    assert sf.remove_smart_folder("docs") is True
    assert sf.remove_smart_folder("docs") is False
    assert sf.get_smart_folder_criteria("docs") is None


def test_definitions_persist_to_disk(app_env, docs_dir):
    SmartFolderSystem().create_smart_folder("docs", {"base_directories": [str(docs_dir)]})

    assert SmartFolderSystem().list_smart_folders() == ["docs"]


def test_unknown_folder_resolves_to_nothing(sf):
    assert sf.get_smart_folder_files("missing") == []


def test_base_directories_only_returns_visible_files(sf, docs_dir):
    assert resolve(sf, {"base_directories": [str(docs_dir)]}) == [
        "archive.zip", "ml_notes.txt", "python_guide.txt", "space.md",
    ]


def test_base_directories_accepts_individual_files(sf, docs_dir):
    assert resolve(sf, {"base_directories": [str(docs_dir / "space.md")]}) == ["space.md"]


def test_file_type_filter(sf, docs_dir):
    assert resolve(sf, {"base_directories": [str(docs_dir)], "file_types": [".TXT"]}) == [
        "ml_notes.txt", "python_guide.txt",
    ]


def test_tag_filters(sf, docs_dir):
    tagger = TaggingSystem()
    tagger.add_tag_to_file(str(docs_dir / "python_guide.txt"), "code")
    tagger.add_tag_to_file(str(docs_dir / "python_guide.txt"), "draft")
    tagger.add_tag_to_file(str(docs_dir / "nested" / "ml_notes.txt"), "code")
    tagger.add_tag_to_file(str(docs_dir / "space.md"), "science")
    sf.tagging_system = TaggingSystem()  # reload tags written above

    base = {"base_directories": [str(docs_dir)]}
    assert resolve(sf, {**base, "tags_all_of": ["code", "draft"]}) == ["python_guide.txt"]
    sf.remove_smart_folder("test")
    assert resolve(sf, {**base, "tags_any_of": ["draft", "science"]}) == ["python_guide.txt", "space.md"]
    sf.remove_smart_folder("test")
    assert resolve(sf, {**base, "tags_all_of": ["code"], "tags_none_of": ["draft"]}) == ["ml_notes.txt"]


def test_content_keyword_filters(sf, docs_dir):
    base = {"base_directories": [str(docs_dir)]}

    assert resolve(sf, {**base, "content_keywords_all_of": ["planet", "STAR"]}) == ["space.md"]
    sf.remove_smart_folder("test")
    assert resolve(sf, {**base, "content_keywords_any_of": ["python", "neural"]}) == [
        "ml_notes.txt", "python_guide.txt",
    ]


def test_semantic_filter_uses_threshold(sf, docs_dir):
    criteria = {
        "base_directories": [str(docs_dir)],
        "file_types": [".txt", ".md"],
        "semantic_query": "planet and star",
        "semantic_threshold": 0.5,
    }

    assert resolve(sf, criteria) == ["space.md"]


def test_semantic_filter_reuses_indexed_embeddings(sf, docs_dir, monkeypatch):
    space_file = str(docs_dir / "space.md")
    indexer = sf._get_faiss_indexer("fake-model")
    os.makedirs(os.path.dirname(indexer.index_file_path), exist_ok=True)
    indexer.add_embedding(sf.embedding_generator.generate_embeddings_for_text(["planet star"])[0], space_file)
    indexer.save_index()

    extracted = []
    original_extract = sf.content_extractor.extract_text
    monkeypatch.setattr(sf.content_extractor, "extract_text", lambda p: extracted.append(p) or original_extract(p))

    files = resolve(sf, {
        "base_directories": [space_file],
        "semantic_query": "planet",
        "semantic_threshold": 0.5,
    })

    assert files == ["space.md"]
    assert extracted == []  # embedding came from the index, file was never re-read
