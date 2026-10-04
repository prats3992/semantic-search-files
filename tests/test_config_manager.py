import json
import os

from semantic_search.config_manager import PROJECT_ROOT, ConfigManager


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_reads_settings_from_file(config_path):
    config = ConfigManager(str(config_path))

    assert config.get_default_model_name() == "fake-model"
    assert config.get_embedding_dimension("fake-model") == 8
    assert config.get_recommended_models()[0]["name"] == "fake-model"


def test_missing_file_falls_back_to_defaults(tmp_path):
    config = ConfigManager(str(tmp_path / "does_not_exist.json"))

    assert config.get_default_model_name() == "all-MiniLM-L6-v2"
    assert config.get_recommended_models() == []


def test_invalid_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ not valid json")

    assert ConfigManager(str(path)).get_default_model_name() == "all-MiniLM-L6-v2"


def test_unknown_model_has_no_dimension(config_path):
    assert ConfigManager(str(config_path)).get_embedding_dimension("unknown-model") is None


def test_relative_paths_resolve_against_project_root(tmp_path, monkeypatch):
    # Paths are created on access, so point the project root at the temp dir.
    monkeypatch.setattr("semantic_search.config_manager.PROJECT_ROOT", str(tmp_path))
    config = ConfigManager(write_config(tmp_path, {
        "index_base_directory": "./indices",
        "tag_file_path": "./data/tags.json",
        "smart_dirs_file_path": "smart.json",
    }))

    assert os.path.normpath(config.get_index_base_directory()) == os.path.join(str(tmp_path), "indices")
    assert config.get_tag_file_path() == os.path.join(str(tmp_path), "data", "tags.json")
    assert config.get_smart_dirs_file_path() == os.path.join(str(tmp_path), "smart.json")
    assert (tmp_path / "indices").is_dir()
    assert (tmp_path / "data").is_dir()


def test_absolute_paths_are_kept(config_path, tmp_path):
    config = ConfigManager(str(config_path))

    assert config.get_tag_file_path() == str(tmp_path / "data" / "tags.json")
    assert config.get_index_base_directory() == str(tmp_path / "indices")


def test_repository_config_lists_default_model_with_dimension():
    config = ConfigManager(os.path.join(PROJECT_ROOT, "config.json"))

    default_model = config.get_default_model_name()
    assert config.get_embedding_dimension(default_model) is not None
