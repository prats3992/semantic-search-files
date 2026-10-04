import json

import numpy as np
import pytest

from semantic_search import cli, smart_folders, tagging_system
from semantic_search.config_manager import ConfigManager

FAKE_MODEL = "fake-model"
VOCAB = ["python", "code", "space", "planet", "star", "neural", "learning", "data"]


class FakeEmbeddingGenerator:
    """Deterministic bag-of-words embedder so tests run offline without downloading models."""

    def __init__(self, *args, **kwargs):
        pass

    def generate_embeddings_for_text(self, texts, model_name=None):
        valid = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid:
            return np.array([])
        vectors = [[t.lower().count(word) for word in VOCAB] for t in valid]
        return np.array(vectors, dtype=np.float32) + 1e-3


@pytest.fixture
def config_path(tmp_path):
    """A config.json whose storage paths all live inside the test's temp directory."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "default_model_name": FAKE_MODEL,
        "index_base_directory": str(tmp_path / "indices"),
        "tag_file_path": str(tmp_path / "data" / "tags.json"),
        "smart_dirs_file_path": str(tmp_path / "data" / "smart_dirs.json"),
        "recommended_models": [{"name": FAKE_MODEL, "dimension": len(VOCAB)}],
    }))
    return path


@pytest.fixture
def app_env(monkeypatch, config_path):
    """Points every component at the temp config and the fake embedder."""
    for module in (cli, tagging_system, smart_folders):
        monkeypatch.setattr(module, "ConfigManager", lambda: ConfigManager(str(config_path)))
        monkeypatch.setattr(module, "EmbeddingGenerator", FakeEmbeddingGenerator)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return config_path


@pytest.fixture
def docs_dir(tmp_path):
    """A small document tree covering three topics plus files that should be skipped."""
    docs = tmp_path / "docs"
    (docs / "nested").mkdir(parents=True)
    (docs / ".hidden_dir").mkdir()
    (docs / "python_guide.txt").write_text("Python code basics. Writing python code is fun.")
    (docs / "space.md").write_text("Space exploration: every planet and star in the space sky.")
    (docs / "nested" / "ml_notes.txt").write_text("Neural networks and deep learning need lots of data.")
    (docs / ".secret.txt").write_text("hidden python file")
    (docs / ".hidden_dir" / "inside.txt").write_text("hidden python dir")
    (docs / "archive.zip").write_bytes(b"PK\x03\x04 not really a zip")
    return docs
