import numpy as np
import pytest

from semantic_search.index_storage.faiss_indexer import FaissIndexer

DIM = 4


@pytest.fixture
def paths(tmp_path):
    return {
        "index_file_path": str(tmp_path / "index.faiss"),
        "metadata_file_path": str(tmp_path / "metadata.pkl"),
    }


def vec(*values):
    return np.array(values, dtype=np.float32)


def test_new_index_is_empty(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)

    assert indexer.index.ntotal == 0
    assert indexer.search(vec(1, 0, 0, 0)) == []


def test_search_returns_nearest_files_in_order(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(vec(1, 0, 0, 0), "/a.txt")
    indexer.add_embedding(vec(0, 1, 0, 0), "/b.txt")
    indexer.add_embedding(vec(0, 0, 1, 0), "/c.txt")

    results = indexer.search(vec(0.9, 0.1, 0, 0), k=2)

    assert [path for _, path in results] == ["/a.txt", "/b.txt"]
    assert results[0][0] < results[1][0]


def test_k_larger_than_index_returns_all_items(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(vec(1, 0, 0, 0), "/a.txt")

    assert len(indexer.search(vec(1, 0, 0, 0), k=10)) == 1


def test_wrong_dimension_embeddings_are_rejected(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(np.ones(DIM + 1, dtype=np.float32), "/bad.txt")

    assert indexer.index.ntotal == 0
    assert indexer.search(np.ones(DIM + 1, dtype=np.float32)) == []


def test_get_embedding_by_filepath(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(vec(1, 2, 3, 4), "/a.txt")

    np.testing.assert_allclose(indexer.get_embedding_by_filepath("/a.txt"), vec(1, 2, 3, 4))
    assert indexer.get_embedding_by_filepath("/missing.txt") is None


def test_save_and_reload_round_trip(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(vec(1, 0, 0, 0), "/a.txt")
    indexer.add_embedding(vec(0, 1, 0, 0), "/b.txt")
    indexer.save_index()

    reloaded = FaissIndexer(dimension=DIM, **paths)

    assert reloaded.index.ntotal == 2
    assert reloaded.search(vec(0, 1, 0, 0), k=1)[0][1] == "/b.txt"
    np.testing.assert_allclose(reloaded.get_embedding_by_filepath("/a.txt"), vec(1, 0, 0, 0))


def test_reload_with_different_dimension_starts_fresh(paths):
    indexer = FaissIndexer(dimension=DIM, **paths)
    indexer.add_embedding(vec(1, 0, 0, 0), "/a.txt")
    indexer.save_index()

    reloaded = FaissIndexer(dimension=DIM * 2, **paths)

    assert reloaded.index.ntotal == 0
    assert reloaded.get_embedding_by_filepath("/a.txt") is None


def test_corrupt_files_start_fresh(paths):
    for path in paths.values():
        with open(path, "wb") as f:
            f.write(b"garbage")

    assert FaissIndexer(dimension=DIM, **paths).index.ntotal == 0
