from semantic_search.metadata_extraction.extractor import get_file_metadata


def test_returns_basic_file_metadata(tmp_path):
    path = tmp_path / "Notes.MD"
    path.write_text("hello")

    metadata = get_file_metadata(str(path))

    assert metadata["file_name"] == "Notes.MD"
    assert metadata["file_path"] == str(path)
    assert metadata["file_size_bytes"] == 5
    assert metadata["file_type"] == ".md"
    assert "modification_date" in metadata


def test_missing_file_returns_none(tmp_path):
    assert get_file_metadata(str(tmp_path / "missing.txt")) is None
