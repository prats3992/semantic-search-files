# Semantic Search Files

[![Tests](https://github.com/prats3992/semantic-search-files/actions/workflows/tests.yml/badge.svg)](https://github.com/prats3992/semantic-search-files/actions/workflows/tests.yml)

Search your local files by **meaning**, not just keywords. Semantic Search Files is a command-line tool that turns documents into vector embeddings with [Sentence Transformers](https://www.sbert.net/), stores them in a [FAISS](https://github.com/facebookresearch/faiss) index, and lets you query them in natural language. It also includes AI-assisted tagging and rule-based "smart folders".

```console
$ semsearch index examples/sample_docs
$ semsearch search "how do I run code concurrently in python" -k 3 --snippet

Search Results (Distance, File Path):
  - '.../examples/sample_docs/python/async_python.md' (Distance: 0.9998)
    Snippet: - **Tasks**: Used to schedule coroutines concurrently.
  - '.../examples/sample_docs/python/testing_in_python.txt' (Distance: 1.2841)
    Snippet: Good tests ensure code quality, prevent regressions (bugs that reappear), ...
  - '.../examples/sample_docs/python/python_basics.txt' (Distance: 1.2930)
    Snippet: - Cross-platform: Python code can run on various operating systems like ...
```

## Features

- **Semantic search**: natural-language queries ranked by embedding similarity, with optional context snippets.
- **Pluggable models**: use any Sentence Transformers model. Each model gets its own index, so you can compare them side by side.
- **Broad file support**: plain text and source code, `.docx`, and (optionally) PDFs and images via the Gemini API.
- **Tagging**: tag files or whole directories, and get tag suggestions from YAKE keyword extraction plus semantic similarity to your existing tags.
- **Smart folders**: virtual folders defined by rules (directories, file types, tags, keywords, and a semantic query with a similarity threshold) and resolved on demand.
- **GPU aware**: uses FAISS GPU if available and falls back to CPU otherwise. Indexes are always saved in a portable CPU format.

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/prats3992/semantic-search-files.git
cd semantic-search-files

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

This installs the `semsearch` command. You can also skip the install and run `pip install -r requirements.txt` followed by `python -m semantic_search ...`.

> The first run downloads the selected embedding model from Hugging Face (about 90 MB for the default model).

### Optional: PDF and image support

PDFs and images (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`) are converted to text by the Gemini API. To enable this, copy the example environment file and add your [API key](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
# then edit .env: GEMINI_API_KEY=your-key
```

Without a key, these files are skipped. Other file types are always processed locally. Only the extensions listed above are ever uploaded to Gemini.

### Optional: GPU acceleration

```bash
pip uninstall faiss-cpu
pip install faiss-gpu    # check FAISS docs for CUDA compatibility
```

## Usage

Run `semsearch --help` or `semsearch <command> --help` for full details. The global `--model` flag selects the embedding model and goes **before** the subcommand.

### Index and search

```bash
# Build an index (use --force to rebuild an existing one)
semsearch index ~/Documents/notes

# Search it
semsearch search "quarterly budget planning" -k 10 --snippet

# Use a different model (search must use the same model you indexed with)
semsearch --model all-mpnet-base-v2 index ~/Documents/notes
semsearch --model all-mpnet-base-v2 search "quarterly budget planning"

# Show the models configured in config.json
semsearch list-models
```

Search results show L2 distance, so **lower means more similar**.

### Tags

```bash
semsearch tag add       --file-path notes/plan.md --tag-name roadmap
semsearch tag add_dir   --dir-path  notes/2025    --tag-name archive
semsearch tag remove    --file-path notes/plan.md --tag-name roadmap
semsearch tag list_tags --file-path notes/plan.md
semsearch tag list_files --tag-name archive
semsearch tag list_all_tags

# Suggest tags from file content
semsearch tag suggest     --file-path notes/plan.md --top-n 5
semsearch tag suggest_dir --dir-path  notes/2025
```

### Smart folders

A smart folder is a saved set of criteria. Pass the criteria as a JSON string or as a path to a JSON file:

```bash
semsearch sf create ml-intro examples/smart_folder_criteria.json
semsearch sf list
semsearch sf show ml-intro
semsearch sf get-files ml-intro
semsearch sf remove ml-intro
```

Supported criteria (only `base_directories` is required):

```json
{
  "base_directories": ["./docs", "/projects/archive"],
  "file_types": [".md", ".txt"],
  "tags_all_of": ["report"],
  "tags_any_of": ["q4", "final"],
  "tags_none_of": ["draft"],
  "content_keywords_all_of": ["budget"],
  "content_keywords_any_of": ["revenue", "forecast"],
  "semantic_query": "summary of project finances for the last quarter",
  "semantic_model": "all-MiniLM-L6-v2",
  "semantic_threshold": 0.5
}
```

Filters are applied in the order shown above, with the semantic filter last. That filter compares cosine similarity against `semantic_threshold` and reuses embeddings from the model's index where it can.

> **Tuning the threshold:** whole-document similarity scores are often lower than you'd expect. With `all-MiniLM-L6-v2`, relevant documents typically score between 0.35 and 0.6. If a smart folder comes back empty, lower `semantic_threshold` first.

## Configuration

Settings live in [`config.json`](config.json) at the repository root:

| Key | Description | Default |
| --- | --- | --- |
| `default_model_name` | Model used when `--model` is not given | `all-MiniLM-L6-v2` |
| `index_base_directory` | Where FAISS indexes are stored (one subfolder per model) | `./semantic_indices` |
| `tag_file_path` | JSON file holding tag data | `./semantic_tags.json` |
| `smart_dirs_file_path` | JSON file holding smart folder definitions | `./smart_directories.json` |
| `recommended_models` | Models shown by `list-models`, with their embedding dimensions | see file |

To use a model that isn't listed, add it to `recommended_models` with the correct `dimension`. The indexer needs this value.

Relative paths are resolved against the repository root. The index, tag, and smart folder files are created on first use and are git-ignored.

## Try it with the sample data

[`examples/sample_docs/`](examples/sample_docs) contains short documents on three topics (Python, astronomy, machine learning):

```bash
semsearch index examples/sample_docs
semsearch search "telescopes observing distant galaxies" -k 3
semsearch search "training a classifier on labelled data" -k 3 --snippet
```

## Project structure

```
semantic_search/
├── cli.py                    # Command-line interface (entry point)
├── config_manager.py         # Loads config.json and .env
├── tagging_system.py         # Tag storage and suggestions
├── smart_folders.py          # Smart folder definitions and resolution
├── file_discovery/           # Recursive directory scanning
├── content_extraction/       # Text, .docx, and Gemini-based extraction
├── embedding_generation/     # Sentence Transformers wrapper with model caching
├── index_storage/            # FAISS index persistence (CPU/GPU)
└── metadata_extraction/      # File system metadata helpers
tests/                        # pytest suite
examples/                     # Sample documents and smart folder criteria
config.json                   # Model and storage configuration
```

## Limitations

- Each file is embedded as a single vector, so very long documents are truncated to the model's maximum sequence length. Chunking is not implemented yet.
- Re-indexing rebuilds the whole index, because incremental updates are not supported yet.
- Index metadata is stored with `pickle`. Only load index files that you created yourself.

## Roadmap

- Chunk-level embeddings for long documents
- Incremental indexing of new and modified files
- Index management commands (status, clear, list indexed directories)

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use a small fake embedding model, so they run offline in a few seconds and never download real models or call the Gemini API.

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss the approach.

## License

[MIT](LICENSE)
