# Advanced File System Navigator with Semantic Search

**Version:** 1.2.0 (As of May 29, 2025)

This project provides a powerful command-line tool to navigate and search your file system using semantic understanding of file content, file tags, and other criteria. Instead of just matching keywords, it understands the meaning behind your queries and finds the most relevant files.

It uses state-of-the-art Sentence Transformer models to generate embeddings (numerical representations) of your text files, stores them in an efficient FAISS index, and then uses this index to perform fast and accurate semantic searches. It also includes a tagging system and a "Smart Folders" feature for dynamic file organization.

## Key Features

-   **File Discovery:** Scans specified directories for text-based files.
-   **Content Extraction:** 
    -   Extracts plain text content from discovered files.
    -   Supports `.docx` files using `python-docx`.
    -   Utilizes the Gemini API for content extraction from other file types like PDFs and images, if an API key is provided.
-   **Embedding Generation:**
    -   Utilizes Sentence Transformer models to convert text content into meaningful vector embeddings.
    -   Supports selection of different models via a configuration file or command-line arguments.
    -   Caches loaded models for efficiency.
-   **FAISS Indexing:**
    -   Stores generated embeddings in a FAISS (Facebook AI Similarity Search) index for fast retrieval.
    -   Supports retrieval of specific embeddings by file path if already indexed.
    -   Automatically detects and utilizes GPU for FAISS if available, otherwise defaults to CPU.
    -   Saves indices in a CPU-compatible format.
    -   Manages separate index files for each Sentence Transformer model used.
-   **Semantic Search:**
    -   Allows users to search for files using natural language queries.
    -   Retrieves files based on semantic similarity.
    -   Option to display context-aware snippets from search results.
-   **Tagging System:**
    -   Add/remove tags to/from individual files or entire directories.
    -   List tags for a file, or list files associated with a specific tag.
    -   List all unique tags in the system.
    -   Suggest relevant tags for files or directories based on content (using semantic models).
    -   Tag data is stored in a JSON file (`tags.json` by default, configurable).
-   **Smart Folders:**
    -   Define virtual folders based on a rich set of criteria:
        -   Base directories to search within.
        -   File types (extensions).
        -   Tags (all of, any of, none of).
        -   Content keywords (all of, any of).
        -   Semantic similarity to a natural language query (using specified model and threshold).
    -   Dynamically resolves and displays files matching the defined criteria.
    -   Smart Folder definitions are stored in a JSON file (configurable via `config.json`).
    -   Optimized semantic filtering: reuses existing embeddings from the FAISS index if available for candidate files, otherwise generates them on-the-fly.
-   **Command-Line Interface (CLI):**
    -   `index`: To scan a directory and build/update a semantic index.
    -   `search`: To perform a semantic search over an indexed directory.
    -   `list-models`: To display available and recommended Sentence Transformer models.
    -   `tag`: A group of sub-commands to manage file tags (`add`, `add_dir`, `remove`, `list_tags`, `list_files`, `list_all_tags`, `suggest`, `suggest_dir`).
    -   `sf`: A group of sub-commands to manage Smart Folders (`create`, `remove`, `list`, `show`, `get-files`).
-   **Configuration:**
    -   Uses a `config.json` file to manage settings like default models, index storage paths, tag file path, smart folders definition file path, and model details.
    -   Supports a `.env` file for managing sensitive information like API keys (e.g., `GEMINI_API_KEY`).
-   **Shell Script Wrappers:** Basic shell scripts are provided as examples.

## Project Structure

```
semantic-search-files/
├── .env                        # For API keys and other environment variables (e.g., GEMINI_API_KEY)
├── config.json                 # Configuration for models, paths, etc.
├── smart_directories.json      # Example/default for Smart Folder definitions
├── tags.json                   # Example/default for Tagging System data
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── scripts/                    # Shell script wrappers
│   ├── index_directory.sh
│   └── search.sh
├── src/                        # Source code
│   ├── cli.py                  # Main command-line interface
│   ├── config_manager.py       # Handles loading config.json
│   ├── smart_folders.py        # Implements the Smart Folder system
│   ├── tagging_system.py       # Implements the Tagging system
│   ├── main_app.py             # (Initial demo, largely superseded by CLI)
│   ├── content_extraction/
│   ├── embedding_generation/
│   ├── file_discovery/
│   ├── index_storage/
│   └── metadata_extraction/
├── semantic_indices/           # Default directory for storing FAISS indexes
│   └── <model_name_variant>/   # Each model gets its own subdirectory
│       ├── semantic_index.faiss
│       └── semantic_metadata.pkl
└── test_data_<1,2,3>/          # Sample directories with text files for testing
```

## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd semantic-search-files
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\\Scripts\\activate`)*

3.  **Install Dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary packages, including `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`), `python-docx`, `google-generativeai`, and `python-dotenv`.

    **Note on FAISS:**
    -   The `requirements.txt` file specifies `faiss-cpu`.
    -   If you have a compatible NVIDIA GPU and want to use GPU-accelerated FAISS, you can install `faiss-gpu` instead:
        ```bash
        pip uninstall faiss-cpu
        pip install faiss-gpu  # Check FAISS documentation for specific CUDA compatibility
        ```

4.  **Configuration (Important):**
    Review and customize `config.json` in the project root. You need to ensure paths like `tag_file_path` and `smart_dirs_file_path` are set.
    -   Change the `default_model_name`.
    -   Modify the `index_base_directory` where FAISS indices will be stored.
    -   Set `tag_file_path` (e.g., `"./tags.json"`).
    -   Set `smart_dirs_file_path` (e.g., `"./smart_folders_definitions.json"`).
    -   Add or update information in `recommended_models`.

5.  **Set up Gemini API Key (Optional for advanced content extraction):**
    If you plan to extract content from files like PDFs or images, you'll need a Gemini API key.
    -   Create a file named `.env` in the root of the project (`semantic-search-files/.env`).
    -   Add your Gemini API key to this file:
        ```
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        ```
    -   Replace `"YOUR_ACTUAL_GEMINI_API_KEY"` with your real API key.
    -   **Important:** Add `.env` to your `.gitignore` file to prevent committing your API key to version control.

## Usage

All commands should be run from the root directory of the project (`semantic-search-files/`).
The primary interface is `src/cli.py`.

### 1. Listing Available Models

To see a detailed list of recommended Sentence Transformer models, their use cases, and other properties:
```bash
python src/cli.py list-models
```
This information is sourced from `config.json`.

### 2. Indexing a Directory

Before you can search, you need to index the files in a directory. This process scans the files, generates embeddings for their content, and saves them into a FAISS index.

**Command:**
```bash
python src/cli.py [--model <model_name>] index <path_to_directory> [--force]
```

**Arguments:**
-   `--model <model_name>` (Optional): Specify the Sentence Transformer model to use for indexing.
    -   If not provided, the `default_model_name` from `config.json` will be used.
    -   Example: `--model all-mpnet-base-v2`
    -   Use `list-models` to see available options.
-   `path_to_directory`: The absolute or relative path to the directory you want to index.
    -   Example: `test_data_1/` or `/mnt/my_documents`
-   `--force` (Optional): If an index already exists for the specified model and directory, this flag will force re-indexing from scratch. Without it, if an index exists, the command will inform you and exit unless `--force` is used.

**Example:**
To index the `test_data_1/` directory using the default model:
```bash
python src/cli.py index test_data_1/
```

To index `test_data_2/` using a specific model and force re-indexing:
```bash
python src/cli.py --model multi-qa-MiniLM-L6-cos-v1 index test_data_2/ --force
```
Index files will be saved under the directory specified by `index_base_directory` in `config.json` (default: `semantic_indices/`), in a subfolder named after the model (e.g., `semantic_indices/all-MiniLM-L6-v2/`).

### 3. Searching Indexed Files

Once a directory is indexed with a particular model, you can perform semantic searches.

**Command:**
```bash
python src/cli.py [--model <model_name>] search \"<your_query>\" [-k <number_of_results>] [--snippet]
```

**Arguments:**
-   `--model <model_name>` (Optional): Specify the Sentence Transformer model whose index you want to search.
    -   This **must** be the same model used to index the directory.
    -   If not provided, the `default_model_name` from `config.json` will be used.
-   `"your_query"`: The natural language query string. Enclose in quotes if it contains spaces.
    -   Example: `"information about python decorators"`
-   `-k <number_of_results>` (Optional): The number of top matching files to return (default: 5).
    -   Example: `-k 10`
-   `--snippet` (Optional): If provided, the search results will include a small, contextually relevant snippet from each found file.

**Example:**
To search for "python asynchronous programming concepts" using the default model's index, showing top 3 results with snippets:
```bash
python src/cli.py search "python asynchronous programming concepts" -k 3 --snippet
```

To search using a specific model:
```bash
python src/cli.py --model multi-qa-MiniLM-L6-cos-v1 search "details about Mars missions" --snippet
```

### 4. Managing Tags

The `tag` command group allows you to manage file tags.

**General Syntax:**
```bash
python src/cli.py tag <tag_action> [options]
```

**Actions & Options:**

*   **Add tag to file:**
    ```bash
    python src/cli.py tag add --file-path <path/to/file.txt> --tag-name "my_tag"
    ```
*   **Add tag to all files in a directory (recursively):**
    ```bash
    python src/cli.py tag add_dir --dir-path <path/to/directory> --tag-name "project_alpha"
    ```
*   **Remove tag from file:**
    ```bash
    python src/cli.py tag remove --file-path <path/to/file.txt> --tag-name "my_tag"
    ```
*   **List tags for a file:**
    ```bash
    python src/cli.py tag list_tags --file-path <path/to/file.txt>
    ```
*   **List files with a specific tag:**
    ```bash
    python src/cli.py tag list_files --tag-name "project_alpha"
    ```
*   **List all unique tags:**
    ```bash
    python src/cli.py tag list_all_tags
    ```
*   **Suggest tags for a file (uses semantic model):**
    ```bash
    python src/cli.py tag suggest --file-path <path/to/file.txt> [--model <model_name>] [--top-n <number>]
    ```
*   **Suggest tags for all files in a directory:**
    ```bash
    python src/cli.py tag suggest_dir --dir-path <path/to/directory> [--model <model_name>] [--top-n <number>]
    ```

### 5. Managing Smart Folders

The `sf` (Smart Folder) command group allows you to define and use dynamic virtual folders.

**General Syntax:**
```bash
python src/cli.py sf <sf_action> [options]
```

**Actions & Options:**

*   **Create a Smart Folder:**
    Provide criteria either as a JSON string or a path to a JSON file.
    ```bash
    python src/cli.py sf create <folder_name> '<json_criteria_string>'
    # OR
    python src/cli.py sf create <folder_name> /path/to/criteria.json
    ```
    **Example Criteria JSON:**
    ```json
    {
        "base_directories": ["./docs", "/projects/archive"],
        "file_types": [".md", ".txt"],
        "tags_all_of": ["report", "final"],
        "content_keywords_any_of": ["budget", "q4"],
        "semantic_query": "summary of project finances for the last quarter",
        "semantic_model": "all-MiniLM-L6-v2", 
        "semantic_threshold": 0.7 
    }
    ```

*   **List all Smart Folders:**
    ```bash
    python src/cli.py sf list
    ```

*   **Show criteria for a Smart Folder:**
    ```bash
    python src/cli.py sf show <folder_name>
    ```

*   **Get (resolve) files in a Smart Folder:**
    ```bash
    python src/cli.py sf get-files <folder_name>
    ```

*   **Remove a Smart Folder definition:**
    ```bash
    python src/cli.py sf remove <folder_name>
    ```

### Using Shell Scripts (Alternative)

The `scripts/` directory contains simple wrapper scripts:
-   `scripts/index_directory.sh <directory_path> [model_name] [--force]`
-   `scripts/search.sh "<query>" [model_name] [k_results] [--snippet]`

**Examples:**
```bash
bash scripts/index_directory.sh test_data_1/ all-MiniLM-L6-v2
bash scripts/search.sh "basics of machine learning" all-MiniLM-L6-v2 5 --snippet
```
Make sure the scripts are executable (`chmod +x scripts/*.sh`).

## Future Enhancements (Planned / Considered)

-   **Advanced File Navigation/Organization:** ~~Tagging System (manual & AI-suggested)~~, ~~Smart Folders~~. (Completed)
-   **Index Management:** More granular control (status, clear specific index, list indexed directories/models).
-   **Enhanced Content Extraction:** ~~Support for more file types (e.g., PDF, DOCX, code files with comment extraction).~~ (DOCX support added directly; PDF and other types supported via Gemini API).
-   **Advanced FAISS Index Types:** Explore options for very large datasets.
-   **Incremental Indexing:** Update index only for new/modified files without full re-scan.
-   **User Authentication & Permissions:** For multi-user environments.
-   **Comprehensive Unit and Integration Tests.**

## Contributing

(Details to be added if the project becomes open to contributions)

## License

(To be determined)
