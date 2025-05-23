#!/usr/bin/env python
import argparse
import os
import json
import textwrap
from datetime import datetime

from src.file_discovery.discoverer import FileDiscoverer
from src.content_extraction.extractor import ContentExtractor
from src.embedding_generation.generator import EmbeddingGenerator
from src.index_storage.faiss_indexer import FaissIndexer
from src.config_manager import ConfigManager
from src.tagging_system import TaggingSystem
from src.smart_folders import SmartFolderSystem

INDEX_FILE_NAME_SUFFIX = "semantic_index.faiss"
METADATA_FILE_NAME_SUFFIX = "semantic_metadata.pkl"

def get_indexer(model_name: str) -> FaissIndexer | None:
    """Initializes and returns a FaissIndexer instance for the given model name."""
    dimension = ConfigManager().get_embedding_dimension(model_name)
    if dimension is None:
        print(f"Error: Could not determine embedding dimension for model {model_name}. Cannot initialize indexer.")
        return None

    index_base_dir = ConfigManager().get_index_base_directory()
    # Ensure the specific model's directory exists
    model_specific_index_dir = os.path.join(index_base_dir, model_name.replace('/', '_'))
    os.makedirs(model_specific_index_dir, exist_ok=True)

    index_file_path = os.path.join(model_specific_index_dir, INDEX_FILE_NAME_SUFFIX)
    metadata_file_path = os.path.join(model_specific_index_dir, METADATA_FILE_NAME_SUFFIX)

    print(f"Using index file: {index_file_path}")
    print(f"Using metadata file: {metadata_file_path}")

    return FaissIndexer(dimension=dimension, index_file_path=index_file_path, metadata_file_path=metadata_file_path)


def handle_index(args):
    """Handles the 'index' command: Scans a directory, generates embeddings, and indexes them."""
    print(f"Starting indexing process for directory: {args.path}")
    print(f"Using model: {args.model}")

    if not os.path.isdir(args.path):
        print(f"Error: Provided path '{args.path}' is not a valid directory.")
        return

    indexer = get_indexer(args.model)
    if not indexer:
        return

    if not args.force and indexer.index and indexer.index.ntotal > 0:
        print(f"Index already contains {indexer.index.ntotal} items. Use --force to re-index.")
        return

    if args.force and indexer.index and indexer.index.ntotal > 0:
        print("Force re-indexing: Clearing existing index...")
        dimension = ConfigManager().get_embedding_dimension(args.model)
        if dimension is None:
            return
        index_file_path = indexer.index_file_path
        metadata_file_path = indexer.metadata_file_path
        indexer = FaissIndexer(dimension=dimension, index_file_path=index_file_path, metadata_file_path=metadata_file_path)

    files_processed = 0
    files_indexed = 0
    abs_scan_path = os.path.abspath(args.path)
    print(f"Scanning in '{abs_scan_path}' for content to index...")

    # Initialize discoverer, extractor, and generator here to avoid re-creation in loop
    discoverer = FileDiscoverer()
    content_extractor = ContentExtractor()
    embedding_generator = EmbeddingGenerator() # Will use model specified or default

    for file_path in discoverer.scan_directory(abs_scan_path):
        files_processed += 1
        content = content_extractor.extract_text(file_path) # Use instance method
        if content:
            # Use instance method of embedding_generator
            embedding_array = embedding_generator.generate_embeddings_for_text([content], model_name=args.model)
            if embedding_array is not None and embedding_array.size > 0:
                indexer.add_embedding(embedding_array[0], file_path)
                files_indexed += 1
            elif embedding_array is not None and embedding_array.size == 0:
                print(f"Warning: Empty embedding returned for {file_path}. Skipping.")
        if files_processed % 100 == 0:
            print(f"  Processed {files_processed} files, indexed {files_indexed}...")

    if files_indexed > 0 or args.force:
        indexer.save_index()
        print(f"Indexing complete. Processed {files_processed} files, successfully indexed {files_indexed} files.")
        print(f"Index saved for model '{args.model}'.")
    elif files_processed > 0 and files_indexed == 0:
        print(f"Indexing complete. Processed {files_processed} files, but no new content was suitable for indexing.")
    else:
        print("No files found or processed in the given path. Index remains unchanged.")


def handle_search(args):
    """Handles the 'search' command: Takes a query, generates its embedding, and searches the index."""
    print(f"Searching for: '{args.query}'")
    print(f"Using model: {args.model}")
    print(f"Number of results: {args.k}")

    indexer = get_indexer(args.model)
    if not indexer or indexer.index is None or indexer.index.ntotal == 0:
        print(f"Error: Index for model '{args.model}' is empty or not initialized. Please index documents first.")
        return

    embedding_generator = EmbeddingGenerator()
    query_embedding_array = embedding_generator.generate_embeddings_for_text([args.query], model_name=args.model)
    
    if query_embedding_array is None or query_embedding_array.size == 0:
        print("Error: Could not generate embedding for the query.")
        return
    
    query_embedding = query_embedding_array[0]

    search_results = indexer.search(query_embedding, k=args.k)

    if search_results:
        print("\nSearch Results (Distance, File Path):")
        for distance, path in search_results:
            print(f"  - '{path}' (Distance: {distance:.4f})")
            if args.snippet:
                try:
                    content = ContentExtractor().extract_text(path) # Use instance method
                    if content:
                        snippet_text = generate_smart_snippet(content, args.query)
                        print(f"    Snippet: {snippet_text}")
                except Exception as e:
                    print(f"    Could not load snippet: {e}")
        print(f"\nFound {len(search_results)} results.")
    else:
        print("No results found.")


def generate_smart_snippet(content: str, query: str, snippet_length: int = 200, context_window: int = 50) -> str:
    """
    Generates a relevant snippet from content based on the query.
    Tries to find sentences with query terms, or falls back to the first occurrence.
    """
    import re

    query_terms = set(re.split(r'\W+', query.lower()))
    query_terms = {term for term in query_terms if term}

    if not query_terms:
        return (content[:snippet_length] + "...") if len(content) > snippet_length else content

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', content)
    if not sentences:
        return (content[:snippet_length] + "...") if len(content) > snippet_length else content

    best_sentence_idx = -1
    max_overlap = -1
    best_sentence_content = ""

    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        overlap = sum(1 for term in query_terms if term in sentence_lower)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence_idx = i
            best_sentence_content = sentence
        elif overlap == max_overlap and best_sentence_idx != -1:
            if len(sentence) < len(sentences[best_sentence_idx]):
                best_sentence_idx = i
                best_sentence_content = sentence

    if best_sentence_idx == -1 or max_overlap == 0:
        # Fallback: find the first occurrence of any query term in the whole content
        first_match_overall_pos = -1
        for term in query_terms:
            try:
                pos = content.lower().index(term)
                if first_match_overall_pos == -1 or pos < first_match_overall_pos:
                    first_match_overall_pos = pos
            except ValueError:
                continue
        if first_match_overall_pos != -1:
            start = max(0, first_match_overall_pos - context_window // 2)
            end = min(len(content), first_match_overall_pos + snippet_length - context_window // 2)
            snippet = content[start:end]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""
            return (prefix + snippet + suffix).replace('\n', ' ').strip()
        else:
            return (content[:snippet_length] + "...") if len(content) > snippet_length else content

    # If a relevant sentence was found, extract snippet around the first query term match in that sentence
    first_match_in_sentence_pos = -1
    for term in query_terms:
        try:
            pos = best_sentence_content.lower().index(term)
            if first_match_in_sentence_pos == -1 or pos < first_match_in_sentence_pos:
                first_match_in_sentence_pos = pos
        except ValueError:
            continue

    start_index_in_sentence = 0
    end_index_in_sentence = len(best_sentence_content)

    if first_match_in_sentence_pos != -1:
        desired_start = max(0, first_match_in_sentence_pos - context_window)
        desired_end = min(len(best_sentence_content), first_match_in_sentence_pos + (snippet_length - context_window))
        snippet = best_sentence_content[desired_start:desired_end]

        prefix = "..." if desired_start > 0 else ""
        suffix = "..." if desired_end < len(best_sentence_content) else ""
        final_snippet = prefix + snippet + suffix

    else:
        snippet = best_sentence_content[:snippet_length]
        suffix = "..." if len(best_sentence_content) > snippet_length else ""
        final_snippet = snippet + suffix

    return final_snippet.replace('\n', ' ').strip()


def handle_list_models(args):
    """Handles the 'list-models' command: Displays recommended Sentence Transformer models from config."""
    recommended = ConfigManager().get_recommended_models()
    if not recommended:
        print("No recommended models found in configuration.")
        print("Please check your config.json file.")
        return

    print("\n--- Recommended Sentence Transformer Models ---")
    print("You can use these model names with the --model flag for indexing and searching.")
    print("For a more comprehensive list, visit the Hugging Face Model Hub for Sentence Transformers.\n")

    for model_info in recommended:
        print(f"Model: {model_info.get('name', 'N/A')}")
        print(f"  Dimension: {model_info.get('dimension', 'N/A')}")
        print(f"  Size/Speed: {model_info.get('size_speed', 'N/A')}")

        use_case = model_info.get('use_case', 'No use case provided.')
        print("  Use Case:")
        print(textwrap.fill(use_case, width=70, initial_indent="    ", subsequent_indent="    "))

        notes = model_info.get('notes', 'No notes provided.')
        print("  Notes:")
        print(textwrap.fill(notes, width=70, initial_indent="    ", subsequent_indent="    "))
        print("-" * 40)


def handle_tag_operations(args):
    """Handles all operations related to the 'tag' command."""
    tag_system = TaggingSystem()
    if args.tag_action == 'add':
        if not args.file_path or not args.tag_name:
            print("Error: --file-path and --tag-name are required for adding a tag to a single file.")
            return
        tag_system.add_tag_to_file(args.file_path, args.tag_name)
    elif args.tag_action == 'add_dir':
        if not args.dir_path or not args.tag_name:
            print("Error: --dir-path and --tag-name are required for adding a tag to a directory.")
            return
        tag_system.add_tag_to_directory(args.dir_path, args.tag_name)
    elif args.tag_action == 'remove':
        if not args.file_path or not args.tag_name:
            print("Error: --file-path and --tag-name are required for removing a tag.")
            return
        tag_system.remove_tag_from_file(args.file_path, args.tag_name)
    elif args.tag_action == 'list_tags':
        if not args.file_path:
            print("Error: --file-path is required for listing tags of a file.")
            return
        tags = tag_system.get_tags_for_file(args.file_path)
        if tags:
            print(f"Tags for '{args.file_path}': {', '.join(tags)}")
        else:
            print(f"No tags found for '{args.file_path}'.")
    elif args.tag_action == 'list_files':
        if not args.tag_name:
            print("Error: --tag-name is required for listing files with a tag.")
            return
        files = tag_system.get_files_for_tag(args.tag_name)
        if files:
            print(f"Files tagged with '{args.tag_name}':")
            for f_path in files:
                print(f"  - {f_path}")
        else:
            print(f"No files found with tag '{args.tag_name}'.")
    elif args.tag_action == 'list_all_tags':
        all_tags = tag_system.list_all_tags()
        if all_tags:
            print("All unique tags:")
            for t_name in all_tags:
                print(f"  - {t_name}")
        else:
            print("No tags found in the system.")
    elif args.tag_action == 'suggest':
        if not args.file_path:
            print("Error: --file-path is required for suggesting tags.")
            return
        model_name = args.model or ConfigManager().get_default_model_name()
        suggestions = tag_system.suggest_tags_for_file(args.file_path, model_name_for_suggestion=model_name, top_n=args.top_n or 5)
        if suggestions:
            print(f"Suggested tags for '{args.file_path}' (using model '{model_name}'):")
            for s_tag in suggestions:
                print(f"  - {s_tag}")
        else:
            print(f"No tags could be suggested for '{args.file_path}'.")
    elif args.tag_action == 'suggest_dir':
        if not args.dir_path:
            print("Error: --dir-path is required for suggesting tags for a directory.")
            return
        model_name = args.model or ConfigManager().get_default_model_name()
        print(f"Suggesting tags for directory: {args.dir_path} using model '{model_name}'")
        all_dir_suggestions = tag_system.suggest_tags_for_directory(args.dir_path, model_name_for_suggestion=model_name, top_n=args.top_n or 5)
        if all_dir_suggestions:
            print("\n--- Directory Tag Suggestions ---")
            for file_path, suggestions in all_dir_suggestions.items():
                if suggestions:
                    print(f"  File: {file_path}")
                    for s_tag in suggestions:
                        print(f"    - {s_tag}")
                else:
                    print(f"  File: {file_path} - No tags suggested.")
            print("-----------------------------")
        else:
            print(f"No files found or no tags could be suggested for any file in '{args.dir_path}'.")


def handle_sf_operations(args, sf_system: SmartFolderSystem):
    """Handles all operations related to the 'sf' (Smart Folder) command."""
    if args.sf_action == 'create':
        criteria = None
        if os.path.exists(args.criteria_json):
            try:
                with open(args.criteria_json, 'r') as f:
                    criteria = json.load(f)
            except Exception as e:
                print(f"Error reading criteria file '{args.criteria_json}': {e}")
                return
        else:
            try:
                criteria = json.loads(args.criteria_json)
            except json.JSONDecodeError as e:
                print(f"Error: Criteria string is not valid JSON: {e}")
                print("Please provide a valid JSON string or a path to a JSON file.")
                print('Example JSON: { "base_directories": ["./my_docs"], "file_types": [".txt"], "semantic_query": "project plans" }')
                return
        
        if criteria:
            sf_system.create_smart_folder(args.name, criteria)

    elif args.sf_action == 'remove':
        sf_system.remove_smart_folder(args.name)

    elif args.sf_action == 'list':
        sf_system.list_smart_folders()

    elif args.sf_action == 'show':
        criteria = sf_system.get_smart_folder_criteria(args.name)
        if criteria:
            print(f"Criteria for Smart Folder '{args.name}':")
            print(json.dumps(criteria, indent=2))
        # else: Error message already printed by get_smart_folder_criteria

    elif args.sf_action == 'get-files':
        files = sf_system.get_smart_folder_files(args.name)
        if files:
            print(f"Files in Smart Folder '{args.name}':")
            for f_path in files:
                print(f"  - {f_path}")
        # else: Message about no files or error already printed by get_smart_folder_files


def main():
    # Load default model name from configuration for command-line help
    default_model_from_config = ConfigManager().get_default_model_name()

    parser = argparse.ArgumentParser(
        description="Semantic File Navigator: Index, search, and manage files with semantic understanding and tags.",
        formatter_class=argparse.RawTextHelpFormatter # Allows for better formatting of help messages
    )
    parser.add_argument("--model", type=str, default=default_model_from_config,
                        help=f"Sentence Transformer model for embeddings (default: {default_model_from_config}).\n" \
                             f"Use 'list-models' to see options. Example: 'all-mpnet-base-v2'.")

    subparsers = parser.add_subparsers(dest="command", required=True, title="Available Commands",
                                     help="Run '[command] --help' for more information on a specific command.")

    # --- Index Sub-command ---
    parser_index = subparsers.add_parser("index", help="Index files in a directory for semantic search.",
                                       description="Scans a directory, extracts content from supported files, generates embeddings, and stores them in a searchable index.")
    parser_index.add_argument("path", type=str, help="Path to the directory to index.")
    parser_index.add_argument("--force", action="store_true", help="Force re-indexing even if an index already exists for the model.")
    parser_index.set_defaults(func=handle_index)

    # --- Search Sub-command ---
    parser_search = subparsers.add_parser("search", help="Search for files using a semantic query.",
                                        description="Searches the indexed files for content semantically similar to your query.")
    parser_search.add_argument("query", type=str, help="The semantic query string.")
    parser_search.add_argument("-k", type=int, default=5, help="Number of top results to retrieve (default: 5).")
    parser_search.add_argument("--snippet", action="store_true", help="Show a small snippet from the found files.")
    parser_search.set_defaults(func=handle_search)

    # --- Tagging Sub-command ---
    parser_tag = subparsers.add_parser("tag", help="Manage file tags and suggest tags.",
                                     description=textwrap.dedent("""
                                     Manage tags for files and directories. Tags can be used for organization and in Smart Folder criteria.
                                     Actions:
                                       add:         Tag a specific file.
                                       add_dir:     Tag all files in a directory.
                                       remove:      Remove a tag from a file.
                                       list_tags:   List tags for a specific file.
                                       list_files:  List files associated with a specific tag.
                                       list_all_tags: List all unique tags in the system.
                                       suggest:     Suggest tags for a file based on content and existing tags.
                                       suggest_dir: Suggest tags for all files in a directory.
                                     """))
    parser_tag.add_argument("tag_action", choices=["add", "add_dir", "remove", "list_tags", "list_files", "list_all_tags", "suggest", "suggest_dir"], 
                            help="The tagging operation to perform.")
    parser_tag.add_argument("--file-path", type=str, help="Path to the file for tag operations (e.g., add, remove, list_tags, suggest).")
    parser_tag.add_argument("--dir-path", type=str, help="Path to the directory for batch tagging (add_dir, suggest_dir).")
    parser_tag.add_argument("--tag-name", type=str, help="Name of the tag for add, remove, list_files operations.")
    parser_tag.add_argument("--model", type=str, help="Embedding model for tag suggestions (defaults to global --model or config default).")
    parser_tag.add_argument("--top-n", type=int, help="Number of tag suggestions to provide (default: 5).")
    parser_tag.set_defaults(func=handle_tag_operations)

    # --- List Models Sub-command ---
    parser_list_models = subparsers.add_parser("list-models", help="List recommended Sentence Transformer models.",
                                             description="Displays details about recommended Sentence Transformer models suitable for semantic search, as defined in the configuration.")
    parser_list_models.set_defaults(func=handle_list_models)

    # --- Smart Folder Sub-commands ---
    sf_system = SmartFolderSystem() # Initialize SmartFolderSystem once

    parser_sf = subparsers.add_parser("sf", help="Manage Smart Folders.", 
                                    description="Smart Folders dynamically group files based on defined criteria like tags, file types, and semantic content.")
    sf_subparsers = parser_sf.add_subparsers(dest="sf_action", required=True, title="Smart Folder Actions",
                                           help="Run 'sf [action] --help' for more details.")

    # sf create
    parser_sf_create = sf_subparsers.add_parser("create", help="Create a new Smart Folder.",
                                              description="Defines a new Smart Folder with a name and criteria (JSON string or file path).")
    parser_sf_create.add_argument("name", type=str, help="Name of the Smart Folder.")
    parser_sf_create.add_argument("criteria_json", type=str, help="JSON string or path to JSON file defining the criteria.")
    parser_sf_create.set_defaults(func=lambda args: handle_sf_operations(args, sf_system))

    # sf remove
    parser_sf_remove = sf_subparsers.add_parser("remove", help="Remove an existing Smart Folder.",
                                              description="Deletes the definition of a Smart Folder. Does not delete the actual files.")
    parser_sf_remove.add_argument("name", type=str, help="Name of the Smart Folder to remove.")
    parser_sf_remove.set_defaults(func=lambda args: handle_sf_operations(args, sf_system))

    # sf list
    parser_sf_list = sf_subparsers.add_parser("list", help="List all defined Smart Folders.",
                                            description="Shows the names and creation dates of all configured Smart Folders.")
    parser_sf_list.set_defaults(func=lambda args: handle_sf_operations(args, sf_system))

    # sf show
    parser_sf_show = sf_subparsers.add_parser("show", help="Show criteria for a specific Smart Folder.",
                                            description="Displays the JSON criteria used to define a Smart Folder.")
    parser_sf_show.add_argument("name", type=str, help="Name of the Smart Folder.")
    parser_sf_show.set_defaults(func=lambda args: handle_sf_operations(args, sf_system))

    # sf get-files
    parser_sf_get_files = sf_subparsers.add_parser("get-files", help="Resolve and list files in a Smart Folder.",
                                                 description="Dynamically finds and lists all files that currently match the criteria of the specified Smart Folder.")
    parser_sf_get_files.add_argument("name", type=str, help="Name of the Smart Folder.")
    parser_sf_get_files.set_defaults(func=lambda args: handle_sf_operations(args, sf_system))

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
