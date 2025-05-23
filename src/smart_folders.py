import json
import os
from datetime import datetime
from .config_manager import ConfigManager
from .tagging_system import TaggingSystem
from .embedding_generation.generator import EmbeddingGenerator
from .index_storage.faiss_indexer import FaissIndexer
from .file_discovery.discoverer import FileDiscoverer
from .content_extraction.extractor import ContentExtractor

class SmartFolderSystem:
    """
    Manages Smart Folders, which are dynamic collections of files based on defined criteria.
    Criteria can include tags, file types, semantic queries, and content keywords.
    """
    def __init__(self):
        self.config = ConfigManager()
        self.smart_dirs_file_path = self.config.get_smart_dirs_file_path()
        self.tagging_system = TaggingSystem()
        self.embedding_generator = EmbeddingGenerator()
        self.content_extractor = ContentExtractor()
        self.file_discoverer = FileDiscoverer() # Uses default skip_hidden=True
        self._load_smart_folders()

    def _load_smart_folders(self):
        """Loads smart folder definitions from the JSON file specified in the config."""
        if os.path.exists(self.smart_dirs_file_path):
            with open(self.smart_dirs_file_path, 'r') as f:
                try:
                    self.smart_folders = json.load(f)
                except json.JSONDecodeError:
                    self.smart_folders = {}
        else:
            self.smart_folders = {}

    def _save_smart_folders(self):
        """Saves the current smart folder definitions to the JSON file."""
        os.makedirs(os.path.dirname(self.smart_dirs_file_path), exist_ok=True)
        with open(self.smart_dirs_file_path, 'w') as f:
            json.dump(self.smart_folders, f, indent=4)

    def create_smart_folder(self, name: str, criteria: dict) -> bool:
        """
        Creates a new smart folder with a given name and criteria.

        Args:
            name: The unique name for the smart folder.
            criteria: A dictionary defining the rules for including files. Example:
                {
                    "base_directories": ["/path/to/search1", "./project_docs"], (Required)
                    "tags_all_of": ["python", "important"], (Optional)
                    "tags_any_of": ["urgent", "review"],   (Optional)
                    "tags_none_of": ["archive"],            (Optional)
                    "semantic_query": "machine learning concepts", (Optional)
                    "semantic_model": "all-MiniLM-L6-v2", (Optional, defaults to config)
                    "semantic_threshold": 0.5, (Optional, similarity threshold, default 0.5)
                    "file_types": [".txt", ".md"], (Optional, list of extensions like '.txt')
                    "content_keywords_all_of": ["keyword1", "keyword2"], (Optional)
                    "content_keywords_any_of": ["option1", "option2"]    (Optional)
                }
        Returns:
            True if creation was successful, False otherwise.
        """
        if name in self.smart_folders:
            print(f"Error: Smart folder '{name}' already exists.")
            return False
        
        # Basic validation of criteria structure (can be expanded)
        if not isinstance(criteria, dict):
            print("Error: Criteria must be a dictionary.")
            return False
        if not criteria.get("base_directories") or not isinstance(criteria["base_directories"], list):
             print("Error: `base_directories` (list of paths to search) is a required criterion.")
             return False

        self.smart_folders[name] = {
            "name": name,
            "criteria": criteria,
            "created_at": datetime.utcnow().isoformat()
        }
        self._save_smart_folders()
        print(f"Smart folder '{name}' created successfully.")
        return True

    def remove_smart_folder(self, name: str) -> bool:
        """
        Removes a smart folder definition.

        Args:
            name: The name of the smart folder to remove.
        
        Returns:
            True if removal was successful, False otherwise.
        """
        if name not in self.smart_folders:
            print(f"Error: Smart folder '{name}' not found.")
            return False
        del self.smart_folders[name]
        self._save_smart_folders()
        print(f"Smart folder '{name}' removed.")
        return True

    def list_smart_folders(self) -> list[str]:
        """
        Lists the names of all currently defined smart folders.

        Returns:
            A list of smart folder names.
        """
        if not self.smart_folders:
            print("No smart folders defined yet.")
            return []
        print("Available Smart Folders:")
        for name, data in self.smart_folders.items():
            print(f"  - {name} (Created: {data.get('created_at', 'N/A')})")
        return list(self.smart_folders.keys())

    def get_smart_folder_criteria(self, name: str) -> dict | None:
        """
        Retrieves the criteria for a specific smart folder.

        Args:
            name: The name of the smart folder.

        Returns:
            The criteria dictionary if the folder exists, otherwise None.
        """
        folder_data = self.smart_folders.get(name)
        if not folder_data:
            print(f"Error: Smart folder '{name}' not found.")
            return None
        return folder_data.get("criteria")

    def _get_faiss_indexer(self, model_name: str) -> FaissIndexer | None:
        """
        Initializes and returns a FaissIndexer for a given model, loading an existing index if available.
        This is a helper primarily for semantic search within smart folders.
        """
        dimension = self.config.get_embedding_dimension(model_name)
        if dimension is None:
            print(f"Error: Could not determine embedding dimension for model {model_name}.")
            return None
        
        index_base_dir = self.config.get_index_base_directory()
        model_specific_index_dir = os.path.join(index_base_dir, model_name.replace('/', '_'))
        
        index_file_path = os.path.join(model_specific_index_dir, "semantic_index.faiss")
        metadata_file_path = os.path.join(model_specific_index_dir, "semantic_metadata.pkl")

        # The FaissIndexer constructor will try to load an existing index if files are present,
        # or initialize an empty one if not.
        # We print a warning if the index files are not found, as this means semantic search
        # will operate on an empty or incomplete index for this model.
        if not os.path.exists(index_file_path) or not os.path.exists(metadata_file_path):
            print(f"Warning: Index for model '{model_name}' not found at '{model_specific_index_dir}'. "
                  f"Semantic search for this smart folder might use an empty or newly initialized index.")
            
        return FaissIndexer(dimension=dimension, index_file_path=index_file_path, metadata_file_path=metadata_file_path)


    def get_smart_folder_files(self, name: str) -> list[str]:
        """
        Resolves a smart folder by its name and returns a list of absolute file paths
        that match its criteria.

        Args:
            name: The name of the smart folder.

        Returns:
            A sorted list of absolute file paths matching the criteria.
            Returns an empty list if the folder is not found or no files match.
        """
        folder_data = self.smart_folders.get(name)
        if not folder_data:
            print(f"Error: Smart folder '{name}' not found.")
            return []

        criteria = folder_data["criteria"]
        base_directories = [os.path.abspath(p) for p in criteria.get("base_directories", [])]
        
        if not base_directories:
            print(f"Error: Smart folder '{name}' has no base_directories defined to search within.")
            return []

        print(f"Fetching files for smart folder '{name}'...")
        print(f"  Criteria: {json.dumps(criteria, indent=2)}")

        candidate_files = set()
        for base_dir in base_directories:
            if os.path.isdir(base_dir):
                for file_path in self.file_discoverer.scan_directory(base_dir):
                    candidate_files.add(os.path.abspath(file_path))
            elif os.path.isfile(base_dir): # Allow individual files in base_directories
                 candidate_files.add(base_dir)
            else:
                print(f"Warning: Path '{base_dir}' in base_directories is not a valid file or directory. Skipping.")
        
        if not candidate_files:
            print("No files found in the specified base_directories.")
            return []

        print(f"  Initial candidates from base directories: {len(candidate_files)} files.")
        
        # Apply filters
        # 1. File type filter
        file_types = criteria.get("file_types")
        if file_types and isinstance(file_types, list):
            filtered_files = set()
            for f_path in candidate_files:
                if any(f_path.lower().endswith(ft.lower()) for ft in file_types): # Case-insensitive extension check
                    filtered_files.add(f_path)
            candidate_files = filtered_files
            print(f"  After file_type filter: {len(candidate_files)} files.")

        # 2. Tag filters
        tags_all_of = set(t.lower() for t in criteria.get("tags_all_of", []) if isinstance(t, str))
        tags_any_of = set(t.lower() for t in criteria.get("tags_any_of", []) if isinstance(t, str))
        tags_none_of = set(t.lower() for t in criteria.get("tags_none_of", []) if isinstance(t, str))

        if tags_all_of or tags_any_of or tags_none_of:
            filtered_files = set()
            for f_path in candidate_files:
                file_tags = set(self.tagging_system.get_tags_for_file(f_path))
                passes_all_of = not tags_all_of or tags_all_of.issubset(file_tags)
                passes_any_of = not tags_any_of or bool(tags_any_of.intersection(file_tags))
                passes_none_of = not tags_none_of or not bool(tags_none_of.intersection(file_tags))
                if passes_all_of and passes_any_of and passes_none_of:
                    filtered_files.add(f_path)
            candidate_files = filtered_files
            print(f"  After tag filters: {len(candidate_files)} files.")

        # 3. Content keyword filters
        keywords_all_of = [kw.lower() for kw in criteria.get("content_keywords_all_of", []) if isinstance(kw, str)]
        keywords_any_of = [kw.lower() for kw in criteria.get("content_keywords_any_of", []) if isinstance(kw, str)]

        if keywords_all_of or keywords_any_of:
            filtered_files = set()
            for f_path in candidate_files:
                try:
                    content = self.content_extractor.extract_text(f_path)
                    if content:
                        content_lower = content.lower()
                        passes_kw_all = not keywords_all_of or all(kw in content_lower for kw in keywords_all_of)
                        passes_kw_any = not keywords_any_of or any(kw in content_lower for kw in keywords_any_of)
                        if passes_kw_all and passes_kw_any:
                            filtered_files.add(f_path)
                except Exception as e:
                    print(f"Warning: Could not extract content from '{f_path}' for keyword check: {e}")
            candidate_files = filtered_files
            print(f"  After content keyword filters: {len(candidate_files)} files.")

        # 4. Semantic search filter (applied last as it's potentially the most expensive)
        semantic_query = criteria.get("semantic_query")
        if semantic_query and isinstance(semantic_query, str) and candidate_files:
            model_name = criteria.get("semantic_model") or self.config.get_default_model_name()
            effective_threshold = criteria.get("semantic_threshold", 0.5)

            indexer = self._get_faiss_indexer(model_name)
            
            if not indexer:
                print(f"Warning: Could not initialize indexer for model '{model_name}'. Skipping semantic filter.")
            else:
                query_embedding_array = self.embedding_generator.generate_embeddings_for_text([semantic_query], model_name=model_name)
                
                if query_embedding_array is None or query_embedding_array.size == 0:
                    print(f"Warning: Could not generate embedding for semantic query '{semantic_query}'. Skipping semantic filter.")
                else:
                    query_embedding_single = query_embedding_array[0] # First (and only) embedding for the query
                    semantically_matched_files = set()

                    embeddings_to_compare = []
                    paths_for_embeddings_to_compare = []
                    files_needing_fresh_embedding = [] # Files not found in the current model's index

                    print(f"  Optimized semantic check for {len(candidate_files)} candidates (model: '{model_name}')...")
                    # Check existing index first
                    for f_path in candidate_files:
                        retrieved_embedding = indexer.get_embedding_by_filepath(f_path)
                        if retrieved_embedding is not None:
                            embeddings_to_compare.append(retrieved_embedding)
                            paths_for_embeddings_to_compare.append(f_path)
                        else:
                            files_needing_fresh_embedding.append(f_path)
                    
                    if embeddings_to_compare:
                        print(f"    Retrieved {len(embeddings_to_compare)} embeddings from index for model '{model_name}'.")

                    if files_needing_fresh_embedding:
                        print(f"    Generating fresh embeddings for {len(files_needing_fresh_embedding)} files not in index for model '{model_name}'...")
                        
                        contents_to_embed = []
                        paths_for_generated_contents = []
                        for f_path in files_needing_fresh_embedding:
                            try:
                                content = self.content_extractor.extract_text(f_path)
                                if content and content.strip():
                                    contents_to_embed.append(content)
                                    paths_for_generated_contents.append(f_path)
                            except Exception as e:
                                print(f"Warning: Could not extract content from '{f_path}' for semantic check: {e}")
                        
                        if contents_to_embed:
                            generated_embeddings_array = self.embedding_generator.generate_embeddings_for_text(
                                contents_to_embed, model_name=model_name
                            )
                            if generated_embeddings_array is not None and generated_embeddings_array.shape[0] == len(paths_for_generated_contents):
                                for i, emb in enumerate(generated_embeddings_array):
                                    embeddings_to_compare.append(emb) # Add newly generated embedding
                                    paths_for_embeddings_to_compare.append(paths_for_generated_contents[i])
                                print(f"    Successfully generated {generated_embeddings_array.shape[0]} new embeddings.")
                            else:
                                print(f"Warning: Embedding generation for {len(contents_to_embed)} files failed or returned unexpected shape.")
                    
                    if embeddings_to_compare:
                        from sentence_transformers.util import pytorch_cos_sim
                        import torch
                        import numpy as np

                        # Prepare tensors for similarity calculation
                        query_emb_tensor = torch.tensor(query_embedding_single).unsqueeze(0) # Shape: [1, D]
                        
                        valid_candidate_embeddings_np = []
                        valid_paths_for_comparison = []
                        
                        expected_dim = query_embedding_single.shape[0]
                        for i, emb in enumerate(embeddings_to_compare):
                            if isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.shape[0] == expected_dim:
                                valid_candidate_embeddings_np.append(emb)
                                valid_paths_for_comparison.append(paths_for_embeddings_to_compare[i])
                            else:
                                path_for_warning = paths_for_embeddings_to_compare[i] if i < len(paths_for_embeddings_to_compare) else "unknown path"
                                emb_shape_info = emb.shape if isinstance(emb, np.ndarray) else type(emb)
                                print(f"Warning: Invalid or mismatched dimension embedding for path {path_for_warning}. Expected dim {expected_dim}, got {emb_shape_info}. Skipping.")

                        if not valid_candidate_embeddings_np:
                            print("  No valid candidate embeddings found or generated for semantic comparison.")
                        else:
                            candidate_embs_tensor = torch.tensor(np.array(valid_candidate_embeddings_np)) # Shape: [N, D]
                        
                            try:
                                similarities = pytorch_cos_sim(query_emb_tensor, candidate_embs_tensor).squeeze(0) # Should be [N]
                                
                                for i, f_path in enumerate(valid_paths_for_comparison):
                                    sim = similarities[i].item()
                                    if sim >= effective_threshold:
                                        semantically_matched_files.add(f_path)
                            except RuntimeError as e_sim:
                                print(f"Error during similarity calculation: {e_sim}. This might be due to empty tensors or dimension mismatches not caught earlier.")


                    candidate_files = semantically_matched_files
                    print(f"  After semantic filter ('{semantic_query}', model: '{model_name}', threshold: {effective_threshold:.2f}): {len(candidate_files)} files.")
        
        print(f"Smart folder '{name}' resolved to {len(candidate_files)} files.")
        return sorted(list(candidate_files)) # Return sorted list of unique file paths

if __name__ == '__main__':
    # Illustrative example usage.
    # Requires:
    #   - `config.json` to be set up (especially `smart_dirs_file_path`, `tag_file_path`, `index_base_directory`).
    #   - Potentially, an existing FAISS index for the model used in semantic queries.
    #   - Some tagged files and directories (`test_data` in this project).
    print("--- SmartFolderSystem Test (Illustrative) ---")
    sfs = SmartFolderSystem()

    # 0. Ensure config.json has smart_dirs_file_path, e.g., "./smart_directories_test.json"
    #    and that tag_file_path points to a valid tag file (even if empty).
    #    And that an index exists for the model used in semantic queries.

    # 1. Create a smart folder
    test_folder_name = "Python Projects Docs"
    test_criteria = {
        "base_directories": ["./test_data/test_data_1", "./test_data/test_data_2"], # Search in these places
        "tags_all_of": ["python"],          # Must have 'python' tag
        "file_types": [".txt", ".md"],       # Only these file types
        "semantic_query": "advanced python features", # Related to this
        "semantic_model": "all-MiniLM-L6-v2", # Specify model
        "semantic_threshold": 0.1 # Low threshold for testing
    }
    # sfs.create_smart_folder(test_folder_name, test_criteria)

    # 2. List smart folders
    # print("\n--- Listing Smart Folders ---")
    # sfs.list_smart_folders()

    # 3. Get files for the smart folder
    # print(f"\n--- Getting files for '{test_folder_name}' ---")
    # files = sfs.get_smart_folder_files(test_folder_name)
    # if files:
    #     print(f"Files in '{test_folder_name}':")
    #     for f in files:
    #         print(f"  - {f}")
    # else:
    #     print(f"No files found for '{test_folder_name}'. Ensure criteria match existing files, tags, and indexed content.")

    # print("\n--- Another Example: Files tagged 'python' in test_data_1 ---")
    # sfs.create_smart_folder(
    #     "Python Files in TestData1",
    #     {
    #         "base_directories": ["./test_data/test_data_1"],
    #         "tags_any_of": ["python"] # Assuming files in test_data_1 might be tagged 'python'
    #     }
    # )
    # files_py_test1 = sfs.get_smart_folder_files("Python Files in TestData1")
    # if files_py_test1:
    #     print("Files for 'Python Files in TestData1':")
    #     for f in files_py_test1: print(f"  - {f}")
    # else:
    #     print("No files found for 'Python Files in TestData1'.")

    # # Cleanup: Remove test smart folders if they were created by this example run
    # # This requires smart_dirs_file_path to be something like "./smart_directories_test.json" in config
    # # if sfs.smart_dirs_file_path.endswith("_test.json"):
    # #     if os.path.exists(sfs.smart_dirs_file_path):
    # #         os.remove(sfs.smart_dirs_file_path)
    # #         print(f"\\nCleaned up test smart folder file: {sfs.smart_dirs_file_path}")
    # # else:
    # #     print(f"\\nNote: Test smart folders may have been saved to: {sfs.smart_dirs_file_path}")
    pass
