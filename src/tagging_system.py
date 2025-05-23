import json
import os
from datetime import datetime
import torch
from sentence_transformers.util import pytorch_cos_sim
import yake # Added for keyword extraction

from .config_manager import ConfigManager
from .embedding_generation.generator import EmbeddingGenerator
from .content_extraction.extractor import ContentExtractor
from .file_discovery.discoverer import FileDiscoverer # Added for directory tagging

class TaggingSystem:
    def __init__(self):
        self.config = ConfigManager()
        self.tag_file_path = self.config.get_tag_file_path()
        self._load_tags()
        self.embedding_generator = EmbeddingGenerator()
        self.content_extractor = ContentExtractor()
        self.keyword_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=20, features=None)

    def _load_tags(self):
        """Loads tag data from the JSON file specified in the config."""
        if os.path.exists(self.tag_file_path):
            with open(self.tag_file_path, 'r') as f:
                try:
                    self.tags_data = json.load(f)
                    # Ensure essential keys exist
                    if "files" not in self.tags_data:
                        self.tags_data["files"] = {}
                    if "tags" not in self.tags_data:
                        self.tags_data["tags"] = {}
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from tag file: {self.tag_file_path}. Initializing empty tags.")
                    self.tags_data = {"files": {}, "tags": {}}
        else:
            self.tags_data = {"files": {}, "tags": {}}

    def _save_tags(self):
        """Saves the current tag data to the JSON file."""
        os.makedirs(os.path.dirname(self.tag_file_path), exist_ok=True)
        with open(self.tag_file_path, 'w') as f:
            json.dump(self.tags_data, f, indent=4)

    def add_tag_to_directory(self, directory_path: str, tag: str) -> int:
        """
        Adds a tag to all supported files found in the specified directory (recursively).

        Args:
            directory_path: The path to the directory.
            tag: The tag to add.
        
        Returns:
            The number of files successfully tagged.
        """
        directory_path = os.path.abspath(directory_path)
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at '{directory_path}'.")
            return 0

        discoverer = FileDiscoverer()
        tagged_files_count = 0
        print(f"Scanning directory '{directory_path}' to tag files with '{tag}'...")
        for file_path in discoverer.scan_directory(directory_path):
            # add_tag_to_file handles abspath and lowercasing the tag
            self.add_tag_to_file(file_path, tag)
            tagged_files_count += 1
        
        if tagged_files_count > 0:
            print(f"Finished tagging. Attempted to tag {tagged_files_count} files in '{directory_path}' with '{tag}'.")
        else:
            print(f"No files found or tagged in '{directory_path}'.")
        return tagged_files_count

    def add_tag_to_file(self, file_path: str, tag: str):
        """Adds a tag to a single file and updates tag metadata."""
        file_path = os.path.abspath(file_path)
        tag = tag.lower()
        timestamp = datetime.utcnow().isoformat()

        if file_path not in self.tags_data["files"]:
            self.tags_data["files"][file_path] = {"tags": [], "metadata": {}}
        
        if tag not in self.tags_data["files"][file_path]["tags"]:
            self.tags_data["files"][file_path]["tags"].append(tag)
            self.tags_data["files"][file_path]["last_tagged"] = timestamp

        if tag not in self.tags_data["tags"]:
            self.tags_data["tags"][tag] = {"files": [], "created_at": timestamp}
            default_model = self.config.get_default_model_name()
            try:
                tag_embedding = self.embedding_generator.generate_embeddings_for_text([tag], model_name=default_model)
                if tag_embedding is not None and len(tag_embedding) > 0:
                    self.tags_data["tags"][tag]["embedding_info"] = {
                        "model_name": default_model,
                        "embedding": tag_embedding[0].tolist()
                    }
            except Exception as e:
                print(f"Warning: Could not generate embedding for new tag '{tag}': {e}")
        
        if file_path not in self.tags_data["tags"][tag]["files"]:
            self.tags_data["tags"][tag]["files"].append(file_path)
            self.tags_data["tags"][tag]["last_updated"] = timestamp
        
        self._save_tags()
        print(f"Tagged '{file_path}' with '{tag}'.")

    def remove_tag_from_file(self, file_path: str, tag: str):
        """Removes a tag from a single file and updates tag metadata."""
        file_path = os.path.abspath(file_path)
        tag = tag.lower()
        timestamp = datetime.utcnow().isoformat()

        removed = False
        if file_path in self.tags_data["files"] and tag in self.tags_data["files"][file_path]["tags"]:
            self.tags_data["files"][file_path]["tags"].remove(tag)
            self.tags_data["files"][file_path]["last_untagged"] = timestamp
            removed = True

        if tag in self.tags_data["tags"] and file_path in self.tags_data["tags"][tag]["files"]:
            self.tags_data["tags"][tag]["files"].remove(file_path)
            self.tags_data["tags"][tag]["last_updated"] = timestamp
            removed = True
        
        if removed:
            self._save_tags()
            print(f"Removed tag '{tag}' from '{file_path}'.")
        else:
            print(f"Tag '{tag}' not found on '{file_path}'.")

    def get_tags_for_file(self, file_path: str) -> list[str]:
        """Retrieves all tags associated with a given file path."""
        file_path = os.path.abspath(file_path)
        return self.tags_data["files"].get(file_path, {}).get("tags", [])

    def get_files_for_tag(self, tag: str) -> list[str]:
        """Retrieves all file paths associated with a given tag."""
        tag = tag.lower()
        return self.tags_data["tags"].get(tag, {}).get("files", [])

    def list_all_tags(self) -> list[str]:
        """Lists all unique tags currently in the system."""
        return list(self.tags_data["tags"].keys())

    def get_all_tagged_files(self) -> list[str]:
        """Lists all file paths that have at least one tag."""
        return list(self.tags_data["files"].keys())

    def _suggest_tags_from_keywords(self, content: str, top_n: int = 5) -> list[str]:
        """
        Suggests tags by extracting keywords from text content using YAKE.
        Args:
            content: The text content to analyze.
            top_n: The maximum number of keyword-based tags to suggest.
        Returns:
            A list of suggested keyword strings (lowercase).
        """
        if not content:
            return []
        try:
            keywords_with_scores = self.keyword_extractor.extract_keywords(content)
            # YAKE returns keywords with scores (lower is better)
            # We just need the keyword strings
            suggested_keywords = [kw_score[0].lower() for kw_score in keywords_with_scores]
            return suggested_keywords[:top_n]
        except Exception as e:
            print(f"Error extracting keywords with YAKE: {e}")
            return []

    def suggest_tags_for_file(self, file_path: str, model_name_for_suggestion: str = None, top_n: int = 5) -> list[str]:
        """
        Suggests tags for a file based on its content using keyword extraction and semantic similarity to existing tags.

        Args:
            file_path: The absolute path to the file.
            model_name_for_suggestion: The embedding model to use for semantic suggestions.
                                       Defaults to the system's default model.
            top_n: The maximum number of combined suggestions to return.

        Returns:
            A list of suggested tag strings.
        """
        file_path = os.path.abspath(file_path)
        model_name = model_name_for_suggestion or self.config.get_default_model_name()

        try:
            content = self.content_extractor.extract_text(file_path)
            if not content:
                print(f"No content extracted from '{file_path}'. Cannot suggest tags.")
                return []
        except Exception as e:
            print(f"Error extracting content from '{file_path}': {e}")
            return []

        # 1. Keyword-based suggestions
        keyword_suggested_tags = self._suggest_tags_from_keywords(content, top_n=top_n)

        semantic_suggested_tags = []
        # 2. Semantic suggestions (if existing tags and content embedding are available)
        if self.tags_data["tags"]:
            try:
                file_embedding_list = self.embedding_generator.generate_embeddings_for_text([content], model_name=model_name)
                if file_embedding_list is not None and file_embedding_list.size > 0:
                    file_embedding = torch.tensor(file_embedding_list[0]).unsqueeze(0)
                    
                    tag_similarities = []
                    for tag_name, tag_data in self.tags_data["tags"].items():
                        tag_embedding_stored = None
                        tag_embedding_info = tag_data.get("embedding_info")

                        if tag_embedding_info and tag_embedding_info.get("model_name") == model_name:
                            tag_embedding_stored = torch.tensor(tag_embedding_info["embedding"]).unsqueeze(0)
                        else:
                            try:
                                temp_tag_embedding_list = self.embedding_generator.generate_embeddings_for_text([tag_name], model_name=model_name)
                                if temp_tag_embedding_list is not None and temp_tag_embedding_list.size > 0:
                                    tag_embedding_stored = torch.tensor(temp_tag_embedding_list[0]).unsqueeze(0)
                            except Exception as e_emb_gen:
                                print(f"Warning: Could not generate temporary embedding for tag '{tag_name}' with model '{model_name}': {e_emb_gen}")
                                continue

                        if tag_embedding_stored is not None:
                            try:
                                similarity = pytorch_cos_sim(file_embedding, tag_embedding_stored).item()
                                tag_similarities.append((tag_name, similarity))
                            except Exception as e:
                                print(f"Error calculating similarity for tag '{tag_name}': {e}")
                    
                    tag_similarities.sort(key=lambda item: item[1], reverse=True)
                    semantic_suggested_tags = [tag[0] for tag in tag_similarities[:top_n]]
                else:
                    print(f"Could not generate content embedding for '{os.path.basename(file_path)}' for semantic comparison.")
            except Exception as e:
                print(f"Error during semantic tag suggestion for '{os.path.basename(file_path)}': {e}")
        else:
            print("No existing tags in the system for semantic comparison. Relying primarily on keyword extraction.")

        # 3. Combine and deduplicate, prioritizing semantic then keyword tags
        combined_suggestions = []
        seen_tags = set()

        for tag in semantic_suggested_tags: # Semantic tags first (if any)
            if tag not in seen_tags:
                combined_suggestions.append(tag)
                seen_tags.add(tag)
        
        for tag in keyword_suggested_tags: # Then keyword tags
            if tag not in seen_tags:
                combined_suggestions.append(tag)
                seen_tags.add(tag)
        
        if not combined_suggestions:
            # Feedback moved to CLI for better user experience control
            return []

        return combined_suggestions[:top_n]

    def suggest_tags_for_directory(self, directory_path: str, model_name_for_suggestion: str = None, top_n: int = 5) -> dict[str, list[str]]:
        """
        Suggests tags for all supported files found in the specified directory (recursively).

        Args:
            directory_path: The path to the directory.
            model_name_for_suggestion: The model to use for suggestions. Defaults to config default.
            top_n: Number of suggestions per file. Defaults to 5.

        Returns:
            A dictionary where keys are file paths and values are lists of suggested tags.
        """
        directory_path = os.path.abspath(directory_path)
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at '{directory_path}'.")
            return {}

        discoverer = FileDiscoverer() # Assumes default skip_hidden=True
        all_suggestions = {}
        print(f"Scanning directory '{directory_path}' to suggest tags for files...")
        
        files_found = list(discoverer.scan_directory(directory_path))
        if not files_found:
            print(f"No files found in '{directory_path}'.")
            return {}

        for i, file_path in enumerate(files_found):
            print(f"  Suggesting for file {i+1}/{len(files_found)}: '{os.path.basename(file_path)}'...")
            suggestions = self.suggest_tags_for_file(file_path, model_name_for_suggestion, top_n)
            all_suggestions[file_path] = suggestions
            # Individual file suggestion already prints details; consider a quieter mode if needed.
        
        print(f"Finished suggesting tags for files in '{directory_path}'.")
        return all_suggestions

# Example Usage (for testing purposes)
if __name__ == '__main__':
    ts = TaggingSystem()
    
    if not os.path.exists(ts.config.config_path):
        dummy_config_path = ts.config.config_path
        os.makedirs(os.path.dirname(dummy_config_path), exist_ok=True)
        with open(dummy_config_path, 'w') as f:
            json.dump({
                "default_model_name": "all-MiniLM-L6-v2",
                "index_base_directory": "./semantic_indices",
                "tag_file_path": "./semantic_tags_test.json"
            }, f)
        ts = TaggingSystem()

    file1 = "/tmp/test_file_1.txt"
    file2 = "/tmp/test_file_2.md"
    os.makedirs("/tmp", exist_ok=True)
    open(file1, 'a').close()
    open(file2, 'a').close()

    ts.add_tag_to_file(file1, "important")
    ts.add_tag_to_file(file1, "projectA")
    ts.add_tag_to_file(file2, "projectA")
    ts.add_tag_to_file(file2, "urgent")

    print("\nTags for file1:", ts.get_tags_for_file(file1))
    print("Tags for file2:", ts.get_tags_for_file(file2))

    print("\nFiles for tag 'projectA':", ts.get_files_for_tag("projectA"))
    print("Files for tag 'important':", ts.get_files_for_tag("important"))
    
    print("\nAll tags:", ts.list_all_tags())
    print("All tagged files:", ts.get_all_tagged_files())

    ts.remove_tag_from_file(file1, "important")
    print("\nTags for file1 after removal:", ts.get_tags_for_file(file1))
    print("Files for tag 'important' after removal:", ts.get_files_for_tag("important"))
    
    ts.remove_tag_from_file(file2, "nonexistenttag")

    if os.path.exists(file1): os.remove(file1)
    if os.path.exists(file2): os.remove(file2)
    if ts.tag_file_path == "./semantic_tags_test.json" and os.path.exists(ts.tag_file_path):
        os.remove(ts.tag_file_path)
    if ts.config.config_path.endswith("_test.json") and os.path.exists(ts.config.config_path):
         os.remove(ts.config.config_path)
