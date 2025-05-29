import json
import os
from dotenv import load_dotenv # Added for .env support

CONFIG_FILE_NAME = "config.json"
# Determine project root assuming this file is in src/ or a similar subdirectory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, CONFIG_FILE_NAME)

# Load .env file from project root
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(DOTENV_PATH)

class ConfigManager:
    """Manages application configuration settings loaded from a JSON file."""
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initializes the ConfigManager.

        Args:
            config_path: Path to the configuration file. Defaults to 'config.json' in the project root.
        """
        self.config_path = config_path
        self._config_data = None
        self._load_config()

    def _load_config(self):
        """Loads configuration from the JSON file. Handles file not found and JSON decoding errors."""
        try:
            with open(self.config_path, 'r') as f:
                self._config_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {self.config_path}. Using fallback defaults.")
            self._config_data = {
                "default_model_name": "all-MiniLM-L6-v2",
                "index_base_directory": "./semantic_indices",
                "recommended_models": [],
                "tag_file_path": "./semantic_tags.json"
            }
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.config_path}. Check for syntax errors. Using fallback defaults.")
            self._config_data = {
                "default_model_name": "all-MiniLM-L6-v2",
                "index_base_directory": "./semantic_indices",
                "recommended_models": [],
                "tag_file_path": "./semantic_tags.json"
            }
        except Exception as e:
            print(f"An unexpected error occurred while loading config: {e}. Using fallback defaults.")
            self._config_data = {
                "default_model_name": "all-MiniLM-L6-v2",
                "index_base_directory": "./semantic_indices",
                "recommended_models": [],
                "tag_file_path": "./semantic_tags.json"
            }

    def get_setting(self, key: str, default_value=None):
        """Retrieves a configuration setting by key, returning a default value if not found."""
        return self._config_data.get(key, default_value)

    def get_default_model_name(self) -> str:
        """Returns the default sentence transformer model name from the configuration."""
        return self.get_setting("default_model_name", "all-MiniLM-L6-v2")

    def get_index_base_directory(self) -> str:
        """Returns the absolute path to the base directory for storing FAISS indexes."""
        base_dir = self.get_setting("index_base_directory", "./semantic_indices")
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(PROJECT_ROOT, base_dir)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def get_tag_file_path(self) -> str:
        """Returns the absolute path to the JSON file used for storing tags."""
        tag_file = self.get_setting("tag_file_path", "./semantic_tags.json")
        if not os.path.isabs(tag_file) and not tag_file.startswith("./"): # ensure it's either absolute or clearly relative
             tag_file = os.path.join(PROJECT_ROOT, tag_file)
        elif tag_file.startswith("./"): # Relative to project root
            tag_file = os.path.join(PROJECT_ROOT, tag_file[2:])
        
        # Ensure the directory for the tag file exists
        tag_file_dir = os.path.dirname(tag_file)
        if tag_file_dir: # Create directory only if it's not the current directory (e.g. for "tags.json")
            os.makedirs(tag_file_dir, exist_ok=True)
        return tag_file

    def get_smart_dirs_file_path(self) -> str:
        """Returns the absolute path to the JSON file used for storing smart directory definitions."""
        smart_dirs_file = self.get_setting("smart_dirs_file_path", "./smart_directories.json")
        if not os.path.isabs(smart_dirs_file) and not smart_dirs_file.startswith("./"): # ensure it's either absolute or clearly relative
             smart_dirs_file = os.path.join(PROJECT_ROOT, smart_dirs_file)
        elif smart_dirs_file.startswith("./"): # Relative to project root
            smart_dirs_file = os.path.join(PROJECT_ROOT, smart_dirs_file[2:])
        
        # Ensure the directory for the smart directories file exists
        smart_dirs_file_dir = os.path.dirname(smart_dirs_file)
        if smart_dirs_file_dir: # Create directory only if it's not the current directory
            os.makedirs(smart_dirs_file_dir, exist_ok=True)
        return smart_dirs_file

    def get_recommended_models(self) -> list:
        """Returns a list of recommended sentence transformer models from the configuration."""
        return self.get_setting("recommended_models", [])

    def get_embedding_dimension(self, model_name: str) -> int | None:
        """
        Retrieves the embedding dimension for a given model name by looking it up
        in the 'recommended_models' list in the configuration.

        Args:
            model_name: The name of the model (e.g., 'all-MiniLM-L6-v2').

        Returns:
            The dimension as an int if the model is found and has a dimension specified,
            otherwise None.
        """
        recommended_models = self.get_recommended_models()
        for model_info in recommended_models:
            if model_info.get('name') == model_name:
                return model_info.get('dimension')
        # Fallback if model not in recommended list or dimension not specified there
        # This part might be better handled by trying to load the model if not found,
        # but for config, it relies on the config entry.
        print(f"Warning: Model '{model_name}' not found in recommended_models list in config or dimension not specified there. Cannot determine dimension from config.")
        return None

    def get_gemini_api_key(self) -> str | None:
        """Returns the Gemini API key from config.json or .env file."""
        api_key = self.get_setting("gemini_api_key")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in config.json or .env file.")
            return None
        return api_key

# For standalone testing of the ConfigManager class
if __name__ == '__main__':
    print("--- Testing Configuration Manager Class ---")
    
    # Test with default path
    print("\n--- Test 1: Default Config Path ---")
    config_manager = ConfigManager()
    print(f"Loaded config from: {config_manager.config_path}")
    print(f"Default model: {config_manager.get_default_model_name()}")
    print(f"Index base directory (absolute): {config_manager.get_index_base_directory()}")
    print(f"Tag file path (absolute): {config_manager.get_tag_file_path()}")
    print(f"Smart dirs file path (absolute): {config_manager.get_smart_dirs_file_path()}")
    recommended = config_manager.get_recommended_models()
    if recommended:
        print("\nRecommended Models:")
        for model_info in recommended:
            print(f"  - Name: {model_info.get('name')}")
    else:
        print("No recommended models found in config.")

    # Test with a non-existent custom path to check fallback
    print("\n--- Test 2: Non-existent Custom Config Path (should use fallback) ---")
    non_existent_path = os.path.join(PROJECT_ROOT, "non_existent_config.json")
    custom_config_manager = ConfigManager(config_path=non_existent_path)
    print(f"Attempted to load config from: {custom_config_manager.config_path}")
    print(f"Default model (fallback): {custom_config_manager.get_default_model_name()}")
    print(f"Tag file path (fallback, absolute): {custom_config_manager.get_tag_file_path()}")

    # Test creating a dummy config and loading it
    print("\n--- Test 3: Dummy Custom Config File ---")
    dummy_config_content = {
        "default_model_name": "custom-model-test",
        "index_base_directory": "./custom_indices_test",
        "tag_file_path": "./custom_tags_test.json",
        "smart_dirs_file_path": "./custom_smart_dirs_test.json",
        "recommended_models": [{"name": "custom-rec-model"}]
    }
    dummy_config_path = os.path.join(PROJECT_ROOT, "dummy_test_config.json")
    
    # Ensure directory for dummy config exists if it's nested (not the case here)
    # os.makedirs(os.path.dirname(dummy_config_path), exist_ok=True)

    with open(dummy_config_path, 'w') as f:
        json.dump(dummy_config_content, f, indent=4)
    
    dummy_manager = ConfigManager(config_path=dummy_config_path)
    print(f"Loaded config from: {dummy_manager.config_path}")
    print(f"Default model: {dummy_manager.get_default_model_name()}")
    print(f"Index base directory (absolute): {dummy_manager.get_index_base_directory()}")
    print(f"Tag file path (absolute): {dummy_manager.get_tag_file_path()}")
    print(f"Smart dirs file path (absolute): {dummy_manager.get_smart_dirs_file_path()}")
    print(f"Recommended: {dummy_manager.get_recommended_models()}")

    # Clean up dummy config file
    if os.path.exists(dummy_config_path):
        os.remove(dummy_config_path)
    
    # Clean up directories created by get_index_base_directory and get_tag_file_path
    # if they are test-specific and empty.
    custom_index_dir_test = os.path.join(PROJECT_ROOT, "custom_indices_test")
    if os.path.exists(custom_index_dir_test):
        # Be careful with rmtree, ensure it's the correct test directory
        import shutil
        # shutil.rmtree(custom_index_dir_test) 
        print(f"Cleanup: Test index directory '{custom_index_dir_test}' might need manual removal or use shutil.rmtree carefully.")

    custom_tag_file_test = os.path.join(PROJECT_ROOT, "custom_tags_test.json")
    custom_tag_dir_test = os.path.dirname(custom_tag_file_test) # Should be PROJECT_ROOT if file is at root
    if os.path.exists(custom_tag_file_test):
        os.remove(custom_tag_file_test)
        print(f"Cleaned up dummy tag file: {custom_tag_file_test}")
    # Remove directory only if it was created for the test and is now empty
    if custom_tag_dir_test and custom_tag_dir_test != PROJECT_ROOT and os.path.exists(custom_tag_dir_test) and not os.listdir(custom_tag_dir_test):
        os.rmdir(custom_tag_dir_test)
        print(f"Cleaned up empty dummy tag directory: {custom_tag_dir_test}")

    custom_smart_dirs_file_test = os.path.join(PROJECT_ROOT, "custom_smart_dirs_test.json")
    custom_smart_dirs_dir_test = os.path.dirname(custom_smart_dirs_file_test)
    if os.path.exists(custom_smart_dirs_file_test):
        os.remove(custom_smart_dirs_file_test)
        print(f"Cleaned up dummy smart_dirs file: {custom_smart_dirs_file_test}")
    if custom_smart_dirs_dir_test and custom_smart_dirs_dir_test != PROJECT_ROOT and os.path.exists(custom_smart_dirs_dir_test) and not os.listdir(custom_smart_dirs_dir_test):
        os.rmdir(custom_smart_dirs_dir_test)
        print(f"Cleaned up empty dummy smart_dirs directory: {custom_smart_dirs_dir_test}")
