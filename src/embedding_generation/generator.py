from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import os

# Ensure src directory is in path to allow sibling imports
PROJECT_ROOT_FOR_EMBEDDING = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT_FOR_EMBEDDING)

from src.config_manager import ConfigManager

class EmbeddingGenerator:
    def __init__(self):
        self._loaded_models = {} # Cache for loaded models
        self._config_manager = ConfigManager()

    def get_model(self, model_name: str = None) -> SentenceTransformer | None:
        """
        Loads a SentenceTransformer model. Caches models after first load.
        If model_name is None, uses the default from config.

        Args:
            model_name: Name of the Sentence Transformer model.

        Returns:
            The model instance or None if loading fails.
        """
        if model_name is None:
            model_name = self._config_manager.get_default_model_name()

        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        print(f"Loading sentence transformer model: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            self._loaded_models[model_name] = model
            print(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading sentence transformer model {model_name}: {e}")
            return None

    def get_embedding_dimension(self, model_name: str = None) -> int | None:
        """
        Gets the embedding dimension for a given SentenceTransformer model.
        If model_name is None, uses the default from config.

        Args:
            model_name: Name of the Sentence Transformer model.

        Returns:
            The embedding dimension or None if an error occurs.
        """
        if model_name is None:
            model_name = self._config_manager.get_default_model_name()
        model = self.get_model(model_name)
        if model:
            try:
                return model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Error getting embedding dimension for model {model_name}: {e}")
                return None
        return None

    def generate_embeddings_for_text(self, texts: list[str], model_name: str = None) -> np.ndarray | None:
        """
        Generates embeddings for a list of texts using the specified model.
        If model_name is None, uses the default from config.

        Args:
            texts: A list of text strings.
            model_name: Name of the Sentence Transformer model.

        Returns:
            A NumPy array representing the embeddings, or None if an error occurs.
            Returns an empty NumPy array if the input texts list is empty or all texts are invalid.
        """
        if model_name is None:
            model_name = self._config_manager.get_default_model_name()

        model = self.get_model(model_name)
        if model is None:
            print(f"Embedding model {model_name} is not available.")
            return None
        
        if not texts:
            return np.array([]) # Return empty numpy array for empty list
            
        # Filter out empty or whitespace-only strings before encoding
        valid_texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
        if not valid_texts:
            return np.array([]) # Return empty numpy array if all texts were invalid
            
        try:
            embeddings = model.encode(valid_texts)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings with model {model_name}: {e}")
            return None

if __name__ == '__main__':
    print("--- Testing Embedding Generation ---")

    generator = EmbeddingGenerator()

    DEFAULT_MODEL_NAME_FROM_CONFIG = generator._config_manager.get_default_model_name()
    print(f"Default model from config: {DEFAULT_MODEL_NAME_FROM_CONFIG}")

    dim_default = generator.get_embedding_dimension()
    if dim_default:
        print(f"Dimension of default model ({DEFAULT_MODEL_NAME_FROM_CONFIG}): {dim_default}")

    sample_texts_1 = ["This is a test sentence for embedding generation.", "Another sentence here.", "  "] # Added an empty string
    embeddings_1 = generator.generate_embeddings_for_text(sample_texts_1)
    if embeddings_1 is not None and embeddings_1.size > 0:
        print(f"\\nSample texts 1 (default model): {sample_texts_1}")
        print(f"  Embeddings shape: {embeddings_1.shape}") # Should be (2, N) if one string was invalid/empty
    elif embeddings_1 is not None and embeddings_1.size == 0:
        print("Generated empty embeddings for sample_texts_1, all texts might have been invalid or list was empty.")
    else:
        print("Failed to generate embeddings for sample_texts_1.")

    TEST_OTHER_MODEL = 'sentence-transformers/all-MiniLM-L12-v2' 
    print(f"\n--- Testing with a different model: {TEST_OTHER_MODEL} ---")
    dim_other = generator.get_embedding_dimension(TEST_OTHER_MODEL)
    if dim_other:
        print(f"Dimension of model {TEST_OTHER_MODEL}: {dim_other}")
    
    sample_texts_2 = ["Testing another model."]
    embeddings_2 = generator.generate_embeddings_for_text(sample_texts_2, model_name=TEST_OTHER_MODEL)
    if embeddings_2 is not None and embeddings_2.size > 0:
        print(f"\nSample texts 2 ({TEST_OTHER_MODEL}): {sample_texts_2}")
        print(f"  Embeddings shape: {embeddings_2.shape}")

    # Test with empty list
    print("\n--- Testing with empty list ---")
    empty_embeddings = generator.generate_embeddings_for_text([])
    if empty_embeddings is not None and empty_embeddings.size == 0:
        print("Correctly returned empty array for empty list input.")
    else:
        print(f"Error: Did not return empty array for empty list. Got: {empty_embeddings}")

    # Test with list of only invalid texts
    print("\n--- Testing with list of only invalid texts ---")
    invalid_texts_embeddings = generator.generate_embeddings_for_text([None, " ", "\t"])
    if invalid_texts_embeddings is not None and invalid_texts_embeddings.size == 0:
        print("Correctly returned empty array for list of only invalid texts.")
    else:
        print(f"Error: Did not return empty array for list of only invalid texts. Got: {invalid_texts_embeddings}")
