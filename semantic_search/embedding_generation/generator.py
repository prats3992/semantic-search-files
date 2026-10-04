from sentence_transformers import SentenceTransformer
import numpy as np

from ..config_manager import ConfigManager

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
