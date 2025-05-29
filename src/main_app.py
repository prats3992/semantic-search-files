from .file_discovery.discoverer import FileDiscoverer
from .content_extraction.extractor import ContentExtractor # Changed import
from .metadata_extraction.extractor import get_file_metadata
from .embedding_generation.generator import generate_embedding, MODEL_NAME
from sentence_transformers import SentenceTransformer
from index_storage.faiss_indexer import FaissIndexer
import os

# Determine embedding dimension dynamically from the model
EMBEDDING_DIMENSION = 0
try:
    # Attempt to load the model once to get its dimension.
    temp_model = SentenceTransformer(MODEL_NAME) 
    EMBEDDING_DIMENSION = temp_model.get_sentence_embedding_dimension()
    del temp_model # Release memory
    print(f"Successfully determined embedding dimension: {EMBEDDING_DIMENSION} for model {MODEL_NAME}")
except Exception as e:
    print(f"Error determining embedding dimension for model {MODEL_NAME}: {e}.")
    # Fallback dimension if model loading fails. This should ideally match the intended model.
    EMBEDDING_DIMENSION = 384 
    print(f"Falling back to default embedding dimension: {EMBEDDING_DIMENSION}. This might be incorrect for the model {MODEL_NAME}.")

# Define default names for index and metadata files.
INDEX_FILE = "project_faiss_index.idx"
META_FILE = "project_faiss_metadata.pkl"

def main():
    """Demonstrates the core functionalities: indexing files and performing semantic search."""
    print("Advanced File System Navigator with Semantic Search - DEMO")
    
    if EMBEDDING_DIMENSION == 0:
        print("Could not determine embedding dimension. Exiting.")
        return

    indexer = FaissIndexer(dimension=EMBEDDING_DIMENSION, index_file_path=INDEX_FILE, metadata_file_path=META_FILE)

    # Define a list of directories to scan for content.
    scan_paths = ["test_data_1", "test_data_2", "test_data_3"]
    
    # --- Indexing Phase ---
    # Check if the index is already populated. If not, or if empty, proceed with indexing.
    if indexer.index is None or indexer.index.ntotal == 0:
        print("\\n--- Starting File Scan and Indexing ---")
        discoverer = FileDiscoverer() # Instantiate FileDiscoverer
        for scan_path in scan_paths:
            abs_scan_path = os.path.abspath(scan_path)
            if not os.path.exists(abs_scan_path):
                print(f"Scan path {abs_scan_path} does not exist. Skipping.")
                continue
            print(f"Scanning in '{abs_scan_path}'...")
            content_extractor = ContentExtractor() # Instantiate ContentExtractor
            for file_path in discoverer.scan_directory(abs_scan_path):
                print(f"  Processing: {file_path}")
                    
                content = content_extractor.extract_text(file_path) # Use instance method
                if content:
                    embedding = generate_embedding(content)
                    if embedding is not None:
                        indexer.add_embedding(embedding, file_path)
        indexer.save_index()
        print("\n--- Indexing Complete and Saved ---")
    else:
        print(f"\\n--- Index already populated with {indexer.index.ntotal} items. Skipping indexing phase. ---")

    # --- Search Phase ---
    print("\\n--- Semantic Search Demo ---")
    if indexer.index is None or indexer.index.ntotal == 0:
        print("Index is empty. Cannot perform search. Please run indexing first.")
        return

    # Example queries for demonstration.
    queries = [
        "python web development frameworks",
        "information about space telescopes",
        "machine learning classification algorithms",
        "how to handle hidden files in python script",
        "future of mars exploration"
    ]

    for user_query in queries:
        print(f"\nSearching for: '{user_query}'")
        query_embedding = generate_embedding(user_query)
        if query_embedding is not None:
            search_results = indexer.search(query_embedding, k=3)
            if search_results:
                print("  Results (Distance, File Path):")
                for distance, path in search_results:
                    print(f"    - {path} (Distance: {distance:.4f})")
            else:
                print("  No results found.")
        else:
            print("  Could not generate embedding for the query.")
            
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
