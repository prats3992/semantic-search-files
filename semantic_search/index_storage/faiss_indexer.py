import faiss
import numpy as np
import os
import pickle

class FaissIndexer:
    def __init__(self, dimension: int, index_file_path: str = "faiss_index.idx", metadata_file_path: str = "faiss_metadata.pkl"):
        """
        Initializes the FaissIndexer.

        Args:
            dimension: Dimensionality of the embeddings.
            index_file_path: Path to save/load the FAISS index file.
            metadata_file_path: Path to save/load metadata (mapping index IDs to file paths).
        """
        self.dimension = dimension
        self.index_file_path = index_file_path
        self.metadata_file_path = metadata_file_path
        
        self.index = None
        self.doc_id_to_filepath = {} # Maps internal FAISS ID to original file path
        self.filepath_to_doc_id = {} # Maps original file path to internal FAISS ID
        self.next_doc_id = 0

        self.gpu_res = None
        self.use_gpu = self._check_gpu_availability()

        self._load_index()

    def _check_gpu_availability(self) -> bool:
        """Checks for FAISS GPU availability and initializes resources if possible."""
        self.gpu_res = None
        try:
            if not hasattr(faiss, 'StandardGpuResources') or not hasattr(faiss, 'GpuIndexFlatL2'):
                print("FAISS Info: GPU extensions not found. Using CPU.")
                return False

            if faiss.get_num_gpus() > 0:
                print(f"FAISS Info: Found {faiss.get_num_gpus()} GPU(s). Attempting to use GPU.")
                try:
                    self.gpu_res = faiss.StandardGpuResources()
                    print("FAISS Info: GPU resources initialized successfully.")
                    return True
                except Exception as e_res:
                    print(f"FAISS Warning: Failed to initialize GPU resources: {e_res}. Using CPU.")
                    self.gpu_res = None
                    return False
            else:
                print("FAISS Info: No GPUs found. Using CPU.")
                return False
        except Exception as e_check:
            print(f"FAISS Warning: Error during FAISS GPU check: {e_check}. Using CPU.")
            return False

    def _initialize_index_if_needed(self):
        """Initializes a new FAISS index (CPU or GPU based on availability)."""
        if self.index is not None:
            return

        cpu_index = faiss.IndexFlatL2(self.dimension)
        if self.use_gpu and self.gpu_res:
            print(f"FAISS Info: Initializing new FAISS GPU index with dimension {self.dimension}")
            try:
                self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
                print("FAISS Info: Index successfully created on GPU.")
            except Exception as e_gpu_init:
                print(f"FAISS Warning: Failed to move initial index to GPU: {e_gpu_init}. Using CPU index.")
                self.index = cpu_index
                self.use_gpu = False
                self.gpu_res = None
        else:
            print(f"FAISS Info: Initializing new FAISS CPU index with dimension {self.dimension}")
            self.index = cpu_index
        
        self.doc_id_to_filepath = {}
        self.filepath_to_doc_id = {}
        self.next_doc_id = 0
        print("FAISS Info: Index initialized.")

    def add_embedding(self, embedding: np.ndarray, file_path: str):
        """
        Adds a single embedding and its associated file path to the index.

        Args:
            embedding: The embedding vector.
            file_path: The file path corresponding to the embedding.
        """
        self._initialize_index_if_needed()
        
        if embedding is None or embedding.ndim == 0:
            print(f"Skipping invalid or empty embedding for file: {file_path}")
            return

        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        if embedding.shape[1] != self.dimension:
            print(f"Error: Embedding dimension {embedding.shape[1]} does not match index dimension {self.dimension} for {file_path}. Skipping.")
            return

        try:
            self.index.add(embedding.astype(np.float32))
            current_faiss_id = self.index.ntotal - 1 # Get the ID assigned by FAISS
            self.doc_id_to_filepath[current_faiss_id] = file_path
            self.filepath_to_doc_id[file_path] = current_faiss_id
        except Exception as e:
            print(f"Error adding embedding for {file_path} to FAISS index: {e}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
        """
        Searches the index for the k most similar embeddings to the query_embedding.

        Args:
            query_embedding: The query embedding vector.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains (distance, file_path).
            Lower distance means higher similarity for L2 distance.
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is not initialized or is empty. Cannot search.")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        if query_embedding.shape[1] != self.dimension:
            print(f"Error: Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}. Cannot search.")
            return []

        try:
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            results = []
            for i in range(indices.shape[1]):
                faiss_id = indices[0][i]
                dist = distances[0][i]
                if faiss_id != -1:
                    if faiss_id in self.doc_id_to_filepath:
                        results.append((float(dist), self.doc_id_to_filepath[faiss_id]))
                    else:
                        print(f"Warning: FAISS ID {faiss_id} not found in metadata mapping.")
            return results
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []

    def get_embedding_by_filepath(self, file_path: str) -> np.ndarray | None:
        """
        Retrieves the embedding vector for a given file path.

        Args:
            file_path: The file path to retrieve the embedding for.

        Returns:
            The embedding vector if found, otherwise None.
        """
        if self.index is None:
            print("Index is not initialized. Cannot retrieve embedding.")
            return None
        
        doc_id = self.filepath_to_doc_id.get(file_path)
        if doc_id is None:
            # File path not found, this is a common case, no error print needed here.
            return None
        
        if doc_id >= self.index.ntotal:
            print(f"Error: Document ID {doc_id} for file path '{file_path}' is out of bounds for index size {self.index.ntotal}.")
            # Attempt to clean up inconsistent metadata
            del self.filepath_to_doc_id[file_path]
            # Also remove from reverse mapping if it exists
            keys_to_del = [k for k, v in self.doc_id_to_filepath.items() if v == file_path and k == doc_id]
            for key in keys_to_del:
                del self.doc_id_to_filepath[key]
            return None

        try:
            embedding = self.index.reconstruct(doc_id)
            return embedding
        except Exception as e:
            print(f"Error reconstructing embedding for file path '{file_path}' (ID: {doc_id}): {e}")
            return None

    def save_index(self):
        """Saves the FAISS index (as CPU index) and metadata to disk."""
        if self.index is None:
            print("Index not initialized. Nothing to save.")
            return

        index_to_save = self.index
        is_gpu_index_active = self.use_gpu and self.gpu_res is not None and self.index is not None

        if is_gpu_index_active:
            print("FAISS Info: Current index is on GPU. Converting to CPU for saving...")
            try:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            except Exception as e_gpu_to_cpu:
                print(f"FAISS Error: Error converting GPU index to CPU for saving: {e_gpu_to_cpu}. Index not saved.")
                return 
        
        try:
            print(f"Saving FAISS index to {self.index_file_path}...")
            faiss.write_index(index_to_save, self.index_file_path)
            print(f"Saving metadata to {self.metadata_file_path}...")
            with open(self.metadata_file_path, 'wb') as f:
                pickle.dump({
                    'doc_id_to_filepath': self.doc_id_to_filepath,
                    'filepath_to_doc_id': self.filepath_to_doc_id,
                    'next_doc_id': self.next_doc_id,
                    'dimension': self.dimension
                }, f)
            print("Index and metadata saved successfully.")
        except Exception as e_save:
            print(f"Error saving FAISS index or metadata: {e_save}")

    def _load_index(self):
        """Loads the FAISS index and metadata from disk, adapting to GPU if available."""
        if os.path.exists(self.index_file_path) and os.path.exists(self.metadata_file_path):
            try:
                print(f"Loading metadata from {self.metadata_file_path}...")
                with open(self.metadata_file_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                if metadata.get('dimension') != self.dimension:
                    print(f"Warning: Loaded index dimension ({metadata.get('dimension')}) differs from configured dimension ({self.dimension}). Re-initializing index.")
                    self._initialize_index_if_needed() # This will reset filepath_to_doc_id as well
                    return

                print(f"Loading FAISS CPU index from {self.index_file_path}...")
                loaded_cpu_index = faiss.read_index(self.index_file_path)
                
                self.doc_id_to_filepath = metadata['doc_id_to_filepath']
                self.next_doc_id = metadata['next_doc_id']
                # Load filepath_to_doc_id, or initialize if not present for backward compatibility
                self.filepath_to_doc_id = metadata.get('filepath_to_doc_id', {})

                # Integrity check / rebuild filepath_to_doc_id if it seems empty or inconsistent.
                # This is a simple check; more sophisticated checks could be added.
                if not self.filepath_to_doc_id and self.doc_id_to_filepath:
                    print("Warning: filepath_to_doc_id mapping not found or empty in loaded metadata. Rebuilding from doc_id_to_filepath.")
                    self.filepath_to_doc_id = {v: k for k, v in self.doc_id_to_filepath.items()}
                elif len(self.filepath_to_doc_id) != len(self.doc_id_to_filepath):
                     print(f"Warning: Mismatch in lengths of filepath_to_doc_id ({len(self.filepath_to_doc_id)}) and doc_id_to_filepath ({len(self.doc_id_to_filepath)}). Rebuilding filepath_to_doc_id.")
                     self.filepath_to_doc_id = {v: k for k, v in self.doc_id_to_filepath.items()}


                if self.use_gpu and self.gpu_res:
                    print("FAISS Info: Attempting to move loaded CPU index to GPU...")
                    try:
                        self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, loaded_cpu_index)
                        print(f"FAISS Info: Index ({self.index.ntotal} vectors) successfully moved to GPU.")
                    except Exception as e_gpu_move:
                        print(f"FAISS Warning: Failed to move loaded index to GPU: {e_gpu_move}. Using CPU index.")
                        self.index = loaded_cpu_index 
                        self.use_gpu = False 
                        self.gpu_res = None
                else:
                    print(f"Using loaded CPU index ({loaded_cpu_index.ntotal} vectors).")
                    self.index = loaded_cpu_index
                
                print("FAISS index and metadata loaded successfully.")
            except Exception as e_load:
                print(f"Error loading FAISS index or metadata: {e_load}. Re-initializing index.")
                self._initialize_index_if_needed()
        else:
            print("No existing FAISS index found. Will create a new one.")
            self._initialize_index_if_needed()
