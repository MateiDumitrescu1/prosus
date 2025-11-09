"""
FAISS index wrapper for convenient vector search operations.
"""
import numpy as np
from autofaiss import build_index
from typing import Optional, Tuple, List


class FaissIndex:
    """
    A convenient wrapper around FAISS index for vector search operations.
    Uses autofaiss to automatically select optimal index parameters.
    """

    def __init__(self):
        """Initialize an empty FAISS index."""
        self.index = None
        self.index_info = None
        self.embedding_dimension = None
        self.num_vectors = 0

    def initialize(
        self,
        embeddings: np.ndarray,
        save_on_disk: bool = False,
        index_path: Optional[str] = None,
        metric_type: str = "ip"  # "ip" for inner product (cosine), "l2" for euclidean
    ) -> None:
        """
        Initialize the FAISS index with the provided embeddings.

        Args:
            embeddings: A numpy array of shape (n_vectors, embedding_dim) containing
                       the document embeddings to index. Must be float32.
            save_on_disk: Whether to save the index to disk after building.
            index_path: Path where to save the index (if save_on_disk=True).
            metric_type: Distance metric to use - "ip" for inner product (cosine similarity)
                        or "l2" for Euclidean distance.

        Raises:
            ValueError: If embeddings are not in the correct format.
        """
        # Validate input embeddings
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")

        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            print(f"Converting embeddings to float32 from {embeddings.dtype}")

        # Store metadata
        self.num_vectors, self.embedding_dimension = embeddings.shape

        # Build the FAISS index using autofaiss
        # autofaiss automatically selects the best index type based on data size
        self.index, self.index_info = build_index(
            embeddings,
            save_on_disk=save_on_disk,
            index_path=index_path,
            metric_type=metric_type
        )

        print(f"FAISS index initialized with {self.num_vectors} vectors of dimension {self.embedding_dimension}")
        if self.index_info:
            print(f"Index type: {self.index_info.get('index_key', 'N/A')}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for the top-k most similar vectors to the query.

        Args:
            query_vector: A numpy array of shape (1, embedding_dim) or (embedding_dim,)
                         representing the query vector. Will be converted to float32 if needed.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            A tuple of (distances, indices):
                - distances: Array of shape (1, top_k) with similarity scores
                - indices: Array of shape (1, top_k) with the indices of nearest neighbors

        Raises:
            RuntimeError: If the index has not been initialized.
            ValueError: If query_vector has incorrect dimensions.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        # Handle both 1D and 2D query vectors
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Validate query vector dimensions
        if query_vector.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[1]} doesn't match "
                f"index dimension {self.embedding_dimension}"
            )

        # Ensure query is float32
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Perform the search
        # Returns distances and indices of the k nearest neighbors
        distances, indices = self.index.search(query_vector, top_k)

        return distances, indices

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for multiple queries at once.

        Args:
            query_vectors: A numpy array of shape (n_queries, embedding_dim)
                          representing multiple query vectors.
            top_k: Number of nearest neighbors to retrieve for each query.

        Returns:
            A tuple of (distances, indices):
                - distances: Array of shape (n_queries, top_k) with similarity scores
                - indices: Array of shape (n_queries, top_k) with the indices of nearest neighbors

        Raises:
            RuntimeError: If the index has not been initialized.
            ValueError: If query_vectors have incorrect dimensions.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        if len(query_vectors.shape) != 2:
            raise ValueError("Query vectors must be a 2D array")

        if query_vectors.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {query_vectors.shape[1]} doesn't match "
                f"index dimension {self.embedding_dimension}"
            )

        # Ensure queries are float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        # Perform batch search
        distances, indices = self.index.search(query_vectors, top_k)

        return distances, indices

    def get_index_size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            The number of vectors indexed.
        """
        return self.num_vectors

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings in the index.

        Returns:
            The embedding dimension.
        """
        return self.embedding_dimension


#! Example usage
if __name__ == "__main__":
    # Create sample embeddings (in practice, these would be your document embeddings)
    sample_embeddings = np.random.rand(100, 512).astype(np.float32)

    # Initialize the FAISS index
    faiss_index = FaissIndex()
    faiss_index.initialize(sample_embeddings, save_on_disk=False)

    # Create a sample query vector
    query = np.random.rand(512).astype(np.float32)

    # Search for top 5 similar vectors
    distances, indices = faiss_index.search(query, top_k=5)

    print(f"\nTop 5 similar document indices: {indices[0]}")
    print(f"Similarity scores: {distances[0]}")

    # Batch search example
    batch_queries = np.random.rand(3, 512).astype(np.float32)
    batch_distances, batch_indices = faiss_index.batch_search(batch_queries, top_k=3)

    print(f"\nBatch search results:")
    for i, (dists, idxs) in enumerate(zip(batch_distances, batch_indices)):
        print(f"Query {i}: indices={idxs}, scores={dists}")
