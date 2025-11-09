"""
BM25 index wrapper for convenient keyword-based search operations.
"""
import bm25s
from typing import Optional, Tuple, List, Union
import numpy as np


class BM25Index:
    """
    A convenient wrapper around BM25 index for keyword-based search operations.
    Uses the bm25s library for efficient BM25 scoring and retrieval.
    """

    def __init__(self):
        """Initialize an empty BM25 index."""
        self.retriever = None
        self.corpus = None
        self.corpus_tokens = None
        self.num_documents = 0
        self.stopwords = "en"

    def initialize(
        self,
        corpus: List[str],
        stopwords: Optional[Union[str, List[str]]] = "en",
        stemmer: Optional[str] = None
    ) -> None:
        """
        Initialize the BM25 index with the provided corpus.

        Args:
            corpus: A list of strings representing the documents to index.
            stopwords: Stopwords to use during tokenization. Can be:
                      - A string language code (e.g., "en" for English)
                      - A list of stopwords
                      - None to disable stopword removal
            stemmer: Optional stemmer to use. Currently supports basic stemming
                    if provided by the bm25s library.

        Raises:
            ValueError: If corpus is empty or not in the correct format.
        """
        # Validate input corpus
        if not isinstance(corpus, list):
            raise ValueError("Corpus must be a list of strings")

        if len(corpus) == 0:
            raise ValueError("Corpus cannot be empty")

        if not all(isinstance(doc, str) for doc in corpus):
            raise ValueError("All corpus elements must be strings")

        # Store corpus and metadata
        self.corpus = corpus
        self.num_documents = len(corpus)
        self.stopwords = stopwords

        # Tokenize the corpus
        # The tokenize function handles stopword removal and basic preprocessing
        self.corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)

        # Create and initialize the BM25 retriever
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)

        print(f"BM25 index initialized with {self.num_documents} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_documents: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the BM25 index for the top-k most relevant documents to the query.

        Args:
            query: A string representing the search query.
            top_k: Number of top documents to retrieve.
            return_documents: If True, returns document texts instead of just indices.

        Returns:
            A tuple of (results, scores):
                - results: Array of shape (1, top_k) with document texts (if return_documents=True)
                          or document indices (if return_documents=False)
                - scores: Array of shape (1, top_k) with BM25 relevance scores

        Raises:
            RuntimeError: If the index has not been initialized.
            ValueError: If query is not a string.
        """
        if self.retriever is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # Tokenize the query using the same preprocessing as the corpus
        query_tokens = bm25s.tokenize(query, stopwords=self.stopwords)

        # Perform the search
        # retrieve() returns (doc_ids, scores) or (docs, scores) depending on corpus parameter
        if return_documents and self.corpus is not None:
            results, scores = self.retriever.retrieve(
                query_tokens,
                k=top_k,
                corpus=self.corpus
            )
        else:
            results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        return results, scores

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        return_documents: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the BM25 index for multiple queries at once.

        Args:
            queries: A list of strings representing multiple search queries.
            top_k: Number of top documents to retrieve for each query.
            return_documents: If True, returns document texts instead of just indices.

        Returns:
            A tuple of (results, scores):
                - results: Array of shape (n_queries, top_k) with document texts or indices
                - scores: Array of shape (n_queries, top_k) with BM25 relevance scores

        Raises:
            RuntimeError: If the index has not been initialized.
            ValueError: If queries is not a list of strings.
        """
        if self.retriever is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        if not isinstance(queries, list):
            raise ValueError("Queries must be a list of strings")

        if not all(isinstance(q, str) for q in queries):
            raise ValueError("All queries must be strings")

        # Tokenize all queries
        query_tokens = bm25s.tokenize(queries, stopwords=self.stopwords)

        # Perform batch search
        if return_documents and self.corpus is not None:
            results, scores = self.retriever.retrieve(
                query_tokens,
                k=top_k,
                corpus=self.corpus
            )
        else:
            results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        return results, scores

    def save(
        self,
        index_path: str,
        save_corpus: bool = True
    ) -> None:
        """
        Save the BM25 index to disk.

        Args:
            index_path: Path where to save the index (directory will be created).
            save_corpus: Whether to save the corpus along with the index.

        Raises:
            RuntimeError: If the index has not been initialized.
        """
        if self.retriever is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        # Save the index with or without corpus
        if save_corpus and self.corpus is not None:
            self.retriever.save(index_path, corpus=self.corpus)
            print(f"BM25 index and corpus saved to {index_path}")
        else:
            self.retriever.save(index_path)
            print(f"BM25 index saved to {index_path}")

    def load(
        self,
        index_path: str,
        load_corpus: bool = True
    ) -> None:
        """
        Load a BM25 index from disk.

        Args:
            index_path: Path where the index is saved.
            load_corpus: Whether to load the corpus along with the index.

        Raises:
            FileNotFoundError: If the index path doesn't exist.
        """
        # Load the retriever from disk
        self.retriever = bm25s.BM25.load(index_path, load_corpus=load_corpus)

        # Update metadata after loading
        # Note: We need to estimate num_documents from the loaded index
        # The exact number depends on whether corpus was loaded
        if load_corpus:
            # If corpus was saved and loaded, we can access it
            # This assumes bm25s stores corpus in a retrievable way
            print(f"BM25 index loaded from {index_path} with corpus")
        else:
            print(f"BM25 index loaded from {index_path} without corpus")

        # Try to get the corpus from the loaded retriever if available
        if hasattr(self.retriever, 'corpus') and self.retriever.corpus is not None:
            self.corpus = self.retriever.corpus
            self.num_documents = len(self.corpus)
        else:
            self.corpus = None
            self.num_documents = 0

    def get_corpus_size(self) -> int:
        """
        Get the number of documents in the corpus.
        """
        return self.num_documents

    def get_corpus(self) -> Optional[List[str]]:
        """
        Get the corpus documents.

        Returns:
            The list of corpus documents, or None if corpus was not stored.
        """
        return self.corpus


#! ------- TESTING -------
def animal_bm25_example():
    # Create a sample corpus
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]

    # Initialize the BM25 index
    bm25_index = BM25Index()
    bm25_index.initialize(corpus, stopwords="en")

    print("\n--- Single Query Search ---")
    # Perform a single search
    query = "does the fish purr like a cat?"
    results, scores = bm25_index.search(query, top_k=2, return_documents=True)

    print(f"Query: {query}")
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")

    print("\n--- Batch Query Search ---")
    # Perform batch search
    queries = [
        "animals that can fly",
        "pets that live with humans",
    ]
    batch_results, batch_scores = bm25_index.batch_search(
        queries,
        top_k=2,
        return_documents=True
    )

    for query_idx, query in enumerate(queries):
        print(f"\nQuery {query_idx + 1}: {query}")
        for doc_idx in range(batch_results.shape[1]):
            doc = batch_results[query_idx, doc_idx]
            score = batch_scores[query_idx, doc_idx]
            print(f"  Rank {doc_idx + 1} (score: {score:.2f}): {doc}")

    print("\n--- Save and Load ---")
    # Save the index
    bm25_index.save("animal_index_bm25", save_corpus=True)

    # Load the index
    loaded_index = BM25Index()
    loaded_index.load("animal_index_bm25", load_corpus=True)

    # Test the loaded index
    test_query = "animals that swim"
    results, scores = loaded_index.search(test_query, top_k=2)
    print(f"\nQuery after loading: {test_query}")
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")

    print(f"\nCorpus size: {loaded_index.get_corpus_size()}")


if __name__ == "__main__":
    animal_bm25_example()