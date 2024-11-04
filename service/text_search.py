import os
import pickle
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
import re


class TextSearch:
    def __init__(self, index_path: str):
        """
        Initialize BM25 search with a pre-built index

        Args:
            index_path (str): Path to the pickled BM25 index file
        """
        self.index_path = index_path
        self.bm25 = None
        self.documents = None
        self.tokenized_docs = None
        self.stop_words = set()
        self.doc_ids = []  # Add document IDs list

        # Load the index
        self._load_index()

    def _load_index(self) -> None:
        """Load the pre-built BM25 index and associated data"""
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
                self.bm25 = index_data.get('bm25')
                self.documents = index_data.get('documents')
                self.tokenized_docs = index_data.get('tokenized_docs')
                self.stop_words = index_data.get('stop_words', set())

            if not all([self.bm25, self.documents, self.tokenized_docs]):
                raise ValueError("Index file is missing required components")

            # Generate document IDs if they don't exist
            self._ensure_document_ids()

        except Exception as e:
            logging.error(f"Error loading index from {self.index_path}: {str(e)}")
            raise

    def _ensure_document_ids(self):
        """Ensure all documents have IDs, generate if missing"""
        for idx, doc in enumerate(self.documents):
            if isinstance(doc, str):
                # If document is just a string, convert to dict with content and ID
                self.documents[idx] = {
                    'content': doc,
                    'id': f'doc_{idx}',
                    'metadata': {}
                }
            elif isinstance(doc, dict):
                if 'id' not in doc:
                    doc['id'] = f'doc_{idx}'
                if 'content' not in doc:
                    logging.warning(f"Document at index {idx} has no content field")
                    doc['content'] = ''
                if 'metadata' not in doc:
                    doc['metadata'] = {}
            else:
                raise ValueError(f"Invalid document format at index {idx}")

        # Update doc_ids list
        self.doc_ids = [doc['id'] for doc in self.documents]

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function that splits on whitespace and punctuation

        Args:
            text (str): Input text to tokenize

        Returns:
            List[str]: List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split on whitespace and filter empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]

        return tokens

    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess and tokenize the search query

        Args:
            query (str): Raw search query

        Returns:
            List[str]: Tokenized and processed query
        """
        # Tokenize
        tokens = self._tokenize(query)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        return tokens

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Search the indexed documents using BM25

        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            min_score (float): Minimum score threshold for results

        Returns:
            List[Dict]: List of dictionaries containing document content and scores
        """
        try:
            # Preprocess query
            tokenized_query = self.preprocess_query(query)

            if not tokenized_query:
                logging.warning("Empty query after preprocessing")
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top k document indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Prepare results
            results = []
            for idx in top_indices:
                if scores[idx] > min_score:
                    doc = self.documents[idx]
                    result = {
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(scores[idx]),
                        'document_id': doc['id'],
                        'position': idx,
                        'filename': doc.get('filename', ''),  # Add filename to results
                        'filepath': doc.get('filepath', '')
                    }
                    results.append(result)

            return results

        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            raise

    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        Retrieve a specific document by its ID

        Args:
            doc_id (str): Document ID to retrieve

        Returns:
            Dict: Document content and metadata
        """
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the loaded index

        Returns:
            Dict: Index statistics
        """
        return {
            'num_documents': len(self.documents),
            'index_path': self.index_path,
            'vocabulary_size': len(self.bm25.idf),
            'average_document_length': float(np.mean([len(doc) for doc in self.tokenized_docs])),
            'has_metadata': all('metadata' in doc for doc in self.documents)
        }

if __name__ == "__main__":

    index_path = "/Users/shuwenwang/Documents/dev/uiuc/search-engine/index_data/9a66bea3-4dff-4aa3-9dd3-f93a2a55665e.pkl"

    # Initialize the search
    searcher = TextSearch(index_path=index_path)

    # Perform a search
    results = searcher.search("what is storm?", top_k=5)

    # Print results
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Content: {result['content'][:200]}...")  # First 200 chars
        print(f"Document ID: {result['document_id']}")
        print("---")

    # Get index statistics
    stats = searcher.get_index_stats()
    print(f"Total documents: {stats['num_documents']}")

