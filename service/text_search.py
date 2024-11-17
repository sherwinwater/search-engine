import os
import pickle
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
import re
from rapidfuzz import process, fuzz
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
        self.stop_words = set(stopwords.words('english'))
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
        """Tokenize text into words"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return list(filter(None, map(str.strip, text.split())))

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess and tokenize the search query"""
        tokens = self._tokenize(query)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = self.correct_spelling(tokens)
        return tokens

    def correct_spelling(self, tokens: List[str]) -> List[str]:
        """Correct spelling in the query using RapidFuzz for fuzzy matching"""
        corrected_tokens = []
        for token in tokens:
            best_match = process.extractOne(token, self.bm25.idf.keys(), scorer=fuzz.ratio)
            if best_match and best_match[1] > 80:
                corrected_tokens.append(best_match[0])
            else:
                corrected_tokens.append(token)
        return corrected_tokens

    def fuzzy_match_content(self, query: List[str], document: str) -> float:
        """Calculate fuzzy match score between query tokens and document content"""
        scores = []
        for token in query:
            match = process.extractOne(token, document.split(), scorer=fuzz.ratio)
            if match:
                scores.append(match[1])
        return np.mean(scores) if scores else 0.0

    def combine_scores(self, bm25_score: float, semantic_score: float, fuzzy_score: float) -> float:
        """Combine BM25, semantic, and fuzzy scores into a single score"""
        weights = {"bm25": 0.3, "semantic": 0.5, "fuzzy": 0.2}
        return (
            weights["bm25"] * bm25_score +
            weights["semantic"] * semantic_score +
            weights["fuzzy"] * fuzzy_score
        )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """Search the indexed documents using BM25, fuzzy match, and semantic ranking"""
        try:
            tokenized_query = self.preprocess_query(query)
            if not tokenized_query:
                logging.warning("Empty query after preprocessing")
                return []

            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > min_score:
                    doc = self.documents[idx]
                    content = doc['content']
                    fuzzy_score = self.fuzzy_match_content(tokenized_query, content)
                    result = {
                        'content': content,
                        'metadata': doc['metadata'],
                        'bm25_score': float(scores[idx]),
                        'fuzzy_score': fuzzy_score/100,
                        'document_id': doc['id']
                    }
                    results.append(result)

            if self.embedder:
                results = self.rerank_with_semantics(query, results)

            for result in results:
                result['score'] = self.combine_scores(
                    bm25_score=result['bm25_score'],
                    semantic_score=result.get('semantic_score', 0.0),
                    fuzzy_score=result['fuzzy_score']
                )

            return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            raise

    def rerank_with_semantics(self, query: str, results: List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, float]]]:
        """Rerank search results using Sentence-BERT embeddings"""
        query_embedding = self.embedder.encode([query])
        doc_embeddings = self.embedder.encode([result['content'] for result in results])
        cosine_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

        for i, result in enumerate(results):
            result['semantic_score'] = float(cosine_similarities[i])

        return sorted(results, key=lambda x: x['semantic_score'], reverse=True)

    def get_index_stats(self) -> Dict:
        """Get statistics about the loaded index"""
        return {
            'num_documents': len(self.documents),
            'index_path': self.index_path,
            'vocabulary_size': len(self.bm25.idf),
            'average_document_length': float(np.mean([len(doc) for doc in self.tokenized_docs])),
            'has_metadata': all('metadata' in doc for doc in self.documents)
        }

if __name__ == "__main__":
    searcher = TextSearch(index_path='../index_data/5ca59887-5717-4cab-bc0e-0f98bfb5964e.pkl')
    query = 'physvics H?andvles the pvhysic sivmulation'
    # query = 'physics Handles the physics simulation'
    # query = 'the the the a so'
    results = searcher.search(query, top_k=5)
    # print(results[0].keys())
    # dict_keys(['content', 'metadata', 'bm25_score', 'fuzzy_score', 'document_id', 'semantic_score', 'score'])

    for result in results:
        print(f"Score: combined {result['score']:.3f}, semantic {result['semantic_score']:.3f}, bm25 {result['bm25_score']:.3f}, fuzzy {result['fuzzy_score']:.3f}")
        print(f"Content: {result['content'][:200]}...")  # First 200 chars
        print(f"Document ID: {result['document_id']}")
        print("---")

    # # Get index statistics
    # stats = searcher.get_index_stats()
    # print(f"Total documents: {stats['num_documents']}")
