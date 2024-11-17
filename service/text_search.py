import os
import pickle
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
import re
from rapidfuzz import fuzz, process
from collections import defaultdict


class TextSearch:
    def __init__(self, index_path: str, top_k: int = 10, min_similarity: float = 80):
        """
        Initialize BM25 search with a pre-built index

        Args:
            index_path (str): Path to the pickled BM25 index file
            min_similarity (float): Minimum fuzzy match similarity score (0-100)
        """
        self.index_path = index_path
        self.top_k = top_k
        self.bm25 = None
        self.documents = None
        self.tokenized_docs = None
        self.stop_words = set()
        self.doc_ids = []
        self.min_similarity = min_similarity
        self.vocabulary = set()  # For fuzzy matching

        # Load the index
        self._load_index()
        # Build vocabulary for fuzzy matching
        self._build_vocabulary()

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

    def _build_vocabulary(self) -> None:
        """Build vocabulary from tokenized documents for fuzzy matching"""
        for doc in self.tokenized_docs:
            self.vocabulary.update(doc)
        logging.info(f"Built vocabulary with {len(self.vocabulary)} unique terms")

    def _get_fuzzy_matches(self, query_term: str, max_matches: int = 3) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches for a query term in the vocabulary

        Args:
            query_term (str): Term to find matches for
            max_matches (int): Maximum number of fuzzy matches to return

        Returns:
            List[Tuple[str, float]]: List of (term, similarity) pairs
        """
        matches = []

        for word in self.vocabulary:
            # Skip exact matches and very short terms
            if word == query_term or len(word) < 3:
                continue

            # Calculate similarity score
            similarity = fuzz.ratio(query_term.lower(), word.lower())
            if similarity >= self.min_similarity:
                matches.append((word, similarity))

        # Sort by similarity score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_matches]

    def _ensure_document_ids(self):
        """Ensure all documents have IDs, generate if missing"""
        for idx, doc in enumerate(self.documents):
            if isinstance(doc, str):
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

        self.doc_ids = [doc['id'] for doc in self.documents]

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function that splits on whitespace and punctuation

        Args:
            text (str): Input text to tokenize

        Returns:
            List[str]: List of tokens
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [token.strip() for token in text.split() if token.strip()]
        return tokens

    def preprocess_query(self, query: str, expand_query: bool = True) -> List[str]:
        """
        Preprocess and tokenize the search query, optionally expanding with fuzzy matches

        Args:
            query (str): Raw search query
            expand_query (bool): Whether to expand query with fuzzy matches

        Returns:
            List[str]: Tokenized and processed query
        """
        # Tokenize
        tokens = self._tokenize(query)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        if expand_query:
            expanded_tokens = []
            for token in tokens:
                # Add original token
                expanded_tokens.append(token)

                # Add fuzzy matches
                fuzzy_matches = self._get_fuzzy_matches(token)
                expanded_tokens.extend([match[0] for match in fuzzy_matches])

            return expanded_tokens

        return tokens

    def search(self,
               query: str,
               min_score: float = -30,
               use_fuzzy: bool = True,
               debug: bool = False) -> List[Dict[str, Union[str, float]]]:
        """
        Search the indexed documents using BM25 with optional fuzzy matching

        Args:
            query (str): Search query
            min_score (float): Minimum score threshold for results
            use_fuzzy (bool): Whether to use fuzzy matching
            debug (bool): Whether to include debug information in results

        Returns:
            List[Dict]: List of dictionaries containing document content and scores
        """
        try:
            # Preprocess query with optional fuzzy expansion
            tokenized_query = self.preprocess_query(query, expand_query=use_fuzzy)

            if not tokenized_query:
                logging.warning("Empty query after preprocessing")
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top k document indices
            top_indices = np.argsort(scores)[::-1][:self.top_k:]

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
                        'filename': doc.get('filename', ''),
                        'filepath': doc.get('filepath', ''),
                        'url': doc.get('url', '')
                    }

                    if debug:
                        result['debug'] = {
                            'original_query': query,
                            'expanded_query': tokenized_query,
                            'fuzzy_matches': {
                                token: self._get_fuzzy_matches(token)
                                for token in self._tokenize(query)
                            } if use_fuzzy else {}
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
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': float(np.mean([len(doc) for doc in self.tokenized_docs])),
            'has_metadata': all('metadata' in doc for doc in self.documents),
            'fuzzy_similarity_threshold': self.min_similarity
        }


    def get_word_suggestions(self, word: str, num_suggestions: int = 3, min_word_length: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Get suggested corrections for a potentially misspelled word

        Args:
            word (str): The word to get suggestions for
            num_suggestions (int): Number of suggestions to return
            min_word_length (int): Minimum word length to consider for suggestions

        Returns:
            List[Dict]: List of suggestions with similarity scores
        """
        # Don't suggest for short words or words already in vocabulary
        if len(word) < min_word_length or word in self.vocabulary:
            return []

        # Use rapidfuzz's process module for efficient matching
        suggestions = process.extract(
            word,
            self.vocabulary,
            scorer=fuzz.ratio,
            limit=num_suggestions
        )

        valid_suggestions = []
        for sugg in suggestions:
            match, score, _ = sugg

            # Consider a suggestion valid if:
            # 1. Very high similarity (likely just a small typo)
            # 2. Or reasonable similarity + significant difference in frequency
            if (score >= 90) or (score >= self.min_similarity and
                                 self._is_likely_misspelling(word, match)):
                valid_suggestions.append({
                    'suggested_word': match,
                    'similarity': score,
                    'original_word': word
                })

        return valid_suggestions

    def _is_likely_misspelling(self, original: str, suggestion: str) -> bool:
        """
        Determine if a word is likely a misspelling based on various heuristics

        Args:
            original (str): Original word
            suggestion (str): Suggested correction

        Returns:
            bool: True if the original is likely a misspelling
        """
        # If the original word exists in our vocab, it's probably not a misspelling
        if original in self.vocabulary:
            return False

        # Check for common typo patterns
        if self._has_common_typo_pattern(original, suggestion):
            return True

        # Get the frequency of both words in our corpus
        original_freq = sum(1 for doc in self.tokenized_docs if original in doc)
        suggestion_freq = sum(1 for doc in self.tokenized_docs if suggestion in doc)

        # If the suggested word is significantly more common,
        # it's more likely that the original is a misspelling
        if suggestion_freq > 0 and original_freq == 0:
            return True
        elif suggestion_freq > original_freq * 10:  # Suggested word is 10x more common
            return True

        return False

    def _has_common_typo_pattern(self, original: str, suggestion: str) -> bool:
        """
        Check if the difference between words matches common typo patterns

        Args:
            original (str): Original word
            suggestion (str): Suggested correction

        Returns:
            bool: True if the difference matches common typo patterns
        """
        if len(original) < 3:
            return False

        # Convert to lowercase for comparison
        original = original.lower()
        suggestion = suggestion.lower()

        # Common typo patterns
        patterns = [
            # Repeated letters (e.g., 'panddas' -> 'pandas')
            (r'(.)\1{2,}', r'\1\1'),

            # Missing letter
            (''.join(c for c in original if c in suggestion),
             ''.join(c for c in suggestion if c in original)),
        ]

        # Check for these patterns
        for pattern in patterns:
            if isinstance(pattern, tuple):
                if pattern[0] in original and pattern[1] in suggestion:
                    return True

        # Check for transposed letters (e.g., 'paندas' -> 'pandas')
        for i in range(len(original) - 1):
            if i < len(suggestion) - 1:
                # Get pairs of letters
                orig_pair = original[i:i + 2]
                sugg_pair = suggestion[i:i + 2]

                # Check if pairs are different but contain same letters
                if (orig_pair != sugg_pair and
                        sorted(orig_pair) == sorted(sugg_pair)):
                    return True

        # Check for common character substitutions
        substitutions = {
            '0': 'o', '1': 'l', '5': 's', '$': 's',
            '4': 'a', '3': 'e', '7': 't'
        }

        # Check each character for substitutions
        for i, char in enumerate(original):
            if i < len(suggestion):
                if char in substitutions and suggestion[i] == substitutions[char]:
                    return True
                elif suggestion[i] in substitutions and char == substitutions[suggestion[i]]:
                    return True

        return False

    def search_with_suggestions(self,
                                query: str,
                                min_score: float = -30) -> Dict[str, Union[List, str]]:
        """
        User-friendly search interface that provides the best suggestion and results

        Args:
            query (str): Search query
            min_score (float): Minimum score threshold for results

        Returns:
            Dict containing search results and best suggestion
        """
        # Clean the query of punctuation before processing
        clean_query = re.sub(r'[,?!.]', ' ', query).strip()
        results = self.suggest_and_search(clean_query, min_score)

        suggested_query = None
        suggestion_text = ""

        if results['has_suggestions']:
            query_tokens = self._tokenize(clean_query)
            replacements = {}

            for misspelled, suggs in results['suggestions'].items():
                if suggs and self._is_likely_misspelling(misspelled, suggs[0]['suggested_word']):
                    replacements[misspelled] = suggs[0]['suggested_word']

            if replacements:
                suggested_tokens = [replacements.get(token, token) for token in query_tokens]
                suggested_query = ' '.join(suggested_tokens)

                if suggested_query != clean_query:
                    suggestion_text = suggested_query
                    suggested_results = self.search(suggested_query,min_score=min_score)
                else:
                    suggested_query = None
                    suggested_results = []
            else:
                suggested_results = []
        else:
            suggested_results = []

        return {
            'suggestion_text': suggestion_text,
            'suggested_query': suggested_query,
            'results': suggested_results if suggested_query else results['search_results'],
            'original_query': query,
            'used_suggestion': bool(suggested_query)
        }

    def suggest_and_search(self,
                           query: str,
                           min_score: float = -30,
                           suggest_threshold: float = 80) -> Dict[str, Union[List, Dict]]:
        """
        Analyze query for potential misspellings, suggest corrections, and perform search

        Args:
            query (str): Search query
            min_score (float): Minimum score threshold for results
            suggest_threshold (float): Minimum similarity for suggestions

        Returns:
            Dict containing suggestions and search results
        """
        # Tokenize the query
        query_tokens = self._tokenize(query)

        # Get suggestions for each token
        suggestions = {}
        has_suggestions = False

        for token in query_tokens:
            if token not in self.vocabulary and len(token) > 2:  # Only suggest for tokens longer than 2 chars
                token_suggestions = self.get_word_suggestions(token)
                if token_suggestions:
                    suggestions[token] = token_suggestions
                    has_suggestions = True

        # Perform the search with fuzzy matching
        search_results = self.search(query, min_score=min_score)

        return {
            'original_query': query,
            'has_suggestions': has_suggestions,
            'suggestions': suggestions,
            'search_results': search_results
        }


if __name__ == "__main__":

    index_path = "/Users/shuwenwang/Documents/dev/uiuc/search-engine/index_data/d7be9b28-2285-4b15-a6b2-5443de534774.pkl"

    # Initialize the search
    searcher = TextSearch(index_path=index_path)

    # Perform a search
    results = searcher.search_with_suggestions("whta is padas, hw ues it?")
    print(results)

