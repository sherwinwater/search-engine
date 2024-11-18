import logging
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import PyPDF2
import docx
import pickle
from collections import defaultdict, Counter, OrderedDict
from bs4 import BeautifulSoup
import re
from gensim.models import Word2Vec

from db.database import SearchEngineDatabase
from service.text_summarizer import TextSummarizer
from utils.setup_logging import setup_logging

logger = logging.getLogger(__name__)


class DocumentClustering:
    def __init__(self, task_id, folder_path, url_mapping_file=None):
        self.task_id = task_id
        self.folder_path = Path(folder_path)
        self.clustering_data_path = os.path.join(self.folder_path, "clustering_data")
        self.documents = []
        self.file_paths = []
        self.urls = []  # New list to store URLs
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cluster_labels = None
        self.model = None
        self.word2vec_model = None
        self.document_vectors = None

        self.summarizer = TextSummarizer()

        
        self.logger = setup_logging(name=f"{__name__}", task_id=self.task_id)

        self.url_mapping = {}  # Dictionary to store path -> URL mapping
        self.stop_words_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "stopwords.txt"))
        self.load_stopwords(self.stop_words_path)

        # Load URL mapping if provided
        if url_mapping_file:
            self.load_url_mapping(url_mapping_file)

    def load_url_mapping(self, mapping_file):
        """Load URL mapping from JSON file"""
        try:
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                # Create mapping from file path to URL
                for node in data.get('nodes', []):
                    if 'path' in node and 'url' in node:
                        self.url_mapping[node['path']] = node['url']
        except Exception as e:
            self.logger.exception(f"Error loading URL mapping: {e}")

    def load_stopwords(self, file_path):
        with open(file_path, 'r') as f:
            self.stop_words = set(word.strip().lower() for word in f)

    def generate_summary(self, text, num_sentences=2, max_chars=200):
        """
        Generate a summary of the given text with robust error handling and text processing.

        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences to include in summary
            max_chars (int): Maximum characters in the summary

        Returns:
            str: Generated summary or appropriate fallback text
        """
        try:
            # Input validation
            if not isinstance(text, str):
                return "Invalid input: Text must be a string."

            # Clean and normalize the text
            cleaned_text = ' '.join(text.split())  # Remove extra whitespace
            if not cleaned_text:
                return "No content available for summarization."

            # If text is shorter than max_chars, return it directly
            if len(cleaned_text) <= max_chars:
                return cleaned_text

            # Ensure reasonable number of sentences
            num_sentences = max(1, min(num_sentences, 5))  # Cap between 1 and 5 sentences

            try:
                # Attempt to generate summary using TextSummarizer
                summary = self.summarizer.summarize(cleaned_text, num_sentences)

                # If summarizer returns empty or None, fall back to text truncation
                if not summary:
                    raise ValueError("Summarizer returned empty result")

            except Exception as summarizer_error:
                self.logger.warning(f"Summarizer failed: {summarizer_error}. Falling back to truncation.")
                # Fallback: Take first few sentences
                sentences = cleaned_text.split('.')
                summary = '. '.join(sentences[:num_sentences]) + '.'

            # Ensure the summary doesn't exceed max_chars
            if len(summary) > max_chars:
                # Try to truncate at a sentence boundary first
                truncated = summary[:max_chars]
                last_period = truncated.rfind('.')

                if last_period > max_chars * 0.5:  # If we can find a good sentence break
                    summary = truncated[:last_period + 1]
                else:
                    # Fall back to word boundary
                    last_space = truncated.rfind(' ')
                    summary = truncated[:last_space] + '...'

            # Final cleanup
            summary = summary.strip()
            if not summary.endswith(('.', '!', '?', '...')):
                summary += '...'

            return summary

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."
    def clean_html_text(self, text):
        """Clean extracted HTML text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def read_html_file(self, file_path):
        """Read HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text from specific content areas if they exist
                content = []

                # Try to get content from common content areas
                content_areas = soup.find_all(['article', 'main', 'div', 'section'])
                if content_areas:
                    for area in content_areas:
                        if area.get('class'):
                            class_name = ' '.join(area.get('class')).lower()
                            # Look for divs likely to contain main content
                            if any(term in class_name for term in ['content', 'article', 'main', 'body', 'text']):
                                content.append(area.get_text())

                # If no specific content areas found, get all text
                if not content:
                    content = [soup.get_text()]

                # Clean and join the text
                text = ' '.join(content)
                return self.clean_html_text(text)

        except Exception as e:
            self.logger.info(f"Error reading HTML {file_path}: {e}")
            return ""

    def read_text_file(self, file_path):
        """Read plain text files"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def read_pdf_file(self, file_path):
        """Read PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + ""
        except Exception as e:
            self.logger.info(f"Error reading PDF {file_path}: {e}")
        return text

    def read_docx_file(self, file_path):
        """Read Word documents"""
        try:
            doc = docx.Document(file_path)
            text = "".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            self.logger.info(f"Error reading DOCX {file_path}: {e}")
            return ""

    def load_documents(self):
        """Load all supported documents from the folder"""
        supported_extensions = {'.txt', '.pdf', '.docx', '.html', '.htm'}

        for file_path in self.folder_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    if file_path.suffix.lower() == '.txt':
                        text = self.read_text_file(file_path)
                    elif file_path.suffix.lower() == '.pdf':
                        text = self.read_pdf_file(file_path)
                    elif file_path.suffix.lower() == '.docx':
                        text = self.read_docx_file(file_path)
                    elif file_path.suffix.lower() in ['.html', '.htm']:
                        text = self.read_html_file(file_path)

                    if text.strip():  # Only add non-empty documents
                        self.documents.append(text)
                        file_path_str = str(file_path)
                        self.file_paths.append(file_path_str)
                        # Add URL if available in mapping
                        self.urls.append(self.url_mapping.get(file_path_str, ''))

                except Exception as e:
                    self.logger.info(f"Error processing {file_path}: {e}")

        self.logger.info(f"Loaded {len(self.documents)} documents")

    def analyze_document_metadata(self):
        """Analyze and return metadata about the document collection"""
        metadata = {
            'total_documents': len(self.documents),
            'file_types': defaultdict(int),
            'avg_document_length': 0,
            'empty_files': []
        }

        total_length = 0
        for file_path, doc in zip(self.file_paths, self.documents):
            ext = Path(file_path).suffix.lower()
            metadata['file_types'][ext] += 1

            doc_length = len(doc)
            total_length += doc_length

            if doc_length == 0:
                metadata['empty_files'].append(file_path)

        if metadata['total_documents'] > 0:
            metadata['avg_document_length'] = total_length / metadata['total_documents']

        return metadata

    @staticmethod
    def custom_preprocessor(text):
        """Preprocess text by removing unwanted patterns"""
        # Remove version numbers and timestamps
        text = re.sub(r'\d+\.\d+\.\d+', ' ', text)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text)  # dates
        text = re.sub(r'\b\d+\s*(?:days?|months?|years?)\b', ' ', text)  # time periods

        # Handle special characters
        text = re.sub(r'[_-]', ' ', text)  # Replace underscores and hyphens with spaces
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove other special characters

        # Normalize whitespace
        text = ' '.join(text.split())
        return text.lower()

    @staticmethod
    def custom_tokenizer(text):
        """Tokenize text with special handling for technical terms"""
        words = []
        for word in text.split():
            # Split camelCase
            camel_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', word)
            if camel_words:
                words.extend(w.lower() for w in camel_words)
            else:
                words.append(word.lower())
        return [w for w in words if DocumentClustering.token_filter(w)]

    @staticmethod
    def token_filter(token):
        """Filter tokens based on specific criteria"""
        return (
                len(token) >= 3 and  # Minimum length
                not token.isdigit() and  # No pure numbers
                not re.match(r'^v\d+$', token) and  # No version numbers like v1, v2
                not any(char.isdigit() for char in token)  # No mixed alphanumeric
        )

    @staticmethod
    def simple_stem(word):
        """Simple stemming function"""
        word = word.lower()
        if word.endswith('ing'): return word[:-3]
        if word.endswith('ed'): return word[:-2]
        if word.endswith('s'): return word[:-1]
        if word.endswith('ies'): return word[:-3] + 'y'
        return word

    @staticmethod
    def preprocess_text(text):
        """Preprocess text for word2vec"""
        # Remove version numbers and technical patterns
        text = re.sub(r'\d+\.\d+\.\d+', ' ', text)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text)
        text = re.sub(r'\b\d+\s*(?:days?|months?|years?)\b', ' ', text)
        text = re.sub(r'[_-]', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Convert to lowercase and split
        return text.lower().split()

    def train_word2vec(self):
        """Train Word2Vec model on the documents"""
        # Prepare sentences (tokenized documents)
        sentences = [
            self.preprocess_text(doc) for doc in self.documents
        ]

        # Filter out stop words
        sentences = [
            [word for word in sentence if word not in self.stop_words]
            for sentence in sentences
        ]

        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=100,  # Dimension of word vectors
            window=5,  # Context window size
            min_count=2,  # Ignore words that appear less than this
            sg=1,  # Use skip-gram model
            workers=4  # Number of parallel processes
        )

        # Create document vectors by averaging word vectors
        self.document_vectors = np.zeros((len(sentences), self.word2vec_model.vector_size))

        for idx, sentence in enumerate(sentences):
            valid_words = [
                word for word in sentence
                if word in self.word2vec_model.wv
            ]
            if valid_words:
                self.document_vectors[idx] = np.mean(
                    [self.word2vec_model.wv[word] for word in valid_words],
                    axis=0
                )

    def preprocess_and_vectorize(self, max_features=5000):
        """Convert documents to vectors using Word2Vec"""
        # Train Word2Vec model instead of using TF-IDF
        self.train_word2vec()

        # Use document vectors for clustering
        self.tfidf_matrix = self.document_vectors

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        if len(self.documents) == 1:
            return 1  # For a single document, return 1 cluster

        best_score = -1
        best_n = 2

        for n in range(2, min(max_clusters + 1, len(self.documents))):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(self.tfidf_matrix)
            score = silhouette_score(self.tfidf_matrix, labels)

            if score > best_score:
                best_score = score
                best_n = n

        return best_n

    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()

        n_clusters = max(1, min(n_clusters, len(self.documents)))

        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.model.fit_predict(self.tfidf_matrix)

    def get_cluster_keywords(self, cluster_id, top_n=5):
        """Get top keywords for a cluster with optimized performance"""
        if not hasattr(self, '_cluster_keyword_cache'):
            self._cluster_keyword_cache = {}

        # Check cache first
        cache_key = f"{cluster_id}_{top_n}"
        if cache_key in self._cluster_keyword_cache:
            return self._cluster_keyword_cache[cache_key]

        try:
            # Get documents in this cluster
            cluster_docs = np.where(self.cluster_labels == cluster_id)[0]
            if len(cluster_docs) == 0:
                return []

            # Collect all terms and their frequencies
            word_freq = Counter()
            path_terms = set()
            title_terms = set()

            # Process all documents in cluster at once
            for doc_idx in cluster_docs:
                # Extract path terms
                path_parts = Path(self.file_paths[doc_idx]).parts
                for part in path_parts:
                    terms = re.findall(r'[a-zA-Z]+', part.lower())
                    path_terms.update([t for t in terms if len(t) > 2 and t not in self.stop_words])

                # Extract terms from document
                doc_text = self.documents[doc_idx]
                words = re.findall(r'\b[a-zA-Z]+\b', doc_text.lower())
                word_freq.update([w for w in words if len(w) > 2 and w not in self.stop_words])

            # Get terms from other clusters for comparison (sample-based)
            other_docs = np.where(self.cluster_labels != cluster_id)[0]
            other_freq = Counter()
            sample_size = min(len(other_docs), 1000)  # Limit sample size
            if sample_size > 0:
                sampled_docs = np.random.choice(other_docs, sample_size, replace=False)
                for doc_idx in sampled_docs:
                    words = re.findall(r'\b[a-zA-Z]+\b', self.documents[doc_idx].lower())
                    other_freq.update([w for w in words if len(w) > 2 and w not in self.stop_words])

            # Calculate term importance scores
            word_scores = {}
            total_words = sum(word_freq.values())
            total_other_words = sum(other_freq.values())

            # Pre-calculate word vectors for frequently used words
            word_vectors = {}
            for word, count in word_freq.most_common(100):  # Limit to top 100 words
                if word in self.word2vec_model.wv:
                    word_vectors[word] = self.word2vec_model.wv[word]

            # Score calculation
            for word, count in word_freq.most_common(50):  # Focus on top 50 words
                if word in self.word2vec_model.wv:
                    # TF-IDF like score
                    tf = count / total_words
                    other_tf = other_freq[word] / (total_other_words or 1)
                    distinctiveness = max(0, tf - other_tf)

                    # Path term bonus
                    path_score = 2.0 if word in path_terms else 0.0

                    # Semantic coherence score (using pre-calculated vectors)
                    word_vector = word_vectors.get(word, self.word2vec_model.wv[word])
                    coherence_scores = []
                    for other_word, other_vector in word_vectors.items():
                        if other_word != word:
                            similarity = np.dot(word_vector, other_vector) / (
                                    np.linalg.norm(word_vector) * np.linalg.norm(other_vector)
                            )
                            coherence_scores.append(similarity)

                    coherence_score = np.mean(coherence_scores) if coherence_scores else 0

                    # Combined score
                    word_scores[word] = (
                            0.4 * path_score +
                            0.3 * distinctiveness +
                            0.3 * coherence_score
                    )

            # Select diverse keywords
            keywords = []
            seen_vectors = []

            for word, _ in sorted(word_scores.items(), key=lambda x: x[1], reverse=True):
                if len(keywords) >= top_n:
                    break

                word_vector = self.word2vec_model.wv[word]

                # Check similarity with existing keywords
                is_diverse = True
                for seen_vector in seen_vectors:
                    similarity = np.dot(word_vector, seen_vector) / (
                            np.linalg.norm(word_vector) * np.linalg.norm(seen_vector)
                    )
                    if similarity > 0.7:  # Similarity threshold
                        is_diverse = False
                        break

                if is_diverse:
                    keywords.append(word)
                    seen_vectors.append(word_vector)

            # Cache the results
            self._cluster_keyword_cache[cache_key] = keywords
            return keywords

        except Exception as e:
            self.logger.info(f"Error in get_cluster_keywords: {e}")
            return [f"cluster_{cluster_id}"]

    def get_descriptive_cluster_name(self, cluster_id, max_keywords=2):
        """Generate cluster names with only words, no numbers"""
        if not hasattr(self, '_cluster_name_cache'):
            self._cluster_name_cache = {}

        # Check cache
        if cluster_id in self._cluster_name_cache:
            return self._cluster_name_cache[cluster_id]

        try:
            keywords = self.get_cluster_keywords(cluster_id, top_n=max_keywords)

            if not keywords:
                name = f"Cluster_{cluster_id}"
            else:
                # Create name from keywords, remove any numbers
                cleaned_keywords = []
                for k in keywords[:max_keywords]:
                    # Remove numbers and clean the keyword
                    cleaned = re.sub(r'_?\d+', '', k.title())  # Remove numbers
                    cleaned = re.sub(r'_+', '_', cleaned)  # Clean up multiple underscores
                    cleaned = cleaned.strip('_')  # Remove leading/trailing underscores
                    if cleaned:
                        cleaned_keywords.append(cleaned)

                # Join keywords with underscore, no size added
                name = '_'.join(cleaned_keywords)

            # Cache the result
            self._cluster_name_cache[cluster_id] = name
            return name

        except Exception as e:
            self.logger.info(f"Error in get_descriptive_cluster_name: {e}")
            return f"Cluster_{cluster_id}"

    def create_browsable_structure(self):
        """Create hierarchical structure with optimized performance"""
        self.logger.info("Starting create_browsable_structure")

        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")

        self.logger.info(f"Processing {len(self.documents)} documents in {len(set(self.cluster_labels))} clusters")

        # Pre-calculate cluster names for all unique cluster IDs
        self.logger.info("Pre-calculating cluster names...")
        unique_clusters = set(self.cluster_labels)
        cluster_names = {
            cluster_id: self.get_descriptive_cluster_name(cluster_id)
            for cluster_id in unique_clusters
        }
        self.logger.info("Finished pre-calculating cluster names")

        structure = defaultdict(dict)

        # Process documents in batches
        batch_size = 100
        total_batches = (len(self.documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.documents))
            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

            for idx in range(start_idx, end_idx):
                label = self.cluster_labels[idx]
                cluster_name = cluster_names[label]

                # Initialize cluster structure if needed
                if "documents" not in structure[cluster_name]:
                    structure[cluster_name] = {
                        "documents": [],
                        "keywords": self.get_cluster_keywords(label),
                        "size": 0,
                        "cluster_id": int(label)
                    }

                # Get document file type and size
                file_path = self.file_paths[idx]
                file_type = Path(file_path).suffix.lower()

                try:
                    file_size = os.path.getsize(file_path)
                    file_size_mb = round(file_size / (1024 * 1024), 2)  # Convert to MB
                except Exception as e:
                    self.logger.warning(f"Error getting file size for {file_path}: {e}")
                    file_size_mb = 0

                # Extract text and adjust summary parameters based on file type and size
                text = self.documents[idx]

                # Adjust summary parameters based on file type and size
                if file_type in ['.pdf', '.docx']:
                    # For larger document types, use more sentences but keep it concise
                    summary = self.generate_summary(text, num_sentences=3, max_chars=300)
                elif file_type in ['.txt', '.html', '.htm']:
                    # For web and text content, keep it shorter
                    summary = self.generate_summary(text, num_sentences=2, max_chars=200)
                else:
                    # Default case
                    summary = self.generate_summary(text, num_sentences=2, max_chars=150)

                # Create document info with additional metadata
                doc_info = {
                    "file_path": str(file_path),
                    "url": str(self.urls[idx]) if idx < len(self.urls) else "",
                    "file_type": file_type,
                    "file_size_mb": file_size_mb,
                    "index": idx,
                    "preview": summary,
                    "metadata": {
                        "characters": len(text),
                        "estimated_read_time": f"{max(1, len(text.split()) // 200)} min",  # Assuming 200 words per minute
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    }
                }

                structure[cluster_name]["documents"].append(doc_info)
                structure[cluster_name]["size"] += 1

            self.logger.info(f"Processed {end_idx} documents")

        self.logger.info("Finished creating browsable structure")
        return dict(structure)

    def save_results(self):
        """Save clustering results and model with database integration"""
        try:
            self.logger.info("self.task_id Starting save_results")

            # Get database connection
            db = SearchEngineDatabase()

            try:
                # 1. Validate clustering results
                if self.cluster_labels is None or self.model is None:
                    raise ValueError("Clustering has not been performed yet")

                # 2. Create directory
                if not os.path.exists(self.clustering_data_path):
                    os.makedirs(self.clustering_data_path)

                # 3. Create browsable structure
                structure = self.create_browsable_structure()

                # 4. Save cluster structure
                cluster_structure_path = os.path.join(self.clustering_data_path, "cluster_structure.json")
                serializable_structure = self._make_json_serializable(structure)
                with open(cluster_structure_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_structure, f, indent=2, ensure_ascii=False)

                # 5. Save metadata
                metadata = self.analyze_document_metadata()
                metadata_path = os.path.join(self.clustering_data_path, "document_metadata.json")
                serializable_metadata = self._make_json_serializable(metadata)
                with open(metadata_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)

                # 6. Save model
                model_path = os.path.join(self.clustering_data_path, "clustering_model.pkl")
                model_data = {
                    'vectorizer': self.vectorizer,
                    'model': self.model
                }
                with open(model_path, "wb") as f:
                    pickle.dump(model_data, f, protocol=4)

                # 7. Save summary
                summary = {
                    'total_documents': len(self.documents),
                    'num_clusters': len(set(self.cluster_labels)),
                    'documents_per_cluster': self._get_cluster_distribution(),
                    'timestamp': datetime.now().isoformat()
                }
                summary_path = os.path.join(self.clustering_data_path, "clustering_summary.json")
                with open(summary_path, "w", encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)

                # 8. Save to database
                # Save file paths
                db.save_clustering_paths(
                    self.task_id,
                    cluster_structure_path,
                    metadata_path,
                    model_path,
                    summary_path,
                    len(set(self.cluster_labels))
                )

                # Save cluster details
                db.save_cluster_details(self.task_id, structure)

                # Update status to completed
                db.update_clustering_status(self.task_id, 'completed')

                self.logger.info("Save results completed successfully ")

            except Exception as e:
                if 'task_id' in locals():
                    db.update_clustering_status(self.task_id, 'failed', str(e))
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.info(f"CRITICAL ERROR in save_results: {str(e)}")
            raise

    def _make_json_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj

    def _get_cluster_distribution(self):
        """Safely get cluster distribution"""
        if self.cluster_labels is None:
            return {}
        try:
            return {int(k): int(v) for k, v in pd.Series(self.cluster_labels).value_counts().to_dict().items()}
        except Exception:
            return {}

    def build_clustering_data(self):
        """Build clustering data with progress tracking"""
        try:
            self.logger.info("Loading documents...")
            self.load_documents()

            self.logger.info("Preprocessing and vectorizing...")
            self.preprocess_and_vectorize()

            self.logger.info("Performing clustering...")
            self.perform_clustering()

            self.logger.info("Saving results...")
            self.save_results()

            self.logger.info("Clustering process completed successfully")
        except Exception as e:
            self.logger.info(f"Error in build_clustering_data: {str(e)}")
            raise


if __name__ == "__main__":
    folder_path = '/Users/shuwenwang/Documents/dev/uiuc/search-engine/scraped_data/e9f91dc5-9b23-4f4c-bfbf-12fab7338e42'
    data_webpage = '/Users/shuwenwang/Documents/dev/uiuc/search-engine/scraped_data/e9f91dc5-9b23-4f4c-bfbf-12fab7338e42/webpage_graph.json'

    cluster = DocumentClustering("taskId",folder_path, data_webpage)
    try:
        cluster.build_clustering_data()
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
