import json
import os
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm
import pickle
from typing import List, Dict, Optional
import PyPDF2

from db.database import SearchEngineDatabase
from utils.bm250kapi_weighted import BM25OkapiWeighted
from utils.setup_logging import setup_logging


class BuildTextIndex:
    def __init__(self, docs_dir: str, task_id: str):
        self.docs_dir = docs_dir
        self.documents: List[Dict] = []
        self.bm25 = None
        self.tokenized_docs = None
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.index_path = os.path.join(parent_dir, "index_data", f"{task_id}.pkl")
        self.stop_words_path = os.path.join(os.path.dirname(__file__), "stopwords.txt")
        
        self.logger = setup_logging(name=f"{__name__}", task_id=task_id)

        self.load_stopwords(self.stop_words_path)
        self.task_id = task_id
        self.status = 'idle'
        self.message = ''
        self.start_time = datetime.now()
        self.completion_time = None
        self.is_completed = False
        self.total_files =0
        self.error = ''
        self.failed_files = 0
        self.processed_files =0
        self.current_file =''
        self.progress_percentage = 0

        self.db = SearchEngineDatabase()
        self.db.update_text_index(
            task_id=self.task_id,
            text_index=self
        )

        self.page_data = {}
        self.load_webpage_graph(docs_dir)

    def load_webpage_graph(self, docs_dir: str):
        """Load webpage graph data with both URL mapping and rank information."""
        graph_file = os.path.join(docs_dir, 'webpage_graph.json')
        if os.path.exists(graph_file):
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                for node in graph_data['nodes']:
                    # Create a complete data structure for each page
                    page_info = {
                        'url': node.get('url', ''),
                        'final_rank': node.get('final_rank', 0.0),
                        'initial_rank': node.get('initial_rank', 1.0),
                        'weight': node.get('weight', 1.0),
                        'metadata': node.get('metadata', {}),
                        'title': node.get('title', '')
                    }

                    # Store by path
                    path = node.get('path', '')
                    if not path:
                        path = node.get('relative_path', '')

                    if path:
                        # Normalize path (handle Windows/Unix path differences)
                        normalized_path = path.replace('\\', '/')
                        self.page_data[normalized_path] = page_info

                        # Also store with original path for compatibility
                        self.page_data[path] = page_info

    def get_page_info(self, filepath: str) -> Optional[Dict]:
        """Get URL and rank information for a given filepath."""
        relative_path = os.path.relpath(filepath, self.docs_dir)

        # Try different path formats
        paths_to_try = [
            relative_path,
            relative_path.replace('\\', '/'),  # Normalized path
            os.path.basename(relative_path)  # Just filename
        ]

        for path in paths_to_try:
            if path in self.page_data:
                return self.page_data[path]

        return None

    def get_document_rank_score(self, doc: Dict) -> float:
        """Calculate a combined rank score for a document."""
        if not any(k in doc for k in ['final_rank', 'initial_rank', 'weight']):
            return 1.0  # Default weight if no rank data found

        # Calculate combined score using available metrics
        final_rank = doc.get('final_rank', 0.0)
        initial_rank = doc.get('initial_rank', 1.0)
        weight = doc.get('weight', 1.0)

        # Get content quality indicators from metadata
        metadata = doc.get('metadata', {})
        content_length = metadata.get('content_length', 0)
        code_blocks = metadata.get('code_blocks', 0)
        outbound_links = metadata.get('outbound_links', 0)

        # Normalize content length (assuming average length of 5000)
        normalized_length = min(content_length / 5000, 1.0) if content_length else 0.5

        # Calculate content quality score
        content_score = (
                normalized_length * 0.4 +
                min(code_blocks / 10, 1.0) * 0.3 +
                min(outbound_links / 20, 1.0) * 0.3
        )

        # Combine different ranking factors
        combined_score = (
                final_rank * 0.4 +
                initial_rank * 0.2 +
                weight * 0.2 +
                content_score * 0.2
        )

        return max(combined_score, 0.1)
    def load_stopwords(self, file_path):
        with open(file_path, 'r') as f:
            self.stop_words = set(word.strip().lower() for word in f)

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split()
                if token and token not in self.stop_words and len(token) > 1]

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract meaningful text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Get text
        text = soup.get_text(' ', strip=True)
        return self.clean_text(text)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return self.clean_text(' '.join(text))
        except Exception as e:
            self.logger.info(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return ""

    def process_file(self, filepath: str) -> Optional[Dict]:
        """Process a single file and return its content and metadata."""
        try:
            relative_path = os.path.relpath(filepath, self.docs_dir)
            filename = os.path.basename(filepath)

            # Get page information including URL and rank data
            page_info = self.get_page_info(filepath)
            url = page_info['url'] if page_info else None

            if filepath.lower().endswith('.pdf'):
                text_content = self.extract_text_from_pdf(filepath)
            elif filepath.lower().endswith('.html'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                text_content = self.extract_text_from_html(html_content)
            else:
                return None

            if text_content:
                doc = {
                    'content': text_content,
                    'filepath': relative_path,
                    'filename': filename,
                    'url': url,
                    'type': 'pdf' if filepath.lower().endswith('.pdf') else 'html'
                }

                # Add rank information if available
                if page_info:
                    doc.update({
                        'final_rank': page_info['final_rank'],
                        'initial_rank': page_info['initial_rank'],
                        'weight': page_info['weight'],
                        'metadata': page_info['metadata'],
                        'title': page_info['title']
                    })

                return doc
            return None
        except Exception as e:
            self.logger.error(f"Error processing file {filepath}: {str(e)}")
            return None

    def load_documents(self):
        """Load and process all supported documents."""
        self.logger.info("Loading and processing documents...")
        for root, _, files in os.walk(self.docs_dir):
            for file in tqdm(files):
                if file.lower().endswith(('.html', '.pdf')):
                    filepath = os.path.join(root, file)
                    doc = self.process_file(filepath)
                    if doc:
                        self.documents.append(doc)

        self.logger.info(f"Loaded {len(self.documents)} documents")
    def build_index(self):
        """Build the search index from documents."""
        try:
            self.status = 'building'
            self.message = 'Starting index building process'

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            # Get list of documents
            documents = []

            # Look for both HTML and PDF files
            supported_files = []
            for extension in ['.html', '.pdf']:
                supported_files.extend(list(Path(self.docs_dir).rglob(f'*{extension}')))

            self.total_files = len(supported_files)

            if self.total_files == 0:
                self.status = 'no files found'
                self.db.update_text_index(
                    task_id=self.task_id,
                    text_index=self
                )
                raise ValueError(f"No HTML or PDF files found in directory: {self.docs_dir}")

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            for i, file_path in enumerate(supported_files):
                try:
                    doc = self.process_file(str(file_path))
                    if doc and doc['content'].strip():
                        # Add rank score to document
                        doc['rank_score'] = self.get_document_rank_score(doc)
                        documents.append(doc)
                    self.processed_files = i + 1
                    self.progress_percentage = (i + 1) / self.total_files * 100

                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    self.failed_files += 1
                    self.error = f"Error processing file {file_path}: {e}"

                    self.db.update_text_index(
                        task_id=self.task_id,
                        text_index=self
                    )

            if not documents:
                raise ValueError("No valid documents were processed. All files were either empty or failed to process.")

            # Process documents and build BM25 index
            self.documents = documents
            self.tokenized_docs = [self.tokenize(doc['content']) for doc in documents]

            # Validate tokenized documents
            if not all(self.tokenized_docs):
                raise ValueError("Some documents contained no valid tokens after processing")

            doc_weights = [doc.get('rank_score', 1.0) for doc in documents]
            # self.bm25 = BM25Okapi(corpus=self.tokenized_docs) // not weighted
            self.bm25 = BM25OkapiWeighted(corpus=self.tokenized_docs, doc_weights=doc_weights)

            # Update completion status
            self.completion_time = datetime.now()
            self.status = 'completed'
            self.is_completed = True
            self.current_file = ''

            self.processed_files = self.total_files
            self.message = f'Index built successfully. Processed {len(documents)} out of {self.total_files} files.'
            self.progress_percentage = 100

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            return True

        except Exception as e:
            error_msg = f"Error building index: {str(e)}"
            self.logger.error(error_msg)
            self.status = 'failed'
            self.message = error_msg
            self.error = error_msg
            self.is_completed = True

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )
            raise

    def save_index(self):
        """Save the search index and processed documents to a file."""
        if not self.bm25:
            error_msg = "Index not built. Call build_index() first."
            self.logger.error(error_msg)
            self.status = 'failed'
            self.message = error_msg
            self.error = error_msg
            self.is_completed = True

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            raise ValueError(error_msg)

        try:
            # Update status to saving
            self.status = 'saving'
            self.message = 'Saving index to file...'

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            self.logger.info(f"Saving index to {self.index_path}...")

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            data = {
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
                'bm25': self.bm25
            }

            # Save the index
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)

            self.status = 'completed'
            self.message = 'Index saved successfully'
            self.is_completed = True
            self.completion_time = datetime.now()

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )

            self.logger.info("Index saved successfully")

        except Exception as e:
            error_msg = f"Error saving index: {str(e)}"
            self.logger.error(error_msg)
            self.status = 'failed'
            self.message = error_msg
            self.error = error_msg
            self.is_completed = True

            self.db.update_text_index(
                task_id=self.task_id,
                text_index=self
            )
            raise

    def load_index(self):
        """Load an existing index if available."""
        try:
            # Check database first
            index_info = self.db.get_building_text_index_status(self.task_id)
            if not index_info:
                self.logger.warning(f"No index information found for task {self.task_id}")
                return False

            if not os.path.exists(index_info['index_path']):
                self.logger.warning(f"Index file not found at {index_info['index_path']}")
                return False

            self.logger.info(f"Loading index from {index_info['index_path']}...")
            with open(index_info['index_path'], 'rb') as f:
                data = pickle.load(f)

            self.documents = data['documents']
            self.tokenized_docs = data['tokenized_docs']
            self.bm25 = data['bm25']

            self.logger.info("Index loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False


def main():
    docs_dir = "../scraped_data/4959bee9-842e-4d9b-b9f7-552dfe9fcb9a"  # Your documents directory
    task_id = docs_dir.split("/")[-1]

    # Initialize searcher
    buildTextIndex = BuildTextIndex(docs_dir=docs_dir, task_id=task_id)

    # Check if saved index exists
    if os.path.exists(buildTextIndex.index_path):
        print("Found existing index file.")
        try:
            buildTextIndex.load_index()
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Building new index...")
            buildTextIndex.load_documents()
            buildTextIndex.build_index()
            buildTextIndex.save_index()
    else:
        print("No existing index found. Building new index...")
        buildTextIndex.load_documents()
        buildTextIndex.build_index()
        buildTextIndex.save_index()


if __name__ == "__main__":
    main()
