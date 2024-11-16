import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import List, Set
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import warnings


class TextSummarizer:
    def __init__(self, max_sentences: int = 10000):
        self.stop_words_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "stopwords.txt"))
        self.load_stopwords(self.stop_words_path)
        self.max_sentences = max_sentences

    def load_stopwords(self, file_path):
        with open(file_path, 'r') as f:
            self.stop_words = set(word.strip().lower() for word in f)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Handle common abbreviations
        text = re.sub(r'(?<=Mr)\.', '@@@', text)
        text = re.sub(r'(?<=Dr)\.', '@@@', text)
        text = re.sub(r'(?<=Mrs)\.', '@@@', text)
        text = re.sub(r'(?<=Ms)\.', '@@@', text)

        # Split on period, exclamation mark, or question mark followed by space and capital letter
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)

        # Restore periods in abbreviations
        sentences = [s.replace('@@@', '.') for s in sentences]

        # Remove empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the text."""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters except periods
        text = re.sub(r'[^\w\s.]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]

        return ' '.join(words)

    def create_similarity_matrix(self, sentences: List[str]) -> csr_matrix:
        """Create similarity matrix between sentences using TF-IDF with sparse matrices."""
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=5000)  # Limit features to control memory
        tfidf_matrix = tfidf.fit_transform([self.preprocess_text(sent)
                                            for sent in sentences])

        # Calculate similarity matrix using sparse operations
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T)

        return similarity_matrix

    def rank_sentences(self, similarity_matrix: csr_matrix) -> List[float]:
        """Implement memory-efficient ranking algorithm using sparse matrices."""
        # Calculate sentence scores based on similarity
        scores = np.array(similarity_matrix.sum(axis=1)).flatten()

        # Normalize scores
        max_score = np.max(scores) if scores.size > 0 else 1
        if max_score != 0:
            scores = scores / max_score

        return scores.tolist()

    def truncate_text(self, sentences: List[str], max_sentences: int) -> List[str]:
        """Intelligently truncate text to handle very long documents."""
        if len(sentences) <= max_sentences:
            return sentences

        # Keep introduction and conclusion
        intro_size = max_sentences // 4
        conclusion_size = max_sentences // 4
        middle_size = max_sentences - (intro_size + conclusion_size)

        return (sentences[:intro_size] +
                sentences[len(sentences) // 2 - middle_size // 2:len(sentences) // 2 + middle_size // 2] +
                sentences[-conclusion_size:])

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary of the text with memory optimization."""
        # Split text into sentences
        sentences = self.split_into_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        # Handle very long texts
        if len(sentences) > self.max_sentences:
            warnings.warn(f"Text has {len(sentences)} sentences. Truncating to {self.max_sentences} sentences.")
            sentences = self.truncate_text(sentences, self.max_sentences)

        try:
            # Create similarity matrix
            similarity_matrix = self.create_similarity_matrix(sentences)

            # Rank sentences
            scores = self.rank_sentences(similarity_matrix)

            # Create sentence-score pairs and sort by score
            ranked_sentences = sorted(
                [(score, i, sentence)
                 for i, (score, sentence) in enumerate(zip(scores, sentences))],
                reverse=True
            )

            # Select top sentences and sort by original position
            selected_sentences = sorted(
                ranked_sentences[:num_sentences],
                key=lambda x: x[1]  # Sort by original position
            )

            # Join sentences to create summary
            summary = ' '.join(sentence for _, _, sentence in selected_sentences)

        except MemoryError:
            warnings.warn("Memory error occurred. Falling back to simple extraction.")
            # Simple fallback: Take first and last sentences, plus evenly spaced sentences in between
            if num_sentences >= len(sentences):
                return text

            indices = np.linspace(0, len(sentences) - 1, num_sentences, dtype=int)
            summary = ' '.join(sentences[i] for i in indices)

        return summary

    def summarize_documents(self, documents: List[str],
                            sentences_per_doc: int = 3) -> List[str]:
        """Summarize multiple documents."""
        return [self.summarize(doc, sentences_per_doc) for doc in documents]

# Example usage
def main():
    documents = [
        """Natural language processing (NLP) is a field of artificial intelligence 
        that focuses on the interaction between computers and human language. 
        It enables computers to understand, interpret, and generate human language 
        in a valuable way. NLP combines computational linguistics, machine learning, 
        and deep learning models. The applications of NLP include machine translation, 
        sentiment analysis, and text summarization.""",

        """Machine learning is a subset of artificial intelligence that provides 
        systems the ability to automatically learn and improve from experience. 
        It focuses on the development of computer programs that can access data 
        and use it to learn for themselves. The process of learning begins with 
        observations or data. Machine learning algorithms build a mathematical 
        model based on sample data."""
    ]

    summarizer = TextSummarizer()
    summaries = summarizer.summarize_documents(documents)

    for i, summary in enumerate(summaries, 1):
        print(f"\nSummary of Document {i}:")
        print(summary)


if __name__ == "__main__":
    main()