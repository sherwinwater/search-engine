import os

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re


class CompanySentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # nltk.download('averaged_perceptron_tagger')
        self.download_nltk_data()

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Define aspect keywords
        self.aspect_keywords = {
            'financial': ['revenue', 'profit', 'earnings', 'growth', 'margin', 'income'],
            'market': ['market', 'competition', 'share', 'industry', 'demand'],
            'management': ['leadership', 'strategy', 'management', 'executive', 'decision'],
            'innovation': ['product', 'technology', 'innovation', 'development', 'research'],
            'future': ['outlook', 'forecast', 'potential', 'future', 'prospect']
        }

    def download_nltk_data(self):
        data_path = os.path.expanduser('~/nltk_data')
        packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

        for package in packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, download_dir=data_path)

    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words]

        return ' '.join(tokens)

    def extract_aspects(self, text):
        """Extract sentences related to different aspects."""
        sentences = sent_tokenize(text)
        aspect_sentences = {aspect: [] for aspect in self.aspect_keywords}

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for aspect, keywords in self.aspect_keywords.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    aspect_sentences[aspect].append(sentence)

        return aspect_sentences

    def analyze_sentiment(self, text):
        """Analyze sentiment score using TextBlob."""
        return TextBlob(text).sentiment.polarity

    def identify_topics(self, texts, n_topics=3):
        """Identify main topics using LDA."""
        # If we have a single document, split it into sentences for better topic modeling
        if len(texts) == 1:
            texts = sent_tokenize(texts[0])

        # Adjust vectorizer parameters for smaller document collections
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,  # Changed from 2 to 1 to handle smaller document sets
            stop_words='english'
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(texts)

            # Only perform LDA if we have enough unique terms
            if doc_term_matrix.shape[1] >= n_topics:
                lda = LatentDirichletAllocation(
                    n_components=min(n_topics, doc_term_matrix.shape[1]),
                    random_state=42
                )
                lda.fit(doc_term_matrix)

                feature_names = vectorizer.get_feature_names_out()
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
                    topics.append(top_words)

                return topics
            else:
                # If we don't have enough terms, return the most common words as topics
                word_freq = doc_term_matrix.sum(axis=0).A1
                feature_names = vectorizer.get_feature_names_out()
                top_words = [feature_names[i] for i in word_freq.argsort()[-n_topics:][::-1]]
                return [top_words]

        except ValueError:
            # Fallback for very short texts: return the most frequent words
            vectorizer = CountVectorizer(stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(texts)
            word_freq = doc_term_matrix.sum(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            top_words = [feature_names[i] for i in word_freq.argsort()[-n_topics:][::-1]]
            return [top_words]

    def analyze_company_sentiment(self, text):
        """Perform complete sentiment analysis for company text."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)

        # Extract aspect-based sentences
        aspect_sentences = self.extract_aspects(text)

        # Analyze sentiment for each aspect
        results = {}
        for aspect, sentences in aspect_sentences.items():
            if sentences:
                avg_sentiment = sum(self.analyze_sentiment(sent) for sent in sentences) / len(sentences)
                results[aspect] = {
                    'sentiment_score': round(avg_sentiment, 2),
                    'sentiment_label': 'positive' if avg_sentiment > 0 else 'negative' if avg_sentiment < 0 else 'neutral',
                    'sentence_count': len(sentences),
                    'example_sentences': sentences[:2]  # Include up to 2 example sentences
                }

        # Overall sentiment
        overall_sentiment = self.analyze_sentiment(text)

        # Identify main topics
        topics = self.identify_topics([processed_text])

        return {
            'overall_sentiment': {
                'score': round(overall_sentiment, 2),
                'label': 'positive' if overall_sentiment > 0 else 'negative' if overall_sentiment < 0 else 'neutral'
            },
            'aspect_analysis': results,
            'main_topics': topics
        }


if __name__ == "__main__":
    with open("news_data/smci_news_20241116_000734.csv","r") as file:
        text = file.read()

    analyzer = CompanySentimentAnalyzer()
    results = analyzer.analyze_company_sentiment(text)

    # Print results in a readable format
    print("\nOverall Sentiment Analysis:")
    print(f"Score: {results['overall_sentiment']['score']}")
    print(f"Label: {results['overall_sentiment']['label']}")

    print("\nAspect-Based Sentiment Analysis:")
    for aspect, data in results['aspect_analysis'].items():
        print(f"\n{aspect.capitalize()}:")
        print(f"Sentiment Score: {data['sentiment_score']}")
        print(f"Sentiment Label: {data['sentiment_label']}")
        print(f"Number of mentions: {data['sentence_count']}")
        if data['example_sentences']:
            print("Example sentences:")
            for sent in data['example_sentences']:
                print(f"- {sent}")

    print("\nMain Topics Identified:")
    for i, topic_words in enumerate(results['main_topics']):
        print(f"Topic {i + 1}: {', '.join(topic_words)}")
