import numpy as np
import math
from multiprocessing import Pool, cpu_count
from rank_bm25 import BM25


class BM25OkapiWeighted(BM25):
    """
    BM25 Okapi variant that supports document weights.
    Document weights are multiplied with the BM25 score to boost/penalize documents.
    """

    def __init__(self, corpus, doc_weights=None, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        """
        :param corpus: List of tokenized documents
        :param doc_weights: List of weights for each document (default: all 1.0)
        :param k1: Term frequency saturation parameter
        :param b: Length normalization parameter
        :param epsilon: Floor value for IDF scores as epsilon * average_idf
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Initialize doc weights - default to 1.0 if not provided
        if doc_weights is None:
            self.doc_weights = np.ones(len(corpus))
        else:
            if len(doc_weights) != len(corpus):
                raise ValueError(f"Number of document weights ({len(doc_weights)}) must match corpus size ({len(corpus)})")
            self.doc_weights = np.array(doc_weights)

        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        idf_sum = 0
        negative_idfs = []

        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        self.average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * self.average_idf

        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        Calculate weighted BM25 scores for a query against all documents.
        The final score is the product of the BM25 score and document weight.
        """
        base_scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)

        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            base_scores += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                                     (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        # Apply document weights
        weighted_scores = base_scores * self.doc_weights
        return weighted_scores

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate weighted BM25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        base_scores = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        doc_weights_subset = self.doc_weights[doc_ids]

        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            base_scores += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                                     (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        # Apply document weights
        weighted_scores = base_scores * doc_weights_subset
        return weighted_scores.tolist()