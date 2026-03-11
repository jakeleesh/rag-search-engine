from lib.search_utils import load_movies, load_stopwords, CACHE_PATH, BM25_K1, BM25_B
import math
import string
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # token: [doc_id1, doc_id2]
        self.docmap = {} # map document ID: document
        # Not count al across all documents, need one Counter per document
        self.term_frequencies = defaultdict(Counter)

        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'
        # Length of every single document in tokens
        self.doc_lengths = {}
        self.doc_lengths_path = CACHE_PATH/'doc_lengths.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        # Want unique tokens
        for token in set(tokens):
            self.index[token].add(doc_id)
        # Will count all frequency of different tokens
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        if len(lengths) == 0:
            return 0.0
        ttl = 0
        for l in lengths:
            ttl += l
        return ttl / len(lengths)

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docmap)
        # self.index is set, put a token and get list of all doc_id that have that term
        term_doc_count = len(self.index[token])

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docmap)
        # self.index is set, put a token and get list of all doc_id that have that term
        # How many document contain that term
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def bm25_search(self, query, limit=5):
        query_tokens = tokenize_text(query)
        # dict to aggregate
        scores = {}
        for doc_id in self.docmap:
            score = 0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            # Got full score
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = sorted_scores[:limit]
        format_results = []
        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            format_results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "score": score,
                    "description": self.docmap[doc_id]['description']
                }
            )
        return format_results

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        # pickle saves files in small formats
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def bm25_search(query):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tfidf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def clean_text(text):
    text = text.lower()
    # Mapp all characters in third argument to None
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text):
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        tok = tok.strip('\n')
        # if tok says not None like empty string
        if tok and tok not in stopwords:
            return True
        return False
    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res

def has_matching_token(query_tokens, movie_tokens):
    """Compare query to document. Given the two, are they a match or not."""
    # Have every unique combination
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False

def search_command(query, n_results=5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    # Keep track of ones already seen cause don't want duplicates
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        matching_doc_ids = idx.get_documents(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res