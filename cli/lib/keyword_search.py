from lib.search_utils import load_movies, load_stopwords, CACHE_PATH
import string
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict
import os

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # token: [doc_id1, doc_id2]
        self.docmap = {} # map document ID: document
        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        # Want unique tokens
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

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

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

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