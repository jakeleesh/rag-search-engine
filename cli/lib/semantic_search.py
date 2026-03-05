from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.search_utils import load_movies
import re

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {} # doc_id: documents
        self.embeddings_path = Path("cache/movie_embeddings.npy")

    # documents list of dicts, each representing a movie
    def build_embeddings(self, documents: list[dict]):
        # Given documents so can store
        self.documents = documents
        # Reset because could be leftovers if IDs are different
        self.document_map = {}
        # Need list because encode want a list to use
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            # Updating documents, make sure same number
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self,text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        # Only care about first element because only passing in one input
        return self.model.encode([text])[0]
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qry_emb = self.generate_embedding(query)

        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(qry_emb, doc_emb)
            similarities.append((_similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append(
                {'score': sc,
                 'title': doc['title'],
                 'description': doc['description']
                }
            )
        return res
    
def semantic_chunking(text, overlap=0, max_chunk_size=4):
    # Split the text at the whitespace that follows sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks

def chunk_text_semantic(text, overlap=0, max_chunk_size=4):
    chunks = semantic_chunking(text, overlap, max_chunk_size)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def fixed_sized_chunking(text, overlap, chunk_size=200):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    # Start at 0 and jump by chunk_size
    # words[0:200], words[200:400]
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        # Edge Case: last chunk only has overlap
        # Don't want because chunk doesn't help, already enconded
        if len(chunk_words) <= overlap:
            break
        # split into a list, want to put it back into text
        chunks.append(" ".join(chunk_words))
    return chunks

def chunk_text(text, overlap, chunk_size=200):
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    
def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res['description'][:100])
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding
    
def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")