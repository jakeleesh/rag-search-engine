# __all__ = ['query_answering', 'doc_summarization', 'doc_citations']

from lib.llm import answer_question, summarize_documents, citations_documents
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def query_answering(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:")
    for res in rrf_results:
        print(f"  - {res['title']}")
    rag_results = answer_question(query, rrf_results)
    print("RAG Response:")
    print(rag_results)

def doc_summarization(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query=query, limit=limit, k=60)
    print("Search Results:")
    for res in rrf_results:
        print(f"  - {res['title']}")
    rag_results = summarize_documents(query, rrf_results)
    print("LLM Summary:")
    print(rag_results)

def doc_citations(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query=query, limit=limit, k=60)
    print("Search Results:")
    for res in rrf_results:
        print(f"  - {res['title']}")
    rag_results = citations_documents(query, rrf_results)
    print("LLM Answer:")
    print(rag_results)